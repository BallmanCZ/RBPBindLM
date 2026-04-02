# -*- coding: utf-8 -*-
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

from get_dataset_gena_lm import GENA_MODEL_NAME, load_split, pad_padded_sequence_features
from ksu_undersampling import ksu_undersampling
from metrics import calculate_performace
from multi_model_transformer_ca import build_transformer_cross_attention_model
from stochastic_gates import StochasticGateFeatureSelector

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


STOP_AFTER_FIRST_FOLD = False

PAPER_SN_SP_TARGETS = {
    "1_PARCLIP_AGO1234_hg19": {"SN": 0.592, "SP": 0.7873},
}

TARGET_WEIGHT_SN = 3.0
TARGET_WEIGHT_SP = 1.0
TOPK_BY_COST = 20

SEQ_MODEL_NAME = GENA_MODEL_NAME
SEQ_BATCH_SIZE = 64
SEQ_MAX_LENGTH = 512

SG_TARGET_DIM = 300
SG_EPOCHS = 40
SG_BATCH_SIZE = 512
SG_HIDDEN_DIM = 256
SG_LEARNING_RATE = 1e-3

MODEL_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-4


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label_stats(y):
    y = np.asarray(y).astype(int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    tot = pos + neg
    pos_pct = 100.0 * pos / tot if tot else 0.0
    neg_pct = 100.0 * neg / tot if tot else 0.0
    ir = (neg / pos) if pos > 0 else float("inf")
    return {
        "total": tot,
        "pos": pos,
        "neg": neg,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "ratio_str": f"{pos}:{neg}",
        "imbalance_ratio_neg_over_pos": ir,
    }


def _load_train_data(file_name, kmer, seq_kernel, struc_kernel):
    print("[Data] loading training split ...")
    train_dir = PROJECT_ROOT / "datasets" / "clip" / file_name / "30000" / "training_sample_0"
    return load_split(
        str(train_dir),
        kmer=kmer,
        seq_kernel=seq_kernel,
        struc_kernel=struc_kernel,
        seq_model_name=SEQ_MODEL_NAME,
        seq_batch_size=SEQ_BATCH_SIZE,
        seq_max_length=SEQ_MAX_LENGTH,
    )


def _load_test_data(file_name, kmer, seq_kernel, struc_kernel):
    print("[Data] loading test split ...")
    test_dir = PROJECT_ROOT / "datasets" / "clip" / file_name / "30000" / "test_sample_0"
    return load_split(
        str(test_dir),
        kmer=kmer,
        seq_kernel=seq_kernel,
        struc_kernel=struc_kernel,
        seq_model_name=SEQ_MODEL_NAME,
        seq_batch_size=SEQ_BATCH_SIZE,
        seq_max_length=SEQ_MAX_LENGTH,
    )


def _align_train_test_seq_lengths(train_split, test_split):
    max_len = max(train_split["seq_embeddings"].shape[1], test_split["seq_embeddings"].shape[1])
    train_split["seq_embeddings"], train_split["seq_mask"] = pad_padded_sequence_features(
        train_split["seq_embeddings"],
        train_split["seq_mask"],
        max_len,
    )
    test_split["seq_embeddings"], test_split["seq_mask"] = pad_padded_sequence_features(
        test_split["seq_embeddings"],
        test_split["seq_mask"],
        max_len,
    )


def _deterministic_ksu(seq_pooled, seq_embeddings, seq_mask, struc_array, labels):
    """
    Keep the original KSU stage, but run it on mean-pooled 768-d GENA-LM features.
    This preserves the original "sample by sequence branch first, then align the
    structure branch by indices" workflow while keeping the computation tractable.
    """
    seq_resampled, label_resampled, indices = ksu_undersampling(
        seq_pooled,
        labels,
        random_state=SEED,
    )
    indices = np.asarray(indices)
    order = np.argsort(indices, kind="mergesort")
    indices = indices[order]
    seq_resampled = seq_resampled[order]
    label_resampled = np.asarray(label_resampled)[order]
    return {
        "seq_pooled": seq_resampled.astype(np.float32),
        "seq_embeddings": seq_embeddings[indices],
        "seq_mask": seq_mask[indices],
        "struc_features": struc_array[indices],
        "labels": label_resampled.astype(int),
        "indices": indices,
    }


def _sn_sp_mcc_for_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    pos = y_true == 1
    neg = ~pos
    tp = int(np.sum(y_pred[pos] == 1))
    fn = int(np.sum(y_pred[pos] == 0))
    tn = int(np.sum(y_pred[neg] == 0))
    fp = int(np.sum(y_pred[neg] == 1))
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if denom == 0 else (tp * tn - fp * fn) / np.sqrt(denom)
    return sn, sp, mcc


def _choose_threshold_by_targets(y_true, y_prob, sn_target, sp_target, w_sn=1.0, w_sp=1.0, topk=TOPK_BY_COST):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    costs = []
    for i, thr in enumerate(thresholds):
        sn = tpr[i]
        sp = 1.0 - fpr[i]
        cost = w_sn * abs(sn - sn_target) + w_sp * abs(sp - sp_target)
        costs.append((cost, float(thr), sn, sp))
    costs.sort(key=lambda x: x[0])
    candidates = costs[: max(1, min(topk, len(costs)))]

    best = None
    best_mcc = -1.0
    for _, thr, _, _ in candidates:
        sn, sp, mcc = _sn_sp_mcc_for_threshold(y_true, y_prob, thr)
        if mcc > best_mcc:
            best_mcc = mcc
            best = (thr, sn, sp, mcc)
    return best


def _choose_threshold_by_mcc(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    del fpr, tpr
    thresholds = np.clip(thresholds, 0.0, 1.0)
    best = (0.5, 0.0, 0.0, -1.0)
    for thr in thresholds:
        sn, sp, mcc = _sn_sp_mcc_for_threshold(y_true, y_prob, thr)
        if mcc > best[3]:
            best = (float(thr), sn, sp, mcc)
    return best


def _eval_with_metrics(y_true_raw, y_prob, threshold=0.5):
    y_pred = (y_prob >= float(threshold)).astype(int)
    y_true_oh = to_categorical(y_true_raw, num_classes=2)
    p, sn, sp, acc, ba, auc_v, mcc, f1 = calculate_performace(
        len(y_pred),
        y_pred,
        y_prob,
        y_true_oh[:, -1],
    )
    return p, sn, sp, acc, ba, auc_v, mcc, f1


def _print_fold_metrics(fold_idx, p, sn, sp, acc, balanced_acc, auc_roc, auc_calc, mcc, f1, dur, thr):
    msg = (
        f"[Fold {fold_idx}] "
        f"P:{p:.4f}\tSN:{sn:.4f}\tSP:{sp:.4f}\tACC:{acc:.4f}\t"
        f"balanced_acc:{balanced_acc:.4f}\tAUC(roc):{auc_roc:.4f}\t"
        f"AUC(calc):{auc_calc:.4f}\tMCC:{mcc:.4f}\tF1:{f1:.4f}\t"
        f"thr:{thr:.4f}\ttrain_time_sec:{dur:.2f}"
    )
    print(msg)
    sys.stdout.flush()
    return msg


def _plot_and_save_fold_roc(save_dir: Path, fpr, tpr, auc_value, fold_idx, title_prefix):
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"Fold {fold_idx} ROC (AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - Fold {fold_idx}")
    plt.legend(loc="lower right")
    out_path = save_dir / f"roc_fold{fold_idx}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _plot_and_save_mean_roc(save_dir: Path, mean_fpr, tprs, aucs, title_prefix):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC={mean_auc:.4f}+/-{std_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - Mean ROC")
    plt.legend(loc="lower right")
    out_path = save_dir / "roc_mean.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out_path), float(mean_auc), float(std_auc)


def _build_selector(input_dim):
    return StochasticGateFeatureSelector(
        input_dim=int(input_dim),
        target_dim=SG_TARGET_DIM,
        hidden_dim=SG_HIDDEN_DIM,
        max_epochs=SG_EPOCHS,
        batch_size=SG_BATCH_SIZE,
        learning_rate=SG_LEARNING_RATE,
        random_state=SEED,
        verbose=False,
    )


def PreRBP_10fold_and_test(file_name, kmer, seq_kernel, struc_kernel, batch_size=32, epochs=30, n_splits=10):
    results_dir = _ensure_dir(PROJECT_ROOT / "results")
    run_tag = f"{file_name}_GENALM_SG{SG_TARGET_DIM}_TxCA_k{kmer}_seq{seq_kernel}_struc{struc_kernel}"
    cv_dir = _ensure_dir(results_dir / f"{run_tag}_cv")
    test_dir = _ensure_dir(results_dir / f"{run_tag}_test")

    train_split = _load_train_data(file_name, kmer, seq_kernel, struc_kernel)
    test_split = _load_test_data(file_name, kmer, seq_kernel, struc_kernel)
    _align_train_test_seq_lengths(train_split, test_split)

    resampled_train = _deterministic_ksu(
        train_split["seq_pooled"],
        train_split["seq_embeddings"],
        train_split["seq_mask"],
        train_split["struc_features"],
        train_split["labels"],
    )

    stats_before = _label_stats(train_split["labels"])
    stats_after = _label_stats(resampled_train["labels"])

    print("\n[Sampling Stats] BEFORE resampling:")
    print(
        f"  total={stats_before['total']}, pos={stats_before['pos']} ({stats_before['pos_pct']:.2f}%), "
        f"neg={stats_before['neg']} ({stats_before['neg_pct']:.2f}%), ratio={stats_before['ratio_str']} "
        f"(Neg/Pos={stats_before['imbalance_ratio_neg_over_pos']:.3f})"
    )
    print("[Sampling Stats] AFTER resampling (KSU):")
    print(
        f"  total={stats_after['total']}, pos={stats_after['pos']} ({stats_after['pos_pct']:.2f}%), "
        f"neg={stats_after['neg']} ({stats_after['neg_pct']:.2f}%), ratio={stats_after['ratio_str']} "
        f"(Neg/Pos={stats_after['imbalance_ratio_neg_over_pos']:.3f})\n"
    )

    with open(results_dir / f"{run_tag}_sampling_stats.json", "w", encoding="utf-8") as fjs:
        json.dump({"before": stats_before, "after": stats_after}, fjs, ensure_ascii=False, indent=2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    mean_fpr = np.linspace(0, 1, 100)
    fold_tprs, fold_aucs, fold_metrics = [], [], []
    best_threshold_from_cv = 0.5

    compared_path = results_dir / f"GENALM_SG{SG_TARGET_DIM}_TxCA_compared.txt"

    use_targets = file_name in PAPER_SN_SP_TARGETS
    if use_targets:
        sn_t = PAPER_SN_SP_TARGETS[file_name]["SN"]
        sp_t = PAPER_SN_SP_TARGETS[file_name]["SP"]
        print(
            f"[Thresholding] Use paper targets for '{file_name}': "
            f"SN={sn_t}, SP={sp_t}, weights SN:{TARGET_WEIGHT_SN}, SP:{TARGET_WEIGHT_SP}"
        )
    else:
        print("[Thresholding] Paper targets not found -> fallback to max MCC threshold.")

    seq_embeddings = resampled_train["seq_embeddings"]
    seq_pooled = resampled_train["seq_pooled"]
    struc_features = resampled_train["struc_features"]
    y_resampled = resampled_train["labels"]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(seq_pooled, y_resampled), start=1):
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        x_tr_pooled = seq_pooled[train_idx]
        x_tr_seq_full = seq_embeddings[train_idx]
        x_tr_str = struc_features[train_idx].astype(np.float32)
        y_tr_raw = y_resampled[train_idx]

        x_val_pooled = seq_pooled[val_idx]
        x_val_seq_full = seq_embeddings[val_idx]
        x_val_str = struc_features[val_idx].astype(np.float32)
        y_val_raw = y_resampled[val_idx]

        selector = _build_selector(x_tr_pooled.shape[1])
        selector.fit(x_tr_pooled, y_tr_raw)
        x_tr_seq = selector.transform(x_tr_seq_full).astype(np.float32)
        x_val_seq = selector.transform(x_val_seq_full).astype(np.float32)

        with open(cv_dir / f"fold{fold_idx}_sg_dims.json", "w", encoding="utf-8") as fs:
            json.dump(
                {
                    "selected_dims": selector.selected_idx_.tolist(),
                    "gate_probabilities": selector.gate_probabilities_.tolist(),
                },
                fs,
                ensure_ascii=False,
                indent=2,
            )

        y_tr = to_categorical(y_tr_raw, num_classes=2)
        y_val = to_categorical(y_val_raw, num_classes=2)

        model = build_transformer_cross_attention_model(
            seq_len=x_tr_seq.shape[1],
            seq_dim=x_tr_seq.shape[2],
            struc_len=x_tr_str.shape[1],
            struc_dim=x_tr_str.shape[2],
            d_model=MODEL_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=["accuracy"],
        )

        start_t = time.time()
        model.fit(
            x=[x_tr_seq, x_tr_str],
            y=y_tr,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([x_val_seq, x_val_str], y_val),
            verbose=1,
            shuffle=False,
        )
        dur = time.time() - start_t
        print(f"[Fold {fold_idx}] train_time_sec={dur:.2f}")

        prob_val = model.predict([x_val_seq, x_val_str], verbose=0)
        y_prob_val = prob_val[:, -1]

        fpr, tpr, _ = roc_curve(y_val_raw, y_prob_val, pos_label=1)
        auc_roc = roc_auc_score(y_val_raw, y_prob_val)

        if use_targets:
            thr_best, sn_at_thr, sp_at_thr, mcc_at_thr = _choose_threshold_by_targets(
                y_val_raw,
                y_prob_val,
                sn_t,
                sp_t,
                w_sn=TARGET_WEIGHT_SN,
                w_sp=TARGET_WEIGHT_SP,
            )
            print(
                f"[Fold {fold_idx}] chosen thr by targets -> {thr_best:.4f} "
                f"(SN={sn_at_thr:.4f}, SP={sp_at_thr:.4f}, MCC={mcc_at_thr:.4f})"
            )
        else:
            thr_best, sn_at_thr, sp_at_thr, mcc_at_thr = _choose_threshold_by_mcc(y_val_raw, y_prob_val)
            print(
                f"[Fold {fold_idx}] chosen thr by max MCC -> {thr_best:.4f} "
                f"(SN={sn_at_thr:.4f}, SP={sp_at_thr:.4f}, MCC={mcc_at_thr:.4f})"
            )

        best_threshold_from_cv = float(thr_best)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        fold_tprs.append(tpr_interp)
        fold_aucs.append(auc_roc)

        p, sn, sp, acc, ba, auc_calc, mcc, f1 = _eval_with_metrics(
            y_val_raw,
            y_prob_val,
            threshold=thr_best,
        )

        _plot_and_save_fold_roc(cv_dir, fpr, tpr, auc_roc, fold_idx, f"{run_tag} CV")
        _print_fold_metrics(fold_idx, p, sn, sp, acc, ba, auc_roc, auc_calc, mcc, f1, dur, thr_best)

        with open(cv_dir / "per_fold_metrics.txt", "a", encoding="utf-8") as fw:
            fw.write(
                f"Fold {fold_idx}\tP:{p}\tSN:{sn}\tSP:{sp}\tACC:{acc}\t"
                f"balanced_acc:{ba}\tAUC(roc):{auc_roc}\tAUC(calc):{auc_calc}\t"
                f"MCC:{mcc}\tF1:{f1}\tthr:{thr_best}\ttrain_time_sec:{dur:.2f}\n"
            )

        with open(compared_path, "a", encoding="utf-8") as fw:
            fw.write(
                f"[FOLD {fold_idx}] {run_tag}\t"
                f"P:{p:.4f}\tSN:{sn:.4f}\tSP:{sp:.4f}\tACC:{acc:.4f}\t"
                f"balanced_acc:{ba:.4f}\tAUC(roc):{auc_roc:.4f}\tAUC(calc):{auc_calc:.4f}\t"
                f"MCC:{mcc:.4f}\tF1:{f1:.4f}\tthr:{thr_best:.4f}\ttrain_time_sec:{dur:.2f}\n"
            )

        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_time_sec": round(dur, 2),
                "P": float(p),
                "SN": float(sn),
                "SP": float(sp),
                "ACC": float(acc),
                "balanced_acc": float(ba),
                "AUC_roc": float(auc_roc),
                "AUC_calc": float(auc_calc),
                "MCC": float(mcc),
                "F1": float(f1),
                "thr": float(thr_best),
            }
        )

        del x_val_pooled

        if STOP_AFTER_FIRST_FOLD:
            break

    roc_mean_path, mean_auc, std_auc = _plot_and_save_mean_roc(
        cv_dir,
        mean_fpr,
        np.array(fold_tprs),
        np.array(fold_aucs),
        f"{run_tag} CV",
    )

    def _mean_std(key):
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        return float(np.mean(vals)), float(np.std(vals))

    mean_P, std_P = _mean_std("P")
    mean_SN, std_SN = _mean_std("SN")
    mean_SP, std_SP = _mean_std("SP")
    mean_ACC, std_ACC = _mean_std("ACC")
    mean_BA, std_BA = _mean_std("balanced_acc")
    mean_AUC_calc, std_AUC_calc = _mean_std("AUC_calc")
    mean_MCC, std_MCC = _mean_std("MCC")
    mean_F1, std_F1 = _mean_std("F1")
    mean_thr, std_thr = _mean_std("thr")

    best_threshold_from_cv = float(mean_thr)
    print("\n==== CV Summary ====")
    print(f"P: {mean_P:.4f} +/- {std_P:.4f}")
    print(f"SN: {mean_SN:.4f} +/- {std_SN:.4f}")
    print(f"SP: {mean_SP:.4f} +/- {std_SP:.4f}")
    print(f"ACC: {mean_ACC:.4f} +/- {std_ACC:.4f}")
    print(f"balanced_acc: {mean_BA:.4f} +/- {std_BA:.4f}")
    print(f"AUC(roc): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"AUC(calc): {mean_AUC_calc:.4f} +/- {std_AUC_calc:.4f}")
    print(f"MCC: {mean_MCC:.4f} +/- {std_MCC:.4f}")
    print(f"F1: {mean_F1:.4f} +/- {std_F1:.4f}")
    print(f"thr: {mean_thr:.4f} +/- {std_thr:.4f}")
    sys.stdout.flush()

    with open(cv_dir / "summary.txt", "w", encoding="utf-8") as fw:
        fw.write("==== CV Summary ====\n")
        fw.write(f"P: {mean_P:.4f} +/- {std_P:.4f}\n")
        fw.write(f"SN: {mean_SN:.4f} +/- {std_SN:.4f}\n")
        fw.write(f"SP: {mean_SP:.4f} +/- {std_SP:.4f}\n")
        fw.write(f"ACC: {mean_ACC:.4f} +/- {std_ACC:.4f}\n")
        fw.write(f"balanced_acc: {mean_BA:.4f} +/- {std_BA:.4f}\n")
        fw.write(f"AUC(roc): {mean_auc:.4f} +/- {std_auc:.4f}\n")
        fw.write(f"AUC(calc): {mean_AUC_calc:.4f} +/- {std_AUC_calc:.4f}\n")
        fw.write(f"MCC: {mean_MCC:.4f} +/- {std_MCC:.4f}\n")
        fw.write(f"F1: {mean_F1:.4f} +/- {std_F1:.4f}\n")
        fw.write(f"thr: {mean_thr:.4f} +/- {std_thr:.4f}\n")
        fw.write(f"Mean ROC png: {roc_mean_path}\n")

    with open(cv_dir / "per_fold_metrics.csv", "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(fold_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(fold_metrics)

    cv_json = {
        "seed": SEED,
        "n_splits": n_splits,
        "only_first_fold": STOP_AFTER_FIRST_FOLD,
        "sequence_encoder": {
            "name": SEQ_MODEL_NAME,
            "batch_size": SEQ_BATCH_SIZE,
            "max_length": SEQ_MAX_LENGTH,
        },
        "stochastic_gates": {
            "target_dim": SG_TARGET_DIM,
            "epochs": SG_EPOCHS,
            "batch_size": SG_BATCH_SIZE,
            "hidden_dim": SG_HIDDEN_DIM,
            "learning_rate": SG_LEARNING_RATE,
        },
        "transformer_classifier": {
            "d_model": MODEL_DIM,
            "num_heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "dropout": DROPOUT,
        },
        "mean": {
            "P": mean_P,
            "SN": mean_SN,
            "SP": mean_SP,
            "ACC": mean_ACC,
            "balanced_acc": mean_BA,
            "AUC_roc": mean_auc,
            "AUC_calc": mean_AUC_calc,
            "MCC": mean_MCC,
            "F1": mean_F1,
            "thr": mean_thr,
        },
        "std": {
            "P": std_P,
            "SN": std_SN,
            "SP": std_SP,
            "ACC": std_ACC,
            "balanced_acc": std_BA,
            "AUC_roc": std_auc,
            "AUC_calc": std_AUC_calc,
            "MCC": std_MCC,
            "F1": std_F1,
            "thr": std_thr,
        },
        "mean_roc_png": roc_mean_path,
    }
    with open(cv_dir / "summary.json", "w", encoding="utf-8") as fj:
        json.dump(cv_json, fj, ensure_ascii=False, indent=2)

    print("\n===== Independent Test (train on ALL resampled training data) =====")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    final_selector = _build_selector(seq_pooled.shape[1])
    final_selector.fit(seq_pooled, y_resampled)

    with open(test_dir / "final_sg_dims.json", "w", encoding="utf-8") as fs:
        json.dump(
            {
                "selected_dims": final_selector.selected_idx_.tolist(),
                "gate_probabilities": final_selector.gate_probabilities_.tolist(),
            },
            fs,
            ensure_ascii=False,
            indent=2,
        )

    x_tr_seq = final_selector.transform(seq_embeddings).astype(np.float32)
    x_tr_str = struc_features.astype(np.float32)
    y_tr_raw = y_resampled
    y_tr = to_categorical(y_tr_raw, num_classes=2)

    test_seq_reduced = final_selector.transform(test_split["seq_embeddings"]).astype(np.float32)
    test_struc = test_split["struc_features"].astype(np.float32)
    test_y_raw = test_split["labels"].astype(int)

    final_model = build_transformer_cross_attention_model(
        seq_len=x_tr_seq.shape[1],
        seq_dim=x_tr_seq.shape[2],
        struc_len=x_tr_str.shape[1],
        struc_dim=x_tr_str.shape[2],
        d_model=MODEL_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dropout=DROPOUT,
    )
    final_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

    start_t = time.time()
    final_model.fit(
        x=[x_tr_seq, x_tr_str],
        y=y_tr,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=False,
    )
    train_time = time.time() - start_t
    print(f"[Final] train_time_sec={train_time:.2f}")

    prob_test = final_model.predict([test_seq_reduced, test_struc], verbose=0)
    y_prob_test = prob_test[:, -1]
    p, sn, sp, acc, ba, auc_v, mcc, f1 = _eval_with_metrics(
        test_y_raw,
        y_prob_test,
        threshold=best_threshold_from_cv,
    )

    print(
        f"[TEST] {run_tag}\tP:{p:.4f}\tSN:{sn:.4f}\tSP:{sp:.4f}\tACC:{acc:.4f}\t"
        f"balanced_acc:{ba:.4f}\tAUC:{auc_v:.4f}\tMCC:{mcc:.4f}\tF1:{f1:.4f}\t"
        f"thr(CV):{best_threshold_from_cv:.4f}"
    )
    sys.stdout.flush()

    with open(test_dir / "test_metrics.txt", "w", encoding="utf-8") as fw:
        fw.write(
            f"{run_tag}\tP:{p}\tSN:{sn}\tSP:{sp}\tACC:{acc}\t"
            f"balanced_acc:{ba}\tAUC:{auc_v}\tMCC:{mcc}\tF1:{f1}\t"
            f"thr(CV):{best_threshold_from_cv}\n"
        )

    fpr_t, tpr_t, _ = roc_curve(test_y_raw, y_prob_test, pos_label=1)
    test_auc_roc = roc_auc_score(test_y_raw, y_prob_test)
    plt.figure()
    plt.plot(fpr_t, tpr_t, lw=2, label=f"Test ROC (AUC={test_auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{run_tag} - Independent Test ROC")
    plt.legend(loc="lower right")
    test_roc_path = test_dir / "test_roc.png"
    plt.savefig(test_roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    meta = {
        "SEED": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "train_time_sec_final": round(train_time, 2),
        "cv": cv_json,
        "independent_test": {
            "P": float(p),
            "SN": float(sn),
            "SP": float(sp),
            "ACC": float(acc),
            "balanced_acc": float(ba),
            "AUC": float(auc_v),
            "MCC": float(mcc),
            "F1": float(f1),
            "roc_png": str(test_roc_path),
            "thr_from_cv": float(best_threshold_from_cv),
        },
        "sequence_encoder": {
            "name": SEQ_MODEL_NAME,
            "target_dim_after_sg": SG_TARGET_DIM,
        },
        "env": {
            k: os.environ.get(k)
            for k in [
                "PYTHONHASHSEED",
                "TF_ENABLE_ONEDNN_OPTS",
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_DETERMINISTIC",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            ]
            if os.environ.get(k) is not None
        },
    }
    with open(results_dir / f"{run_tag}_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(compared_path, "a", encoding="utf-8") as fw:
        fw.write(
            f"[CV mean+/-std] {run_tag}\t"
            f"P:{mean_P:.4f}+/-{std_P:.4f}\tSN:{mean_SN:.4f}+/-{std_SN:.4f}\t"
            f"SP:{mean_SP:.4f}+/-{std_SP:.4f}\tACC:{mean_ACC:.4f}+/-{std_ACC:.4f}\t"
            f"balanced_acc:{mean_BA:.4f}+/-{std_BA:.4f}\tAUC(roc):{mean_auc:.4f}+/-{std_auc:.4f}\t"
            f"MCC:{mean_MCC:.4f}+/-{std_MCC:.4f}\tF1:{mean_F1:.4f}+/-{std_F1:.4f}\t"
            f"thr:{mean_thr:.4f}+/-{std_thr:.4f}\n"
        )
        fw.write(
            f"[TEST] {run_tag}\tP:{p:.4f}\tSN:{sn:.4f}\tSP:{sp:.4f}\tACC:{acc:.4f}\t"
            f"balanced_acc:{ba:.4f}\tAUC:{auc_v:.4f}\tMCC:{mcc:.4f}\tF1:{f1:.4f}\t"
            f"thr(CV):{best_threshold_from_cv:.4f}\n"
        )


if __name__ == "__main__":
    dataset = "1_PARCLIP_AGO1234_hg19"
    kmer = 2
    seq_kernel = 4
    struc_kernel = 4

    PreRBP_10fold_and_test(dataset, kmer, seq_kernel, struc_kernel)
