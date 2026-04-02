import gzip
import math
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


GENA_MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t-multi"

_gena_tokenizer = None
_gena_model = None
_gena_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_gena_lm(model_name: str = GENA_MODEL_NAME):
    global _gena_tokenizer, _gena_model

    if _gena_tokenizer is not None and _gena_model is not None:
        return _gena_tokenizer, _gena_model

    print(f"[GENA-LM] loading model={model_name} on device={_gena_device} ...")
    _gena_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _gena_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _gena_model.to(_gena_device)
    _gena_model.eval()
    print("[GENA-LM] model ready.")
    return _gena_tokenizer, _gena_model


def _clean_seq_for_dna(seq: str) -> str:
    return seq.replace("U", "T").upper()


def read_seq_file(seq_file: str) -> tuple[list[str], list[str]]:
    seq_list = []
    label_list = []
    neg_num = 0
    pos_num = 0
    seq = ""
    with gzip.open(seq_file, "rt") as fp:
        for line in fp:
            if line[0] == ">":
                label = line[-2]
                if label == "0":
                    neg_num += 1
                else:
                    pos_num += 1
                label_list.append(label)
                if len(seq):
                    seq_list.append(seq)
                seq = ""
            else:
                seq = seq + str(line[:-1])
        if len(seq):
            seq_list.append(seq)
    print(f"[read_seq_file] neg={neg_num}, pos={pos_num}, total={len(seq_list)}")
    return seq_list, label_list


def read_stru_file(stru_file: str) -> list[str]:
    stru_list = []
    with gzip.open(stru_file, "rt") as fp:
        for line in fp:
            if line and line[0] != ">":
                stru_list.append(line[:-1])
    return stru_list


def buildmapper(alphabet: list[str], degree: int) -> dict[str, list[float]]:
    mapper = [""]
    length = degree
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1
        length -= 1

    code = np.eye(len(mapper), dtype=np.float32)
    encoder: dict[str, list[float]] = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i, :])

    number = int(math.pow(len(alphabet), degree))
    encoder["N"] = [1.0 / number] * number
    return encoder


def get_seq_degree(seq: str, degree: int, motif_len: int) -> list[str]:
    half_len = int(motif_len / 2)
    length = len(seq)
    row = length + motif_len - degree + 1
    seqdata = []
    for _ in range(half_len):
        seqdata.append("N")

    for i in range(length - degree + 1):
        seqdata.append(seq[i : i + degree])

    for _ in range(row - half_len, row):
        seqdata.append("N")
    return seqdata


def embed(seq: list[str], mapper: dict[str, list[float]]) -> list[list[float]]:
    mat = []
    for element in seq:
        if element in mapper:
            mat.append(mapper.get(element))
        elif "N" in element:
            mat.append(mapper.get("N"))
        else:
            raise ValueError(f"Unexpected structure token: {element}")
    return mat


def get_kmer_fea(seq: str, degree: int, motif_len: int, alphabet: list[str]) -> list[list[float]]:
    encoder = buildmapper(alphabet, degree)
    seq_data = get_seq_degree(seq, degree, motif_len)
    return embed(seq_data, encoder)


def build_structure_features(
    stru_list: list[str],
    kmer: int,
    struc_kernel: int,
) -> np.ndarray:
    stru_alphabet = ["F", "T", "I", "H", "M", "S"]
    stru_fea = []
    iterator = tqdm(stru_list, desc="Structure k-mer onehot", unit="seq")
    for stru in iterator:
        stru_fea.append(get_kmer_fea(stru, kmer, struc_kernel, stru_alphabet))
    return np.asarray(stru_fea, dtype=np.float32)


def extract_gena_token_embeddings(
    seq_list: list[str],
    model_name: str = GENA_MODEL_NAME,
    batch_size: int = 64,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer, model = _load_gena_lm(model_name=model_name)

    embeddings_list: list[np.ndarray] = []
    lengths: list[int] = []
    hidden_dim = None

    iterator = range(0, len(seq_list), batch_size)
    iterator = tqdm(iterator, desc="GENA-LM embedding", unit="batch")

    for start in iterator:
        end = min(start + batch_size, len(seq_list))
        clean_batch = [_clean_seq_for_dna(s) for s in seq_list[start:end]]
        inputs = tokenizer(
            clean_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(_gena_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            pad_token_id = tokenizer.pad_token_id
            attention_mask = (inputs["input_ids"] != pad_token_id).long()

        batch_size_now = hidden.shape[0]
        hidden_dim = hidden.shape[-1]
        for i in range(batch_size_now):
            valid_len = int(attention_mask[i].sum().item())
            start_idx = 1 if valid_len >= 2 else 0
            end_idx = max(start_idx + 1, valid_len - 1)
            seq_hidden = hidden[i, start_idx:end_idx, :]
            seq_hidden = seq_hidden.detach().cpu().numpy().astype(np.float16)
            embeddings_list.append(seq_hidden)
            lengths.append(seq_hidden.shape[0])

    if hidden_dim is None:
        raise RuntimeError("GENA-LM embedding extraction produced no outputs.")

    max_len = max(lengths)
    num_samples = len(embeddings_list)
    seq_embeddings = np.zeros((num_samples, max_len, hidden_dim), dtype=np.float16)
    seq_mask = np.zeros((num_samples, max_len), dtype=bool)
    for i, emb in enumerate(embeddings_list):
        cur_len = emb.shape[0]
        seq_embeddings[i, :cur_len, :] = emb
        seq_mask[i, :cur_len] = True

    print(
        f"[GENA-LM] seq_embeddings shape={seq_embeddings.shape}, "
        f"dtype={seq_embeddings.dtype}, max_len={max_len}, hidden_dim={hidden_dim}"
    )
    return seq_embeddings, seq_mask


def mean_pool_embeddings(seq_embeddings: np.ndarray, seq_mask: np.ndarray) -> np.ndarray:
    seq_embeddings = np.asarray(seq_embeddings, dtype=np.float32)
    seq_mask = np.asarray(seq_mask, dtype=bool)
    mask = seq_mask[..., None].astype(np.float32)
    summed = np.sum(seq_embeddings * mask, axis=1)
    denom = np.sum(mask, axis=1)
    denom = np.maximum(denom, 1.0)
    return (summed / denom).astype(np.float32)


def pad_padded_sequence_features(
    seq_embeddings: np.ndarray,
    seq_mask: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if seq_embeddings.shape[1] >= target_len:
        return seq_embeddings, seq_mask

    pad_len = target_len - seq_embeddings.shape[1]
    emb_pad = np.pad(
        seq_embeddings,
        ((0, 0), (0, pad_len), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    mask_pad = np.pad(
        seq_mask,
        ((0, 0), (0, pad_len)),
        mode="constant",
        constant_values=False,
    )
    return emb_pad, mask_pad


def load_split(
    file_path: str,
    kmer: int,
    seq_kernel: int,
    struc_kernel: int,
    seq_model_name: str = GENA_MODEL_NAME,
    seq_batch_size: int = 64,
    seq_max_length: int = 512,
) -> dict[str, Any]:
    del seq_kernel

    seq_file = os.path.join(file_path, "sequences.fa.gz")
    stru_file = os.path.join(file_path, "structure.gz")

    seq_list, labels = read_seq_file(seq_file)
    stru_list = read_stru_file(stru_file)

    if len(seq_list) != len(stru_list):
        raise ValueError(
            f"Sequence/structure count mismatch: seq={len(seq_list)} vs stru={len(stru_list)}"
        )

    seq_embeddings, seq_mask = extract_gena_token_embeddings(
        seq_list,
        model_name=seq_model_name,
        batch_size=seq_batch_size,
        max_length=seq_max_length,
    )
    seq_pooled = mean_pool_embeddings(seq_embeddings, seq_mask)
    struc_features = build_structure_features(stru_list, kmer, struc_kernel)

    labels = np.asarray(labels).astype(int)
    print(f"[load_split] seq_pooled shape={seq_pooled.shape}")
    print(f"[load_split] structure shape={struc_features.shape}")

    return {
        "seq_embeddings": seq_embeddings,
        "seq_mask": seq_mask,
        "seq_pooled": seq_pooled,
        "struc_features": struc_features,
        "labels": labels,
    }
