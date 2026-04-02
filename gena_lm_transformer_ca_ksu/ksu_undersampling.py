import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def ksu_undersampling(X, y, target_ratio=1.0, k=6, random_state=None, save_indices_path=None):
    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"X and y size mismatch: X[0]={X.shape[0]} vs y={n}")

    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        indices = np.arange(n, dtype=int)
        indices = np.sort(indices, kind="mergesort")
        if save_indices_path:
            np.savetxt(save_indices_path, indices, fmt="%d")
        return X, y, indices

    target_neg = int(len(pos_idx) * float(target_ratio))
    target_neg = min(target_neg, len(neg_idx))
    if target_neg <= 0:
        target_neg = 1
    remove_total = max(len(neg_idx) - target_neg, 0)

    if remove_total == 0:
        indices = np.concatenate([pos_idx, neg_idx]).astype(int)
        indices = np.sort(indices, kind="mergesort")
        X_resampled = X[indices]
        y_resampled = y[indices]
        if save_indices_path:
            np.savetxt(save_indices_path, indices, fmt="%d")
        return X_resampled, y_resampled, indices

    X_flat = X.reshape((X.shape[0], -1))
    X_neg_flat = X_flat[neg_idx]
    k_eff = int(max(1, min(k, len(neg_idx))))
    kmeans = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_neg_flat)

    clusters = []
    for c in range(k_eff):
        mask = labels == c
        clusters.append(neg_idx[mask] if np.any(mask) else np.array([], dtype=int))

    sizes = np.array([len(idx) for idx in clusters], dtype=float)
    if sizes.sum() == 0:
        keep_neg = neg_idx.copy()
    else:
        alloc = sizes / sizes.sum() * remove_total
        remove_each = np.floor(alloc).astype(int)
        remainder = int(remove_total - remove_each.sum())
        if remainder > 0:
            frac = alloc - np.floor(alloc)
            order = np.argsort(-frac)
            for j in order[:remainder]:
                remove_each[j] += 1

        for i in range(len(clusters)):
            n_i = len(clusters[i])
            if n_i > 0:
                remove_each[i] = int(min(remove_each[i], max(n_i - 1, 0)))
            else:
                remove_each[i] = 0

        to_remove_global = set()
        for i, idx_global in enumerate(clusters):
            n_i = len(idx_global)
            r_i = int(remove_each[i])
            if n_i <= 1 or r_i <= 0:
                continue

            A = X_flat[idx_global]
            col_max = np.max(A, axis=0)
            safe_den = np.where(col_max == 0, 1.0, col_max)
            Pi = A / safe_den

            D_sq = pairwise_distances(Pi, metric="euclidean", squared=True)
            D1 = D_sq / float(n_i)

            triu_r, triu_c = np.triu_indices(n_i, k=1)
            pair_dist = D1[triu_r, triu_c]
            order = np.argsort(pair_dist)

            removed_local = set()
            for p in order:
                if len(removed_local) >= r_i:
                    break
                a, b = triu_r[p], triu_c[p]
                cand = rng.choice([a, b])
                if cand not in removed_local:
                    removed_local.add(cand)

            if len(removed_local) < r_i:
                remain = [j for j in range(n_i) if j not in removed_local]
                extra = rng.choice(remain, size=(r_i - len(removed_local)), replace=False)
                removed_local.update(int(x) for x in np.asarray(extra).ravel())

            for j in removed_local:
                to_remove_global.add(int(idx_global[j]))

        keep_neg = np.array([t for t in neg_idx if t not in to_remove_global], dtype=int)

        if keep_neg.shape[0] > target_neg:
            need_more = keep_neg.shape[0] - target_neg
            extra_drop = rng.choice(keep_neg, size=need_more, replace=False)
            mask = np.isin(keep_neg, extra_drop, invert=True)
            keep_neg = keep_neg[mask]

    indices = np.concatenate([pos_idx, keep_neg]).astype(int)
    indices = np.sort(indices, kind="mergesort")
    X_resampled = X[indices]
    y_resampled = y[indices]

    if save_indices_path:
        np.savetxt(save_indices_path, indices, fmt="%d")

    return X_resampled, y_resampled, indices
