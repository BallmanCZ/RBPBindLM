# GENA-LM + Stochastic Gates + Transformer/Cross-Attention

This directory keeps the original KSU experiment workflow intact at the experiment level:

- KSU undersampling remains the data balancing stage.
- Cross-validation, threshold selection, independent test, ROC export, and metric export remain in place.
- The sequence feature extractor is replaced from `kmer + one-hot` to `GENA-LM`.
- The sequence hidden size is reduced from `768 -> 300` with stochastic gates.
- The classifier is replaced from `CNN + BiLSTM` to `Transformer Encoder + Cross-Attention`.

## Files

- `run_ksu_gena_lm_transformer_ca.py`: main experiment script.
- `get_dataset_gena_lm.py`: GENA-LM sequence embedding and structure feature loading.
- `ksu_undersampling.py`: standalone KSU undersampling implementation.
- `metrics.py`: standalone evaluation metrics implementation.
- `stochastic_gates.py`: supervised stochastic-gate feature selector.
- `multi_model_transformer_ca.py`: Transformer Encoder and bidirectional Cross-Attention classifier.

## Important note

For tractability, KSU is run on the mean-pooled 768-d GENA-LM sequence representation, then the selected sample indices are used to align the token-level sequence embeddings and the structure branch. This preserves the original "sequence branch decides indices, structure branch follows indices" logic without running KSU on a very large flattened token tensor.

## Run

```powershell
python .\gena_lm_transformer_ca_ksu\run_ksu_gena_lm_transformer_ca.py
```

## Output

Outputs are written under the repo-level `results/` directory, using a run tag that includes:

- `GENALM`
- `SG300`
- `TxCA`
