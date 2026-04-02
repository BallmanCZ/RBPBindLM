# RBPBindLM

RBPBindLM is a cleaned repository for RNA-binding protein binding prediction with imbalance-aware training and multimodal sequence-structure modeling.

The repository contains:

- `gena_lm_transformer_ca_ksu/`: the main RBPBindLM implementation that keeps KSU undersampling, replaces sequence `k-mer + one-hot` with `GENA-LM`, applies Stochastic Gates for `768 -> 300` feature reduction, and upgrades the classifier to `Transformer Encoder + Cross-Attention`.
- `datasets/`: train/test datasets used by the cleaned experiment.

## RBPBindLM pipeline

1. KSU undersampling is used to rebalance the training set.
2. Sequence features are extracted with `GENA-LM`.
3. Stochastic Gates reduce sequence features from `768` to `300`.
4. Structure features keep the original `k-mer + one-hot` encoding.
5. Transformer Encoders and bidirectional Cross-Attention are used for final prediction.

## Main code directory

The main implementation is under:

- [gena_lm_transformer_ca_ksu](./gena_lm_transformer_ca_ksu)

Important files:

- `run_ksu_gena_lm_transformer_ca.py`: training, cross-validation, threshold selection, and independent test.
- `get_dataset_gena_lm.py`: GENA-LM sequence embedding and structure feature loading.
- `stochastic_gates.py`: supervised feature reduction.
- `multi_model_transformer_ca.py`: Transformer Encoder + Cross-Attention classifier.

## Dataset layout

The dataset folder follows the original project layout, for example:

```text
datasets/
  clip/
    1_PARCLIP_AGO1234_hg19/
      30000/
        training_sample_0/
        test_sample_0/
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the cleaned RBPBindLM experiment:

```bash
python gena_lm_transformer_ca_ksu/run_ksu_gena_lm_transformer_ca.py
```

## Notes

- The repository includes datasets, but excludes generated training results and saved model weights.
- Only the cleaned `RBPBindLM` implementation is published in this repository.
