# Span-Type Semantic Alignment and Multi-Scale Calibrated Decoding for Chinese Named Entity Recognition

This repository contains the research code accompanying the manuscript **Span-Type Semantic Alignment and Multi-Scale Calibrated Decoding for Chinese Named Entity Recognition**. It includes experiments on CMeEE-V2, CLUENER, and ChiNesE, together with baseline models, component ablations, encoder comparisons, random-seed runs, and computational profiling scripts.

The full model combines a Chinese RoBERTa encoder, BiGRU, and GlobalPointer with span-type semantic alignment, gated multi-scale features, negative-span calibration, PGD adversarial training, and validation-selected decoding parameters.

Each experiment directory is a separate runnable project with its own `config.py`. The commands below assume that the terminal is in the repository root, next to `requirements.txt`.

## 1. Repository structure

### Main experiments and baselines

| Directory | Dataset | Configuration |
| --- | --- | --- |
| `FullModel` | CMeEE-V2 | Full model; training seed 42 |
| `FullModel-CLUENER` | CLUENER | Full model |
| `FullModel - ChiNesE` | ChiNesE | Full model; includes labeled test evaluation |
| `FullModel - baseline` | CMeEE-V2 | RoBERTa-GlobalPointer baseline |
| `FullModel-CLUENER-baseline` | CLUENER | RoBERTa-GlobalPointer baseline |
| `FullModel - ChiNesE-baseline` | ChiNesE | RoBERTa-GlobalPointer baseline |
| `FullModel - ChiNesE-baseline - BERT` | ChiNesE | BERT-GlobalPointer baseline |
| `FullModel - ChiNesE-baseline - BERT-CRF` | ChiNesE | BERT-CRF sequence-labeling baseline |

### Ablations, sensitivity analysis, and profiling

| Directory | Configuration relative to the CMeEE-V2 full model |
| --- | --- |
| `FullModel - woBiGRU` | Disables BiGRU |
| `FullModel - woPGD` | Disables adversarial training |
| `FullModel - woSpan-Type` | Disables span-type auxiliary learning and semantic score fusion; fixes gamma to zero |
| `FullModel - woMulti-Scale` | Disables the multi-scale auxiliary branch, gated main-path fusion, negative calibration, and per-class threshold search; fixes beta to zero |
| `FullModel - bert` | Replaces the encoder with BERT-base-Chinese |
| `FullModel - roberta-base` | Replaces the encoder with Chinese RoBERTa-wwm-ext base |
| `FullModel - GATED_MULTISCALE_MAX_ALPHA = 0.0` | Sets the gated residual scale upper bound to 0.0 |
| `FullModel - GATED_MULTISCALE_MAX_ALPHA = 0.20` | Sets the gated residual scale upper bound to 0.20 |
| `FullModel - seed=43` | Full-model training with seed 43 |
| `FullModel - seed=44` | Full-model training with seed 44 |
| `FullModel - compute` | Full-model computational profiling |
| `FullModel - baseline - compute` | Baseline computational profiling with seed 42 |
| `FullModel - baseline - compute - seed=43` | Baseline computational profiling with seed 43 |
| `FullModel - baseline - compute - seed=44` | Baseline computational profiling with seed 44 |

The `woMulti-Scale` experiment removes the complete multi-scale/calibration/threshold component group. It is not an isolated removal of the auxiliary multi-scale scorer. Likewise, setting the gated residual upper bound to zero leaves the other enabled branches in place.

### Main source files

| File | Purpose |
| --- | --- |
| `config.py` | Model, dataset, optimization, decoding, and output settings |
| `train_CME.py` | Training and validation; also used for CLUENER and ChiNesE despite its filename |
| `predict_CME.py` | Prediction using the selected checkpoint |
| `data_loader.py` | Dataset parsing, label mapping, token alignment, and batch construction |
| `GlobalPointer.py` | GlobalPointer model and, in full-model variants, the semantic and multi-scale modules |
| `adversarial.py` | Adversarial perturbation utilities for applicable variants |
| `train_logger.py` | Training logs, checkpoint saving, and metric serialization |
| `plot_metrics.py` | Precision, recall, F1, and loss curves from training logs |
| `eval_test.py` | Labeled test evaluation in the ChiNesE experiment directories |
| `bert_crf.py` | BERT-CRF model with an internal CRF implementation |

## 2. Installation

Create and activate a Python 3.12 environment. Install the CUDA 12.8 build of PyTorch first, followed by the repository dependencies:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

The principal versions recorded in `requirements.txt` are:

| Package | Version |
| --- | --- |
| PyTorch | `2.8.0+cu128` |
| Transformers | `4.56.2` |
| Tokenizers | `0.22.2` |
| Hugging Face Hub | `0.36.2` |
| NumPy | `2.3.2` |
| Safetensors | `0.7.0` |
| tqdm | `4.66.2` |
| Matplotlib | `3.10.5` |

The BERT-CRF implementation does not require `torchcrf`. Each configuration selects `cuda:0` when CUDA is available and otherwise selects CPU. The supplied PyTorch pin is a CUDA-specific build; a CPU-only environment requires an appropriate replacement for that pin.

Dependency versions are taken from the supplied environment. They are not a record of a fresh installation test.

## 3. Pretrained encoders

Pretrained encoder weights and trained NER checkpoints are not included in the current package. Training requires the encoder files, and prediction additionally requires a trained NER checkpoint.

By default, `PRETRAINED_MODEL_DIR` points to a directory inside each experiment:

```text
<experiment>/pretrained_models/chinese_roberta_wwm_large_ext/
```

| Encoder | Hugging Face model identifier | Default local directory name |
| --- | --- | --- |
| Chinese RoBERTa-wwm-ext large | `hfl/chinese-roberta-wwm-ext-large` | `chinese_roberta_wwm_large_ext` |
| Chinese RoBERTa-wwm-ext base | `hfl/chinese-roberta-wwm-ext` | `chinese-roberta-wwm-ext` |
| BERT-base-Chinese | `google-bert/bert-base-chinese` | `bert-base-chinese` |

For example, download the large encoder into the default CMeEE-V2 full-model location:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hfl/chinese-roberta-wwm-ext-large', local_dir='FullModel/pretrained_models/chinese_roberta_wwm_large_ext', allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model*.bin'])"
```

The local directory must contain the model configuration, tokenizer files, and PyTorch-compatible weights. The code loads them through `AutoTokenizer.from_pretrained(..., use_fast=True)` and `AutoModel.from_pretrained(...)`.

To share one encoder between experiments, edit `PRETRAINED_MODEL_DIR` in the relevant `config.py` files to point to the same local directory. Retain the encoder variant used by that experiment. The original pretrained-model revision is not pinned in the supplied configuration.

## 4. Datasets

Dataset files are included under the three full-model directories:

| Dataset | Location | Training file | Validation file | Test file | Train / validation / test records |
| --- | --- | --- | --- | --- | --- |
| CMeEE-V2 | `FullModel/datasets` | `CMeEE-V2_train.json` | `CMeEE-V2_dev.json` | `CMeEE-V2_test.json` | 15,000 / 5,000 / 3,000 |
| CLUENER | `FullModel-CLUENER/datasets` | `train.json` | `dev.json` | `test.json` | 10,748 / 1,343 / 1,345 |
| ChiNesE | `FullModel - ChiNesE/datasets` | `new_train.json` | `new_eval.json` | `new_test.json` | 17,500 / 1,000 / 1,500 |

CMeEE-V2 files are JSON arrays. The included CLUENER and ChiNesE files are JSON Lines files, even though their extension is `.json`.

Baseline, ablation, and profiling directories do not contain separate copies of these datasets. Set their `DATA_DIR` to the corresponding full-model dataset directory, or place the required files in their own `datasets` directory. For a CMeEE-V2 variant, a shared path can be configured as:

```python
DATA_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..', 'FullModel', 'datasets'))
```

For CLUENER or ChiNesE variants, replace `FullModel` in this expression with `FullModel-CLUENER` or `FullModel - ChiNesE`, respectively. Keep the original split assignments and filenames.

### Annotation conventions

The CMeEE-V2 loader reads a `text` field and an `entities` list containing `start_idx`, `end_idx`, and `type`. Its current configuration uses `END_INDEX_IS_EXCLUSIVE = True`, so it interprets input spans as `[start_idx, end_idx)`.

CLUENER and ChiNesE use a nested label dictionary:

```text
label -> entity type -> mention text -> [[start, end], ...]
```

Their supplied annotations use inclusive character endpoints, `[start, end]`. ChiNesE supports overlapping and nested spans. Label mappings are defined in each experiment's `data_loader.py`.

The supplied CMeEE-V2 and CLUENER test splits do not contain usable gold entity annotations. Their prediction scripts generate entity files; they do not establish local test-set F1. The supplied ChiNesE test split is labeled and can be evaluated with `eval_test.py`.

Some dataset directories also contain prediction files from earlier runs. Use the configured training, validation, and test input files for experiments; prediction files are output artifacts.

## 5. Full-model configuration

Edit the selected experiment's `config.py` before running it. The scripts read configuration constants directly; they do not expose command-line options such as `--epochs` or `--dataset`.

| Setting | Default full-model value |
| --- | --- |
| Encoder | Chinese RoBERTa-wwm-ext large |
| Maximum sequence length | 256 tokenizer positions, including special tokens |
| Batch size | 16 |
| DataLoader workers | 4 |
| Training epochs | 10; early stopping disabled |
| Optimizer | AdamW |
| Encoder learning rate | `1e-5` |
| Ordinary task-layer learning-rate multiplier | 20 |
| Span-type learning-rate multiplier | 10 |
| Multi-scale and gated-fusion learning-rate multiplier | 5 |
| Weight decay | 0.01 |
| Schedule | 10% linear warmup, followed by linear decay |
| Gradient norm clipping | 1.0 |
| BiGRU | 1 layer; hidden size 256 per direction; output dropout 0.20 |
| GlobalPointer head size | 64, with rotary position encoding |
| PGD | 3 steps; radius 0.30; step size 0.10 |
| Span-type loss weight | 0.05 |
| Span-type negative-loss weight | 0.20 |
| Span-type temperature / negative margin | 0.07 / 0.20 |
| Span-type projection / MLP hidden size | 256 / 256 |
| Span-width embedding size / maximum width index | 32 / 64 |
| Span-type positive / negative sampling limits | 128 / 256 per batch |
| Encoder layers for multi-scale features | `[-1, -2, -4, -6]` |
| Multi-scale auxiliary loss weight | 0.01 |
| Negative calibration weight / top-k | 0.005 / 256 |
| Gated residual scale | Initial value 0.0; maximum 0.10 |
| Threshold search | Per class, from -1.50 to 2.00 in steps of 0.05 |

The auxiliary multi-scale branch detaches encoder states by default. The gated main-path fusion does not detach them. Chinese entity-type descriptions in `config.py` initialize the trainable type prototypes; these strings are model inputs.

Validation selects the decoding coefficients and per-class thresholds using:

```text
final_score = globalpointer_score + gamma * span_type_score + beta * multiscale_score

gamma candidates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
beta candidates  = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]

prediction rule = final_score > class_threshold
```

The best checkpoint is selected by validation micro-F1. For the full model, its corresponding thresholds, gamma, and beta are saved to `best_thresholds.json` and loaded for prediction.

## 6. Training and prediction

Complete the encoder and dataset setup before running these commands. If trained artifacts are available, prediction and evaluation can be run without repeating training.

### CMeEE-V2

```bash
python "FullModel/train_CME.py"
python "FullModel/predict_CME.py"
python "FullModel/plot_metrics.py"
```

The default checkpoint directory is `FullModel/outputs`. Predictions are written to `FullModel/datasets/CMeEE-V2_test_pred.json`.

### CLUENER

```bash
python "FullModel-CLUENER/train_CME.py"
python "FullModel-CLUENER/predict_CME.py"
python "FullModel-CLUENER/plot_metrics.py"
```

The default checkpoint directory is `FullModel-CLUENER/outputs_CLUENER`. Prediction creates `datasets/test_pred_entities.json` and `datasets/test_pred_cluener.jsonl` inside this experiment directory.

### ChiNesE

```bash
python "FullModel - ChiNesE/train_CME.py"
python "FullModel - ChiNesE/eval_test.py"
python "FullModel - ChiNesE/predict_CME.py"
python "FullModel - ChiNesE/plot_metrics.py"
```

The default checkpoint directory is `FullModel - ChiNesE/outputs_chinese`. Test evaluation produces `test_metrics.json` and `test_class_metrics.csv` there. Prediction produces `FullModel - ChiNesE/datasets/new_test_pred.json`.

`eval_test.py` uses the checkpoint and decoding settings selected on the validation split. It does not search for new parameters on the test split. Evaluation compares exact entity types and token-span boundaries and reports overall and per-class precision, recall, and F1.

### Baselines and ablations

Run the scripts in the corresponding experiment directory after configuring its data and encoder paths. For example:

```bash
python "FullModel - woPGD/train_CME.py"
python "FullModel - woPGD/predict_CME.py"
```

For the ChiNesE BERT-CRF baseline:

```bash
python "FullModel - ChiNesE-baseline - BERT-CRF/train_CME.py"
python "FullModel - ChiNesE-baseline - BERT-CRF/eval_test.py"
```

The BERT-CRF baseline converts nested annotations to flat BIO training labels using `FLAT_NER_STRATEGY = 'innermost'`. Its evaluation retains the full gold entity set. This protocol should be considered when interpreting comparisons with span-based models.

## 7. Saved artifacts and inference setup

Full-model training writes the following files under the configured `OUTPUT_DIR`:

| Artifact | Contents |
| --- | --- |
| `best_model.pth` | Model state dictionary selected by validation micro-F1 |
| `best_thresholds.json` | Matching per-class thresholds and selected fusion coefficients |
| `last_thresholds.json` | Decoding parameters from the latest completed epoch |
| `train_params.json` | Recorded training configuration |
| `train_log.csv` | Epoch-level loss and validation metrics |
| `train_step_log.csv` | Step-level training log |
| `class_metrics.csv` | Per-class validation metrics across epochs |
| `best_class_metrics.csv` | Per-class metrics for the best checkpoint |
| `best_models/` | Retained improving checkpoints and their `best_models.json` metadata |

`plot_metrics.py` generates `f1_curve.png`, `precision_curve.png`, `recall_curve.png`, and `loss_curve.png` from the logs. Baseline variants use their own output settings and do not all create threshold files.

For full-model inference, provide the pretrained encoder/tokenizer, `best_model.pth`, and its matching `best_thresholds.json`, using the same model configuration and label mapping as training. If the threshold file is absent, the full-model prediction script falls back to the configured defaults: threshold 0.0, gamma 0.0, and beta 0.0. That fallback does not preserve validation-selected decoding.

Prediction filenames are derived from `TEST_FILE` for the three full-model projects. Reusing the same output directory or shared prediction destination can overwrite earlier artifacts, so use separate destinations when retaining several runs.

The CMeEE-V2 full-model predictor follows `END_INDEX_IS_EXCLUSIVE = True` when exporting `end_idx`. The supplied CMeEE-V2 baseline predictors export inclusive `end_idx` values. Account for this difference when using exported predictions with an external evaluator.

## 8. Computational profiling

The profiling variants record parameter counts, training time, inference throughput and latency, and CUDA memory statistics when available. Configure their data and encoder paths in the same way as the other variants.

```bash
python "FullModel - compute/train_CME.py"
python "FullModel - compute/predict_CME.py"
python "FullModel - baseline - compute/train_CME.py"
python "FullModel - baseline - compute/predict_CME.py"
```

Profiling outputs include `train_compute_cost.csv`, `train_compute_cost.json`, and `inference_compute_cost.json` under each experiment's `OUTPUT_DIR`.

The full-model inference timer measures single-sample end-to-end prediction, including tokenization, device transfers, model forward computation, score fusion, and entity decoding. Model loading and prediction-file writing are excluded. Its first three debug predictions serve as warmup. The baseline profiling script uses `INFERENCE_WARMUP_SAMPLES = 10`. Retain the recorded measurement definitions and environment details when reporting these results.

## 9. Reproduction notes

- Keep the original dataset splits, annotation endpoint conventions, label mappings, pretrained encoder files, and experiment configuration together.
- Select checkpoints and decoding settings on validation data; evaluate labeled test data with those settings fixed.
- The main CMeEE-V2 training script uses seed 42. Dedicated full-model and baseline directories provide seed-43 and seed-44 runs. Most GlobalPointer training scripts set the seed inside `main()` rather than through a `SEED` constant in `config.py`.
- Retain `train_params.json`, metrics, checkpoints, and the corresponding threshold file for each run. A checkpoint from a different architecture or ablation is not interchangeable with the selected experiment's model definition.
- Input text is truncated to `MAX_LEN`; increasing that value changes the experimental setup.
- If DataLoader multiprocessing causes problems on Windows, set `NUM_WORKERS = 0` in the experiment's `config.py`.

This README documents the supplied source code and data layout. Training, prediction, profiling, and dependency installation were not executed as part of preparing this document.
