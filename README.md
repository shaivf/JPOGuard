# JPOGuard

This repository provides data and code for multimodal news misleadingness detection and headline correction with joint supervised fine-tuning and preference optimization.

## Get Started

### Repository Structure

```text
.
├── data/
│   ├── sft_targets.jsonl
│   ├── dpo_train.jsonl
│   ├── dpo_dev.jsonl
│   └── test_inputs.jsonl
└── code/
    ├── train_sft.py
    ├── train_dpo.py
    ├── evaluate.py
    ├── evaluate_csr.py
    ├── run_ablation.py
    ├── prompts/
    ├── splits/
    ├── model/
    ├── dpo_runtime/
    └── requirements.txt
```

### Data Preparation

Download `train_data.json`, `test_data.json`, `test_correction.json`, `train_img`, and `test_img` following the [OMGuard data preparation instructions](https://github.com/fanxiao15/OMGuard#data-preparation). Place them in one directory and set its path as `DATA`.

The `data/` directory provides SFT targets, DPO preference pairs, and test reader views. Sample splits are in `code/splits/`.

### Installation

```bash
pip install -r code/requirements.txt
```

Set `BASE` to the base model directory, `INIT` to the initial adapter directory, and `WORK` to a writable output directory. If the adapter has weights but no configuration, copy `code/initial_adapter_config.json` into `INIT` as `adapter_config.json`.

### Model Training

```bash
python code/train_sft.py --data "$DATA" --targets data/sft_targets.jsonl \
  --base-model "$BASE" --initial-adapter "$INIT" --output "$WORK/sft"

python code/train_dpo.py --data "$DATA" --pairs-dir data --base-model "$BASE" \
  --sft-adapter "$WORK/sft/selected_sft_checkpoint.json" --output "$WORK/dpo"
```

### Detection and Headline Correction

```bash
python code/evaluate.py --data "$DATA" --reader-data data/test_inputs.jsonl \
  --base-model "$BASE" --adapter "$WORK/dpo/selected_dpo_checkpoint.json" --output "$WORK/test"
```

### Evaluation

Set the evaluation model and endpoint as `MODEL` and `ENDPOINT`, and set `JUDGE_API_KEY` in the environment.

```bash
python code/evaluate_csr.py --data "$DATA" --predictions "$WORK/test/correction.jsonl" \
  --judge-model "$MODEL" --judge-endpoint "$ENDPOINT" --output "$WORK/csr"
```

### Ablation Experiments

`code/run_ablation.py` supports `joint_sft`, `rewrite_error_only`, `original_only`, and `mixed_90`. Run `python code/run_ablation.py --help` for the required parameters.

## Reference

This work uses the MM-MISLEADING dataset from [OMGuard](https://github.com/fanxiao15/OMGuard).
