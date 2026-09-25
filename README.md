# CLIK-Diffusion on intraoral scans

Tooth alignment prediction with [CLIK-Diffusion](https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion)
(Dou et al., *Medical Image Analysis* 2025) on the 1060-patient intraoral-scan `Orthodontic_dental_dataset`
(Wang et al., 2024), and finetuning of its diffusion model on that dataset.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU.

```bash
git clone https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion.git src/CLIK-Diffusion
uv sync
```

`uv sync` creates `.venv` with Python 3.12 and the locked dependencies, including torch built for CUDA 13.0.

Download the released checkpoints from the authors'
[Google Drive](https://drive.google.com/drive/folders/1o9tVJ6p8Jbad3gu0ZUkX0tE5dp7Vkh9g) into
`src/CLIK-Diffusion/Code/checkpoint/`, and unpack the dataset into `data/`, so that
`data/Orthodontic_dental_dataset/00*/ori/*.stl` exists.

## Usage

```bash
uv run src/infer_oral_scan.py -n 0007        # predict one patient
uv run src/infer_oral_scan.py -b             # predict every patient (resumes an interrupted run)
uv run src/finetune_diffusion.py --wandb     # finetune the diffusion model
uv run src/eval_finetune.py -r version_xx    # released vs finetuned weights, on the run's test patients
```

Predictions are written to `outputs/Output/<patient>/results/`; extraction cases are skipped.
Finetuning takes `--max_epochs`, `--patience`, `--lr`, `--batch_size` and `--wandb`, and saves
`outputs/finetune_runs/lightning_logs/version_<n>/finetuned.pth`, which `infer_oral_scan.py --ckpt` also accepts.
Without `-r`, `eval_finetune.py` evaluates the latest run.
