# CSD: Cooperative Sign Decoder

CSD reconstructs one-bit attention keys using retained values and shared information across heads and neighboring layers. This repository contains the implementation and a Qwen2.5-3B-Instruct study with three seeds.

## Run

Requires Python 3.12 and an NVIDIA CUDA GPU with BF16 support. Run from the repository root.

```powershell
python -m pip install -r requirements.txt
python -B -m csd campaign
```

The study uses `configs/study.json` and a cumulative five-hour limit. Repeat the campaign command to resume within the remaining budget.

Preview the workload, run correctness checks, or report the current campaign.

```powershell
python -B -m csd campaign --dry-run
python -B -m csd smoke
python -B -m csd report --name csd_current
```

Results are saved in `results/`. The paper is in `paper/`. View the retained historical results with `python -B -m csd report`.
