

# Don’t Click That: Teaching Web Agents to Resist Deceptive Interfaces
<p align="center">
  <!-- <a href="https://xiaowu0162.github.io/long-mem-eval/"><img src="https://img.shields.io/badge/🌐-Website-red" height="23"></a> -->
  <a href="TODO"><img src="https://img.shields.io/badge/📝-Paper (ACL 2026)-blue" height="23"></a>
  <a href="https://huggingface.co/datasets/Ink0722/RUC" ><img src="https://img.shields.io/badge/🤗-Data-green" height="23"></a>
</p>

Codebase for the ACL 2026 submission on Don’t Click That: Teaching Web Agents to Resist Deceptive Interfaces. The repository focuses on evaluating and improving web-browsing click judgments under deceptive UI conditions, including local multimodal evaluators, GLM-based experience summarization, and supporting data-processing scripts.

## Repository Layout

```text
.
├── src/
│   ├── config.py                 # Centralized settings and environment loading
│   ├── core/                     # Model backends, agent logic, parsing, rewards
│   ├── evaluator/                # Prompt templates for evaluation/summarization
│   └── utils/                    # Dataset loading, formatting, click generation
├── train.py                      # Stage 1 training entrypoint
├── stage1_inference.py           # Stage 1 inference / snapshot generation
├── stage2.py                     # Stage 2 iterative experience optimization
├── run_agent.py                  # Agent runner
├── run_agent_with_evalutor.py    # Agent runner with evaluator
├── nom_results.py                # Result aggregation / nominal analysis
├── opt_exp.py                    # Experience optimization experiments
├── main.py                       # Minimal GLM evaluation example
├── requirements.txt
├── .env.example
└── README.md
```

## Environment

Tested with the dependency set in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU usage, install the PyTorch build that matches your CUDA environment before or instead of the pinned `torch` packages if needed.

## Configuration

This repository now uses centralized configuration via environment variables and `src/config.py`.

1. Copy `.env.example` to `.env`.
2. Fill in the required keys.
3. Adjust paths or default model names if your local setup differs.

Example:

```env
ZHIPUAI_API_KEY=your_key_here
CHATANYWHERE_API_KEY=
DEFAULT_LOCAL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
DEFAULT_EVAL_MODEL=glm-4.6v
DEFAULT_DEVICE=cuda
DATA_PATH=data/annotations/annotation.jsonl
IMAGES_DIR=data/images
OUTPUT_DIR=outputs
HF_ENDPOINT=https://hf-mirror.com
BASE_URL=
```

### Required Variables

- `ZHIPUAI_API_KEY`: required for GLM-based evaluation or summarization workflows.
- `DATA_PATH`: annotation file path.
- `IMAGES_DIR`: directory for referenced UI screenshots.

### Optional Variables

- `DEFAULT_LOCAL_MODEL`
- `DEFAULT_EVAL_MODEL`
- `DEFAULT_DEVICE`
- `OUTPUT_DIR`
- `HF_ENDPOINT`
- `CHATANYWHERE_API_KEY`
- `BASE_URL`

## Data

The repository does not assume public redistribution of the full dataset. By default, local data is ignored by Git.

Expected layout:

```text
data/
├── annotations/
│   └── annotation.jsonl
└── images/
```

If your data file differs, update `DATA_PATH` and `IMAGES_DIR` in `.env`.

## Main Workflows

### 1. Train the evaluator

```bash
python train.py
```

### 2. Run Stage 1 inference

```bash
python stage1_inference.py
```

### 3. Run Stage 2 experience optimization

```bash
python stage2.py
```

### 4. Run the agent with evaluator

```bash
python run_agent_with_evalutor.py
```

### 5. Aggregate results

```bash
python nom_results.py --help
```

## Notes

- Several scripts in this repository were developed for iterative experimentation. The main maintained configuration path is now `src/config.py` plus `.env`.
- Generated outputs such as `stage1_result/`, `stage2_result/`, and `outputs/` are ignored by Git.
- Before publishing the repository, rotate any API keys that were ever stored in local code history.

## Citation

If you use this repository, please cite the associated ACL 2026 paper. Add the final BibTeX entry here after camera-ready details are fixed.

## Runtime Notes
- Local multimodal backends require CUDA GPU. CPU loading is intentionally disabled.

