# Learning to Summarize from Human Feedback — SFT, Reward Model, PPO and DPO from scratch

Generative AI, Assignment 03. We align **SmolLM2-135M** to write Reddit TL;DRs with the full
RLHF pipeline of Stiennon et al. (2020): **supervised fine-tuning → Bradley–Terry reward model →
PPO with a KL penalty**, and compare it with **DPO**. Every loss and every algorithm is implemented
in plain PyTorch. The project uses no TRL, PEFT, trlx, Hugging Face `Trainer` or
`AutoModelForSequenceClassification`. From `transformers`, we use only `AutoTokenizer`,
`AutoModelForCausalLM` and `model.generate`.

<!-- RESULTS-START -->
> **Results below come from our own runs** on an RTX 3050 Laptop GPU (4 GB), using the assignment's
> hyperparameters with a **reduced number of steps** (see [Training schedule](#training-schedule-full-vs-reduced)).

### Table 1 – Data

| Quantity | Value |
|---|---|
| SFT train / validation pairs | 20,000 / 500 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.4 / 448 |
| Mean / max summary length, SFT train (tokens, incl. EOS) | 33.8 / 48 |

### Table 2 – Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT (400 steps) | 2.390 | val. loss: **2.398** (2.800 before) | 17 min / RTX 3050 Laptop |
| Reward model (500 steps) | 0.650 | pairwise acc.: **62.4%** (53.1% before) | 22 min / RTX 3050 Laptop |
| PPO (60 iterations) | 0.159 (total, last iter.) | final mean r_φ: **+0.40** (last batch; −1.58 at iter. 1) | 14 min / RTX 3050 Laptop |
| DPO (500 steps) | 0.651 | implicit reward acc.: **60.1%** | 41 min / RTX 3050 Laptop |

### Table 3 – Final comparison (300 evaluation posts, greedy, ≤ 48 tokens)

| Policy | Mean r_φ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | −1.548 | – | 0 | 30.7 | 0.2568 |
| PPO | −0.556 | 77.0% | 7.871 | 47.9 | 0.1783 |
| DPO | −1.061 | 83.7% | 1.751 | 39.2 | 0.2529 |

Human reference summaries score −1.369. Per-post outputs: `results/eval_summaries.jsonl`.

### Table 4 – Human evaluation (30 posts)

**Not yet filled in.** The blind sheet is in `results/human_eval.csv` (readable version:
`results/human_eval_sheet.md`). Each member fills in their own column without opening `results/human_eval_key.csv`;
then `python evaluate.py --human` writes `results/table4_human_eval.md`.

### Key findings

* The reward model is clearly above chance but **length-biased**: it is 80.7% accurate when the preferred summary is longer
  and 37.1% when it is shorter.
* **PPO exploited that bias.** Summaries grew to the 48-token cap (EOS rate 83% → 2%), 88% contain a repeated 4-word
  phrase, ROUGE-L dropped 0.257 → 0.178, and the reward still rose. A higher r_φ is not a better summary.
* **DPO** improved on SFT more conservatively: an 83.7% win rate, ROUGE-L kept, and 4.5× less KL than PPO.

Full answers to Task 7: [`results/analysis.md`](results/analysis.md). Ten annotated examples: [`results/samples.md`](results/samples.md).

### Figures

| | |
|---|---|
| ![SFT](results/figures/fig1_sft_loss.png) | ![RM](results/figures/fig2_reward_model.png) |
| ![PPO](results/figures/fig3_ppo_curves.png) | ![DPO](results/figures/fig4_dpo.png) |

Also: `fig0_token_lengths.png`, `fig3b_ppo_losses.png` (the three PPO loss terms), `fig6_comparison.png`, `fig5_frontend.png`.

### Correctness checks (Section 3.3), output of `scripts/collect_checks.py`

| Check | Result | Evidence |
|---|---|---|
| SFT mask: masked tokens decode to exactly summary + EOS | PASS | token ids identical on 16 examples |
| Frozen models (PPO): every parameter of π_ref and r_φ has `requires_grad=False` | PASS | asserted at start and after every iteration |
| Frozen models (DPO): π_ref frozen | PASS | asserted at start and end |
| PPO iteration 1: KL to reference = 0 | PASS | KL = 0.0 |
| PPO every iteration: first epoch r_t(θ) = 1 and clip fraction 0 | PASS | max abs(r_t − 1) = 0.0 over all 60 iterations; first-epoch clip fraction always 0 |
| DPO first step: loss = log 2 | PASS | 0.693147 vs 0.693147 |
| Padding: generation prompts left-padded; RM reads the last real token via `attention_mask` | PASS | reward with no/right/left padding: max diff 8.1e-06 in fp32 (0.08 under bf16 autocast) |

Full output: [`results/correctness_checks.md`](results/correctness_checks.md). Task 1.3 decoded-mask output is in notebook 01.
<!-- RESULTS-END -->

---

## Repository layout

| Path | Contents |
|---|---|
| `starter/` | the given starter code, **unchanged** (`config.py`, `models.py`, `data.py`, `batching.py`, `check_starter.py`) |
| `common.py` | `token_logprobs`, `summary_mask`, generation helper, padding/position helpers, micro-batching, activation checkpointing, cached starter data |
| `train_sft.py` | Task 2: summary-only NLL, AdamW 5e-5, batch 16, clip 1.0 |
| `train_reward.py` | Task 3: `RewardModel` (SFT body + `nn.Linear(576,1)` at the last real token) and Bradley–Terry training |
| `train_ppo.py` | Task 4: value head, rollout, KL-penalised per-token rewards, returns/advantages, clipped PPO loss |
| `train_dpo.py` | Task 5: DPO loss, implicit-reward accuracy |
| `evaluate.py` | Task 6: 300-post comparison and the blind human-evaluation sheet (`--human` tallies Table 4) |
| `make_figures.py` | Figures 1–4 and 6 from the logs |
| `notebooks/` | six **executed** notebooks (01 data → 06 evaluation); `build_notebooks.py` regenerates them |
| `results/` | logs, tables, figures, `eval_summaries.jsonl`, `samples.md`, `human_eval.csv`, `analysis.md`, `correctness_checks.md` |
| `app/` | Streamlit front end |
| `docs/` | Medium article, LinkedIn post, contribution plan |
| `poster/` | A3 poster (PDF, PNG, editable SVG) and its generator |
| `VIVA_PREPARATION.md` | algorithms, formulas, code walkthrough, 40+ viva questions |
| `scripts/` | `smoke_test.py`, `run_all_notebooks.py`, `collect_checks.py` |

## Setup

```bash
git clone https://github.com/<github-user>/rlhf-tldr.git
cd rlhf-tldr
pip install -r requirements.txt          # torch, transformers, datasets, rouge-score, numpy, matplotlib, streamlit
python starter/check_starter.py          # Task 1.1: downloads the model and both datasets
```

**Colab / Kaggle (T4):** open any notebook. Its first code cell (commented) clones the repo,
installs the requirements and can point `CKPT_DIR` at Google Drive, so checkpoints survive the session.
On a T4 (no bf16) the code automatically runs in fp32.

## Reproducing everything

```bash
python scripts/smoke_test.py             # 3 steps of every stage (~12 min) – catches errors before the long runs
python scripts/run_all_notebooks.py      # executes notebooks 01–06 in order and saves their outputs
# or stage by stage:
python train_sft.py && python train_reward.py && python train_ppo.py && python train_dpo.py && python evaluate.py
python make_figures.py && python scripts/collect_checks.py && python poster/make_poster.py
streamlit run app/app.py                 # front end
python evaluate.py --human               # Table 4, after both members filled in results/human_eval.csv
```

### Training schedule: full vs reduced

All hyperparameters are exactly those of the assignment (learning rates, batch sizes, clipping,
ε = 0.2, β_KL = 0.05, c_V = 0.1, c_H = 0.01, γ = 1, K = 4, 16 prompts per rollout, DPO β = 0.1).
The **number of steps** is selected with the environment variable `RLHF_SCHEDULE`:

| | SFT steps | RM steps | DPO steps | PPO iterations | eval every |
|---|---|---|---|---|---|
| `full` (default = the assignment) | 1,250 (1 epoch) | 2,500 (1 epoch) | 2,500 (1 epoch) | 200 | 250 |
| `reduced` | 400 | 500 | 500 | 60 | 100 |

**The results in this repository were produced with `RLHF_SCHEDULE=reduced`** on a 4 GB
RTX 3050 *laptop* GPU. On that card a single batch of 16 takes about 6 s, so the full schedule would need
about 14 hours of GPU time. The reduced run uses identical code and hyperparameters and sees fewer examples:
6,400 SFT pairs, 4,000 preference pairs for the RM and for DPO, and 960 PPO prompts. To reproduce the
full assignment, run without the variable (about 4–6 h on a Colab/Kaggle T4).

### Fitting a 4 GB GPU without changing the algorithm

* **Gradient accumulation.** A batch of 16 is split into micro-batches. Each micro-batch's *sum* is divided by the
  full batch's token/pair count, so the accumulated gradient equals the batch-16 gradient.
* **Activation checkpointing** with `torch.utils.checkpoint` (gradients identical, verified).
* **LM head only on summary positions** (`masked_token_logprobs`, equal to `token_logprobs`×mask to 3e-5).
* **PPO:** π_ref and r_φ sit on the GPU during rollout scoring and move to the CPU during the update.
  Rollout scoring and update reuse the *same* micro-batches, so ρ = 1 exactly in the first epoch.
* fp32 master weights with bf16 autocast (transformers 5 loads SmolLM2 in bf16; lr 1e-5 updates would vanish in bf16).

## Checkpoints

Checkpoints are **not** committed (see `.gitignore`). Upload the four folders from `checkpoints/`
(`sft/`, `reward/`, `ppo/`, `dpo/`, ~540 MB each in fp32) to Hugging Face or Google Drive:

```bash
huggingface-cli login
huggingface-cli upload <hf-user>/rlhf-tldr-smollm2 checkpoints/sft sft
huggingface-cli upload <hf-user>/rlhf-tldr-smollm2 checkpoints/reward reward
huggingface-cli upload <hf-user>/rlhf-tldr-smollm2 checkpoints/ppo ppo
huggingface-cli upload <hf-user>/rlhf-tldr-smollm2 checkpoints/dpo dpo
```

Download links: **TODO, add after uploading.** Hugging Face: `https://huggingface.co/<hf-user>/rlhf-tldr-smollm2`,
or Google Drive: `<drive-folder-link>`. To use them, download into `checkpoints/` (or set `CKPT_DIR`):

```bash
huggingface-cli download <hf-user>/rlhf-tldr-smollm2 --local-dir checkpoints
```

## Front end

`streamlit run app/app.py` loads the SFT, PPO, DPO and reward-model checkpoints, takes a post
(subreddit, title, body; formatted exactly like the training prompts), and shows the three greedy
summaries side by side, each with its reward-model score.

![Front end](results/figures/fig5_frontend.png)

## Team and contributions

Member 1: *name*. Member 2: *name*. See `docs/CONTRIBUTION_PLAN.md`; each member commits their own work.
