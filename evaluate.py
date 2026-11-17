"""Task 6 - compare SFT, PPO and DPO on the 300 held-out evaluation posts.

For every policy: one greedy summary (<=48 new tokens) per post, then
  mean r_phi            frozen reward model score
  win rate vs SFT       share of posts where r_phi(method) > r_phi(SFT)
  KL to SFT             mean over posts of sum_t [log pi(y_t|x,y<t) - log pi_sft(y_t|x,y<t)] on the method's own summary
  length                mean summary length in tokens (including the final EOS when produced)
  ROUGE-L               F-measure vs the human reference (rouge-score package)
Writes results/eval_summaries.jsonl, results/table3_comparison.{csv,md},
       results/human_eval.csv (blind sheet, 30 posts), results/human_eval_key.csv.
`python evaluate.py --human` tallies a filled-in human_eval.csv into Table 4.
"""
import os
import csv
import json
import random
import argparse
from functools import partial

import numpy as np
import torch
from rouge_score import rouge_scorer

from common import (CKPT_DIR, RESULTS_DIR, SEED, set_seed, load_tokenizer, load_policy, eval_split,
                    collate_sft, sequence_logprob, summarize_posts, freeze, to_device, chunks,
                    write_json, Timer, gpu_name)
from train_reward import RewardModel

METHODS = ["sft", "ppo", "dpo"]
N_HUMAN = 30


def as_items(posts, summary_ids):
    """prompt + generated summary in the starter's SFT format (for right-padded scoring)."""
    out = []
    for p, s in zip(posts, summary_ids):
        out.append({"prompt_ids": p["prompt_ids"], "summary_ids": s if len(s) else [0]})
    return out


@torch.no_grad()
def seq_logps(model, items, pad, bs=16):
    res = []
    for a, b in chunks(len(items), bs):
        batch = to_device(collate_sft(items[a:b], pad))
        res += sequence_logprob(model, batch["input_ids"], batch["attention_mask"], batch["prompt_lens"]).tolist()
    return res


@torch.no_grad()
def rm_scores(rm, items, pad, bs=16):
    res = []
    for a, b in chunks(len(items), bs):
        batch = to_device(collate_sft(items[a:b], pad))
        res += rm(batch["input_ids"], batch["attention_mask"]).tolist()
    return res


def bootstrap_ci(x, n=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_eval():
    set_seed(SEED)
    timer = Timer()
    tok = load_tokenizer()
    pad = tok.pad_token_id
    posts = eval_split(tok)
    print(f"evaluation posts: {len(posts)} | device={gpu_name()}")

    summaries, ids, own_logp = {}, {}, {}
    for name in METHODS:
        model = freeze(load_policy(os.path.join(CKPT_DIR, name)))
        summaries[name], ids[name] = summarize_posts(model, tok, posts)
        own_logp[name] = seq_logps(model, as_items(posts, ids[name]), pad)
        del model
        torch.cuda.empty_cache()
        print(f"generated {name} summaries | {timer}")

    sft = freeze(load_policy(os.path.join(CKPT_DIR, "sft")))
    sft_logp = {n: seq_logps(sft, as_items(posts, ids[n]), pad) for n in METHODS}
    del sft
    torch.cuda.empty_cache()

    rm = freeze(RewardModel.load(os.path.join(CKPT_DIR, "reward")))
    scores = {n: rm_scores(rm, as_items(posts, ids[n]), pad) for n in METHODS}
    scores["reference"] = rm_scores(rm, posts, pad)          # human summaries, for context
    del rm
    torch.cuda.empty_cache()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = {n: [scorer.score(p["summary"].strip(), s.strip())["rougeL"].fmeasure
                 for p, s in zip(posts, summaries[n])] for n in METHODS}
    kl = {n: [a - b for a, b in zip(own_logp[n], sft_logp[n])] for n in METHODS}
    length = {n: [len(x) for x in ids[n]] for n in METHODS}

    # ---- per-post file -----------------------------------------------------------------
    with open(os.path.join(RESULTS_DIR, "eval_summaries.jsonl"), "w", encoding="utf-8") as f:
        for i, p in enumerate(posts):
            row = {"id": i, "prompt": p["prompt"], "reference": p["summary"].strip(),
                   "reference_score": scores["reference"][i]}
            for n in METHODS:
                row[n] = summaries[n][i].strip()
                row[f"{n}_score"] = scores[n][i]
                row[f"{n}_kl"] = kl[n][i]
                row[f"{n}_len"] = length[n][i]
                row[f"{n}_rougeL"] = rouge[n][i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Table 3 -------------------------------------------------------------------------
    table = []
    for n in METHODS:
        s = np.array(scores[n])
        win = None if n == "sft" else float((s > np.array(scores["sft"])).mean())
        win_ci = None if n == "sft" else bootstrap_ci((s > np.array(scores["sft"])).astype(float))
        table.append({"policy": n.upper(), "mean_score": float(s.mean()), "score_ci95": bootstrap_ci(s),
                      "win_rate_vs_sft": win, "win_rate_ci95": win_ci,
                      "kl_to_sft": float(np.mean(kl[n])), "length": float(np.mean(length[n])),
                      "rougeL": float(np.mean(rouge[n])), "rougeL_ci95": bootstrap_ci(rouge[n]),
                      "eos_rate": float(np.mean([1.0 if (x and x[-1] == tok.eos_token_id) else 0.0
                                                 for x in ids[n]]))})
    ref_mean = float(np.mean(scores["reference"]))
    write_json(os.path.join(RESULTS_DIR, "table3_comparison.json"),
               {"rows": table, "reference_mean_score": ref_mean, "eval_time_s": timer.elapsed(),
                "gpu": gpu_name()})
    with open(os.path.join(RESULTS_DIR, "table3_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Policy", "Mean r_phi", "Win rate vs SFT", "KL to SFT", "Length (tokens)", "ROUGE-L"])
        for r in table:
            w.writerow([r["policy"], f"{r['mean_score']:.3f}",
                        "-" if r["win_rate_vs_sft"] is None else f"{r['win_rate_vs_sft']:.3f}",
                        f"{r['kl_to_sft']:.3f}", f"{r['length']:.1f}", f"{r['rougeL']:.4f}"])
    md = ["| Policy | Mean r_phi | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |",
          "|---|---|---|---|---|---|"]
    for r in table:
        win = "–" if r["win_rate_vs_sft"] is None else "%.1f%%" % (100 * r["win_rate_vs_sft"])
        md.append(f"| {r['policy']} | {r['mean_score']:.3f} | {win} | "
                  f"{r['kl_to_sft']:.3f} | {r['length']:.1f} | {r['rougeL']:.4f} |")
    md.append(f"\nHuman reference summaries: mean r_phi = {ref_mean:.3f}")
    with open(os.path.join(RESULTS_DIR, "table3_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    make_human_eval_sheet(posts, summaries)
    print(f"done in {timer}")
    return table


def make_human_eval_sheet(posts, summaries, n=N_HUMAN, seed=SEED):
    """30 posts (fixed seed), the three summaries in random order, model names hidden.
    The key (letter -> model) is written to a separate file that evaluators must not open."""
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(posts)), n))
    sheet = os.path.join(RESULTS_DIR, "human_eval.csv")
    key = os.path.join(RESULTS_DIR, "human_eval_key.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as fs, open(key, "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["item", "post_id", "post", "summary_A", "summary_B", "summary_C",
                     "member1_best (A/B/C)", "member2_best (A/B/C)"])
        wk.writerow(["item", "post_id", "A", "B", "C"])
        for k, i in enumerate(chosen, start=1):
            order = METHODS[:]
            rng.shuffle(order)
            ws.writerow([k, i, posts[i]["prompt"]] + [summaries[m][i].strip() for m in order] + ["", ""])
            wk.writerow([k, i] + order)
    # a readable version of the same blind sheet
    with open(os.path.join(RESULTS_DIR, "human_eval_sheet.md"), "w", encoding="utf-8") as f:
        f.write("# Blind human evaluation sheet (30 posts)\n\nFor each post, pick the best summary "
                "(A, B or C). Work independently; do not open `human_eval_key.csv` until both members "
                "have filled in their column of `human_eval.csv`.\n\n")
        with open(sheet, encoding="utf-8") as fs:
            for r in csv.DictReader(fs):
                f.write(f"## Item {r['item']}\n\n```\n{r['post']}\n```\n\n"
                        f"- **A:** {r['summary_A']}\n- **B:** {r['summary_B']}\n- **C:** {r['summary_C']}\n\n"
                        f"Member 1: ___   Member 2: ___\n\n")
    print(f"wrote blind sheet {sheet} and key {key}")


def tally_human_eval():
    """Table 4 from a filled-in sheet. Refuses to report anything while choices are missing."""
    sheet = os.path.join(RESULTS_DIR, "human_eval.csv")
    key = {r["item"]: r for r in csv.DictReader(open(os.path.join(RESULTS_DIR, "human_eval_key.csv"),
                                                      encoding="utf-8"))}
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    c1, c2 = "member1_best (A/B/C)", "member2_best (A/B/C)"
    valid = ("A", "B", "C")
    missing = [r["item"] for r in rows
               if r[c1].strip().upper() not in valid or r[c2].strip().upper() not in valid]
    if missing:
        print(f"Human evaluation incomplete: items {missing} have no valid A/B/C choice. Nothing tallied.")
        return None
    counts = {m: {k: 0 for k in METHODS} for m in ("member1", "member2")}
    agree = 0
    for r in rows:
        k = key[r["item"]]
        a, b = k[r[c1].strip().upper()], k[r[c2].strip().upper()]
        counts["member1"][a] += 1
        counts["member2"][b] += 1
        agree += a == b
    md = ["| | SFT picked | PPO picked | DPO picked |", "|---|---|---|---|"]
    for m in ("member1", "member2"):
        md.append(f"| {m.replace('member', 'Member ')} | {counts[m]['sft']} | {counts[m]['ppo']} | {counts[m]['dpo']} |")
    md.append(f"\nAgreement between members: {agree} / {len(rows)}")
    with open(os.path.join(RESULTS_DIR, "table4_human_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return counts, agree


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="tally the filled-in human_eval.csv")
    args = ap.parse_args()
    tally_human_eval() if args.human else run_eval()
