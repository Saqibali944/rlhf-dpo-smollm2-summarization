"""Task 2 - Stage 1: supervised fine-tuning (SFT).

Loss = mean negative log-likelihood of the summary tokens only
    L_SFT = - 1/#summary_tokens * sum_{t in summary} log pi_theta(x_t | x_<t)
1 epoch over 20,000 pairs, AdamW lr 5e-5, batch 16, grad-norm clip 1.0,
train + validation loss logged every 250 steps, final model -> checkpoints/sft/.
"""
import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from common import (SCHEDULE, SCHEDULE_NAME, MODEL_NAME, CKPT_DIR, LOG_DIR, RESULTS_DIR, DEVICE, SEED,
                    set_seed, load_tokenizer, load_policy, sft_splits, eval_split,
                    collate_sft, masked_token_logprobs, summary_mask, make_adamw,
                    micro_batches, enable_checkpointing, summarize_posts, to_device,
                    chunks, JsonlLogger, write_json, Timer, gpu_name)

LR, BATCH_SIZE, CLIP, EVAL_EVERY = 5e-5, 16, 1.0, 250


def sft_loss_sum(model, batch):
    """Sum of summary-token NLL and number of summary tokens for one (micro)batch."""
    m = summary_mask(batch["prompt_lens"], batch["attention_mask"])
    lp = masked_token_logprobs(model, batch["input_ids"], batch["attention_mask"], m)
    return -(lp * m).sum(), m.sum()


@torch.no_grad()
def validation_loss(model, data, pad_id, batch_size=8):
    model.eval()
    tot, n = 0.0, 0.0
    for b in DataLoader(data, batch_size, shuffle=False, collate_fn=partial(collate_sft, pad_id=pad_id)):
        s, c = sft_loss_sum(model, to_device(b))
        tot, n = tot + s.item(), n + c.item()
    model.train()
    return tot / n


def main(micro_batch=4, max_steps="schedule", out_dir=None, log_name="sft_log.jsonl"):
    set_seed(SEED)
    out_dir = out_dir or os.path.join(CKPT_DIR, "sft")
    tok = load_tokenizer()
    train, val = sft_splits(tok)
    eval_posts = eval_split(tok)
    print(f"SFT train/val: {len(train)}/{len(val)}  device={gpu_name()}")

    model = enable_checkpointing(load_policy(MODEL_NAME))
    model.train()

    # Task 2.5: 5 evaluation posts, greedy, <=48 new tokens, BEFORE SFT
    before, _ = summarize_posts(model, tok, eval_posts[:5])

    opt = make_adamw(model.parameters(), LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train, BATCH_SIZE, shuffle=True, generator=g,
                        collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))
    if max_steps == "schedule":
        max_steps = SCHEDULE["sft_steps"]
    n_steps = len(loader) if max_steps is None else min(max_steps, len(loader))
    logger = JsonlLogger(os.path.join(LOG_DIR, log_name))

    v0 = validation_loss(model, val, tok.pad_token_id)
    print(f"step     0 | val loss {v0:.4f}")
    logger.log(step=0, val_loss=v0, train_loss=None, time_s=0.0)

    timer, window = Timer(), []
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch)
        n_tok = summary_mask(batch["prompt_lens"], batch["attention_mask"]).sum()
        step_loss = 0.0
        for mb in micro_batches(batch, micro_batch):
            # dividing each micro-batch sum by the token count of the FULL batch
            # makes the accumulated gradient identical to a batch-16 step
            s, _ = sft_loss_sum(model, mb)
            loss = s / n_tok
            loss.backward()
            step_loss += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        window.append(step_loss)
        logger.log(step=step, step_loss=step_loss, grad_norm=gnorm)

        if step % SCHEDULE["eval_every"] == 0 or step == n_steps:
            tl = sum(window) / len(window)
            vl = validation_loss(model, val, tok.pad_token_id)
            logger.log(step=step, train_loss=tl, val_loss=vl, time_s=timer.elapsed())
            print(f"step {step:5d}/{n_steps} | train loss {tl:.4f} | val loss {vl:.4f} | {timer}")
            window = []
        if step == n_steps:
            break

    train_time = timer.elapsed()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)

    after, _ = summarize_posts(model, tok, eval_posts[:5])
    samples = [{"post": p["prompt"], "reference": p["summary"], "before_sft": b, "after_sft": a}
               for p, b, a in zip(eval_posts[:5], before, after)]
    if not log_name.startswith("smoke"):
        write_json(os.path.join(RESULTS_DIR, "sft_before_after.json"), samples)
        write_json(os.path.join(RESULTS_DIR, "sft_summary.json"),
                   {"final_train_loss": tl, "final_val_loss": vl, "steps": n_steps, "full_spec_steps": len(loader), "schedule": SCHEDULE_NAME,
                    "train_time_s": train_time, "gpu": gpu_name()})
    return samples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default="schedule", help="smoke test only")
    args = ap.parse_args()
    for s in main(args.micro_batch, args.max_steps):
        print("\nREFERENCE:", s["reference"], "\nBEFORE   :", s["before_sft"], "\nAFTER    :", s["after_sft"])
