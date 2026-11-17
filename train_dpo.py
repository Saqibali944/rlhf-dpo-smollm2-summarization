"""Task 5 - Stage 4: Direct Preference Optimization (DPO).

pi_theta (trainable) and pi_ref (frozen) both start from checkpoints/sft/.
log pi(y|x) = sum of token log-probs over the summary tokens only.
L_DPO = -E[ log sigmoid( beta*(log pi(yw|x)-log ref(yw|x)) - beta*(log pi(yl|x)-log ref(yl|x)) ) ]
beta 0.1, 1 epoch over the 20,000 preference pairs, AdamW lr 2e-6, batch 8 pairs.
Every 250 steps: loss + implicit reward accuracy on the 1,000 validation pairs.
"""
import os
import math
import argparse
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (SCHEDULE, SCHEDULE_NAME, CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, set_seed, load_tokenizer, load_policy,
                    pref_splits, collate_pref, sequence_logprob, freeze, all_frozen,
                    to_device, chunks, make_adamw, micro_batches, enable_checkpointing, JsonlLogger, write_json, Timer, gpu_name)

BETA, LR, BATCH_PAIRS, EVAL_EVERY = 0.1, 2e-6, 8, 250


def pair_logps(model, b):
    lw = sequence_logprob(model, b["chosen_ids"], b["chosen_mask"], b["prompt_lens"])
    ll = sequence_logprob(model, b["rejected_ids"], b["rejected_mask"], b["prompt_lens"])
    return lw, ll


def dpo_logits(pw, pl, rw, rl, beta=BETA):
    """The argument of sigma: beta*log(pi/ref)(yw) - beta*log(pi/ref)(yl)."""
    return beta * (pw - rw) - beta * (pl - rl)


@torch.no_grad()
def all_pair_logps(model, data, pad_id, batch_pairs=8):
    model.eval()
    W, L = [], []
    for b in DataLoader(data, batch_pairs, shuffle=False, collate_fn=partial(collate_pref, pad_id=pad_id)):
        lw, ll = pair_logps(model, to_device(b))
        W.append(lw.cpu())
        L.append(ll.cpu())
    return torch.cat(W), torch.cat(L)


def validate(policy, val, pad_id, ref_w, ref_l):
    pw, pl = all_pair_logps(policy, val, pad_id)
    policy.train()
    z = dpo_logits(pw, pl, ref_w, ref_l)
    return {"acc": (z > 0).float().mean().item(), "loss": -F.logsigmoid(z).mean().item(),
            "margin": z.mean().item() / BETA}


def main(micro_pairs=2, max_steps="schedule", sft_dir=None, out_dir=None, log_name="dpo_log.jsonl"):
    set_seed(SEED)
    sft_dir = sft_dir or os.path.join(CKPT_DIR, "sft")
    out_dir = out_dir or os.path.join(CKPT_DIR, "dpo")
    tok = load_tokenizer()
    train, val = pref_splits(tok)
    pad = tok.pad_token_id

    policy = enable_checkpointing(load_policy(sft_dir))
    policy.train()
    ref = freeze(load_policy(sft_dir))
    assert all_frozen(ref), "reference model must be frozen"
    print(f"[check] reference frozen: every parameter requires_grad=False -> {all_frozen(ref)}")

    # the reference never changes -> its validation log-probs are computed once
    ref_w, ref_l = all_pair_logps(ref, val, pad)
    logger = JsonlLogger(os.path.join(LOG_DIR, log_name))
    v = validate(policy, val, pad, ref_w, ref_l)
    print(f"step     0 | val loss {v['loss']:.4f} | implicit reward acc {v['acc']:.4f}")
    logger.log(step=0, val_loss=v["loss"], val_acc=v["acc"], time_s=0.0)

    opt = make_adamw(policy.parameters(), LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train, BATCH_PAIRS, shuffle=True, generator=g,
                        collate_fn=partial(collate_pref, pad_id=pad))
    if max_steps == "schedule":
        max_steps = SCHEDULE["dpo_steps"]
    n_steps = len(loader) if max_steps is None else min(max_steps, len(loader))

    timer, wl, wa = Timer(), [], []
    first_step_loss = None
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch)
        n = batch["chosen_ids"].shape[0]
        step_loss, step_acc = 0.0, 0.0
        for mb in micro_batches(batch, micro_pairs):
            with torch.no_grad():
                rw, rl = pair_logps(ref, mb)
            pw, pl = pair_logps(policy, mb)
            z = dpo_logits(pw, pl, rw, rl)
            loss = -F.logsigmoid(z).sum() / n
            loss.backward()
            step_loss += loss.item()
            step_acc += (z > 0).float().sum().item() / n
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1:
            first_step_loss = step_loss
            print(f"[check] DPO first-step loss = {step_loss:.6f}  (log 2 = {math.log(2):.6f})")
        wl.append(step_loss)
        wa.append(step_acc)
        logger.log(step=step, step_loss=step_loss, step_acc=step_acc)

        if step % SCHEDULE["eval_every"] == 0 or step == n_steps:
            v = validate(policy, val, pad, ref_w, ref_l)
            tl, ta = sum(wl) / len(wl), sum(wa) / len(wa)
            logger.log(step=step, train_loss=tl, train_acc=ta, val_loss=v["loss"], val_acc=v["acc"],
                       val_margin=v["margin"], time_s=timer.elapsed())
            print(f"step {step:5d}/{n_steps} | train loss {tl:.4f} | val loss {v['loss']:.4f} | "
                  f"implicit reward acc {v['acc']:.4f} | {timer}")
            wl, wa = [], []
        if step == n_steps:
            break

    train_time = timer.elapsed()
    assert all_frozen(ref)
    policy.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)
    if not log_name.startswith("smoke"):
        write_json(os.path.join(RESULTS_DIR, "dpo_summary.json"),
                   {"first_step_loss": first_step_loss, "final_train_loss": tl,
                    "val_implicit_reward_acc": v["acc"], "val_loss": v["loss"],
                    "steps": n_steps, "full_spec_steps": len(loader), "schedule": SCHEDULE_NAME, "train_time_s": train_time, "gpu": gpu_name(),
                    "models_in_gpu_memory": 2, "ref_frozen": all_frozen(ref)})
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_pairs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default="schedule")
    args = ap.parse_args()
    main(args.micro_pairs, args.max_steps)
