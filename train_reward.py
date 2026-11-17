"""Task 3 - Stage 2: reward model r_phi(x, y).

Architecture: SFT transformer body (model.model, LM head unused) + nn.Linear(hidden, 1)
applied to the final hidden state of the LAST REAL token of prompt+summary.
Loss (Bradley-Terry, lecture Eq. 40):  L = -E[ log sigmoid(r(x, y_w) - r(x, y_l)) ]
1 epoch over 20,000 preference pairs, AdamW lr 1e-5, batch 8 pairs.
Validation: pairwise accuracy on the 1,000 validation pairs. Saved to checkpoints/reward/.
"""
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (SCHEDULE, SCHEDULE_NAME, CKPT_DIR, LOG_DIR, RESULTS_DIR, DEVICE, SEED, set_seed, load_tokenizer,
                    load_policy, pref_splits, collate_pref, autocast,
                    position_ids_from_mask, last_token_index, to_device, chunks, make_adamw,
                    micro_batches, enable_checkpointing,
                    JsonlLogger, write_json, Timer, gpu_name)

LR, BATCH_PAIRS, EVAL_EVERY = 1e-5, 8, 250
HEAD_FILE = "reward_head.pt"


class RewardModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm                       # only lm.model (the transformer body) is used
        self.head = nn.Linear(lm.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=1.0 / (lm.config.hidden_size + 1) ** 0.5)
        nn.init.zeros_(self.head.bias)
        self.head.to(next(lm.parameters()).device)

    def forward(self, input_ids, attention_mask):
        with autocast():
            h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=position_ids_from_mask(attention_mask), use_cache=False).last_hidden_state
        idx = last_token_index(attention_mask)          # last REAL token (mask), never pad id
        h_last = h[torch.arange(h.shape[0], device=h.device), idx].float()
        return self.head(h_last).squeeze(-1)            # (B,)

    def save(self, path):
        self.lm.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, HEAD_FILE))

    @classmethod
    def load(cls, path, device=DEVICE):
        rm = cls(load_policy(path, device))
        rm.head.load_state_dict(torch.load(os.path.join(path, HEAD_FILE), map_location=device))
        return rm


def score_pairs(rm, batch):
    rw = rm(batch["chosen_ids"], batch["chosen_mask"])
    rl = rm(batch["rejected_ids"], batch["rejected_mask"])
    return rw, rl


@torch.no_grad()
def validate(rm, data, pad_id, batch_pairs=16):
    rm.eval()
    rws, rls = [], []
    for b in DataLoader(data, batch_pairs, shuffle=False, collate_fn=partial(collate_pref, pad_id=pad_id)):
        rw, rl = score_pairs(rm, to_device(b))
        rws += rw.tolist()
        rls += rl.tolist()
    rm.train()
    rw, rl = torch.tensor(rws), torch.tensor(rls)
    return {"acc": (rw > rl).float().mean().item(),
            "loss": -F.logsigmoid(rw - rl).mean().item(),
            "rw": rws, "rl": rls}


def main(micro_pairs=4, max_steps="schedule", sft_dir=None, out_dir=None, log_name="reward_log.jsonl"):
    set_seed(SEED)
    sft_dir = sft_dir or os.path.join(CKPT_DIR, "sft")
    out_dir = out_dir or os.path.join(CKPT_DIR, "reward")
    tok = load_tokenizer()
    train, val = pref_splits(tok)
    print(f"Pref train/val: {len(train)}/{len(val)}  device={gpu_name()}")

    rm = RewardModel(enable_checkpointing(load_policy(sft_dir)))
    rm.train()
    # only parameters that influence the reward are optimised (lm_head is tied to
    # the input embedding in SmolLM2, so no separate LM-head weight exists)
    opt = make_adamw(rm.parameters(), LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train, BATCH_PAIRS, shuffle=True, generator=g,
                        collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))
    if max_steps == "schedule":
        max_steps = SCHEDULE["reward_steps"]
    n_steps = len(loader) if max_steps is None else min(max_steps, len(loader))
    logger = JsonlLogger(os.path.join(LOG_DIR, log_name))

    v = validate(rm, val, tok.pad_token_id)
    print(f"step     0 | val acc {v['acc']:.4f} | val loss {v['loss']:.4f}")
    logger.log(step=0, val_acc=v["acc"], val_loss=v["loss"], time_s=0.0)

    timer, wl, wa = Timer(), [], []
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch)
        n = batch["chosen_ids"].shape[0]
        step_loss, step_acc = 0.0, 0.0
        for mb in micro_batches(batch, micro_pairs):
            rw, rl = score_pairs(rm, mb)
            loss = -F.logsigmoid(rw - rl).sum() / n      # mean over the 8 pairs of the step
            loss.backward()
            step_loss += loss.item()
            step_acc += (rw > rl).float().sum().item() / n
        opt.step()
        opt.zero_grad(set_to_none=True)
        wl.append(step_loss)
        wa.append(step_acc)
        logger.log(step=step, step_loss=step_loss, step_acc=step_acc)

        if step % SCHEDULE["eval_every"] == 0 or step == n_steps:
            v = validate(rm, val, tok.pad_token_id)
            tl, ta = sum(wl) / len(wl), sum(wa) / len(wa)
            logger.log(step=step, train_loss=tl, train_acc=ta, val_acc=v["acc"],
                       val_loss=v["loss"], time_s=timer.elapsed())
            print(f"step {step:5d}/{n_steps} | train loss {tl:.4f} acc {ta:.3f} | "
                  f"val acc {v['acc']:.4f} loss {v['loss']:.4f} | {timer}")
            wl, wa = [], []
        if step == n_steps:
            break

    train_time = timer.elapsed()
    rm.save(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)

    if not log_name.startswith("smoke"):
        # per-pair validation predictions for the error analysis of Task 7.1
        rows = []
        for ex, rw, rl in zip(val, v["rw"], v["rl"]):
            rows.append({"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"],
                         "r_chosen": rw, "r_rejected": rl, "correct": rw > rl,
                         "len_chosen": len(ex["chosen_ids"]), "len_rejected": len(ex["rejected_ids"])})
        with open(os.path.join(RESULTS_DIR, "reward_val_predictions.jsonl"), "w", encoding="utf-8") as f:
            import json
            for r in rows:
                f.write(json.dumps(r) + "\n")
        write_json(os.path.join(RESULTS_DIR, "reward_summary.json"),
                   {"final_train_loss": tl, "val_pairwise_acc": v["acc"], "val_loss": v["loss"],
                    "steps": n_steps, "full_spec_steps": len(loader), "schedule": SCHEDULE_NAME, "train_time_s": train_time, "gpu": gpu_name()})
    return v["acc"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_pairs", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default="schedule")
    args = ap.parse_args()
    main(args.micro_pairs, args.max_steps)
