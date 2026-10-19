"""Pre-answer states under a NEUTRAL system prompt (the answer-pressure sentences removed).

The pressure prompt used for generation says the model MUST give a definitive answer,
never 'unknown', and should guess if uncertain. Removing just those sentences and
re-reading the without-image pre-answer state gives, per question, a paired contrast
whose only difference is the instruction to answer: pressure vector = mean(pressure -
neutral). Compare it with the mirage->abstain direction (notebook 05).

Output: <run>/acts_neutral.npy [n_records, n_layers, d] fp16, row order = acts/meta.json keys.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .extract import MODEL, messages

PRESSURE = [
    'You MUST always provide a definitive answer. Never say "unknown", "cannot determine", "unclear", or similar.\n',
    "If uncertain, make your best educated guess based on what you see.\n",
]


def neutral(system):
    out = system
    for p in PRESSURE:
        assert p in out, f"pressure sentence not found: {p!r}"
        out = out.replace(p, "")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.load(open(run / "acts" / "meta.json"))
    recs = {(r["unique_id"], r["variant_id"]): r for r in json.load(open(run / "responses.json"))}
    recs = [recs[tuple(k)] for k in meta["keys"]]
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"          # last position = last prompt token for every row
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = meta["layers"]
    # consistency check: batched left-padded path with the ORIGINAL prompt must reproduce extract.py states
    ref = np.load(run / "acts" / "acts.npy", mmap_mode="r")
    chk = recs[: a.bs]
    b = proc(text=[proc.apply_chat_template(messages(r, []), add_generation_prompt=True, tokenize=False) for r in chk],
             return_tensors="pt", padding=True).to(model.device)
    h = model(**b, output_hidden_states=True).hidden_states[layers[len(layers) // 2]][:, -1].float().cpu().numpy()
    r0 = np.asarray(ref[: len(chk), 1, 0, len(layers) // 2], np.float32)
    cos = (h * r0).sum(1) / np.linalg.norm(h, axis=1) / np.linalg.norm(r0, axis=1)
    print(f"check vs extract.py states (L{layers[len(layers) // 2]}): min cos {cos.min():.4f}", flush=True)
    assert cos.min() > 0.99, "batched path does not reproduce extracted states"
    out = np.empty((len(recs), len(layers), model.config.text_config.hidden_size), np.float16)
    for i in tqdm(range(0, len(recs), a.bs), desc="neutral"):
        chunk = recs[i:i + a.bs]
        texts = [proc.apply_chat_template(messages(dict(r, system_prompt=neutral(r["system_prompt"])), []),
                                          add_generation_prompt=True, tokenize=False) for r in chunk]
        batch = proc(text=texts, return_tensors="pt", padding=True).to(model.device)
        hs = model(**batch, output_hidden_states=True).hidden_states
        for j, l in enumerate(layers):
            out[i:i + len(chunk), j] = hs[l][:, -1].float().cpu().numpy()
    np.save(run / "acts_neutral.npy", out)
    print("saved", out.shape)


if __name__ == "__main__":
    main()
