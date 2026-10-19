"""Residual activations for Qwen3-VL-4B on generated VQA-RAD responses.

One forward pass per (record, condition) over prompt + the model's own response.
Two poolings per layer:
  last_prompt  - residual at the last prompt token (assistant header), i.e. the
                 state before any answer token exists. Clean for probing/steering.
  text_mean    - mean over non-special text tokens of the full conversation,
                 response included. This is the paper's `text_nonspecial_mean`.
Conditions: with image / without image (same system prompt + question text).

Output: <out>/acts.npy  array  [n_records, 2 cond, 2 pool, n_layers, d] fp16
        <out>/meta.json  record keys in row order, layers, cond/pool names.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import io
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .analysis import ROOT, gen

MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def messages(rec, images):
    user = [{"type": "image", "image": im} for im in images] + [
        {"type": "text", "text": rec["prompt_text"]}]
    return [{"role": "system", "content": rec["system_prompt"].strip()},
            {"role": "user", "content": user}]


@torch.no_grad()
def features(model, proc, rec, images, response, layers, skip_ids):
    msgs = messages(rec, images)
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = prompt + response + "<|im_end|>\n"
    imgs = images or None
    n_prompt = proc(text=[prompt], images=imgs, return_tensors="pt")["input_ids"].shape[1]
    batch = proc(text=[full], images=imgs, return_tensors="pt").to(model.device)
    ids = batch["input_ids"][0]
    assert ids.shape[0] > n_prompt, "response added no tokens"
    hs = model(**batch, output_hidden_states=True).hidden_states  # emb + one per layer
    text = ~torch.isin(ids, skip_ids)
    out = np.empty((2, len(layers), hs[1].shape[-1]), np.float16)
    for j, l in enumerate(layers):
        h = hs[l][0]
        out[0, j] = h[n_prompt - 1].float().cpu().numpy()
        out[1, j] = h[text].float().mean(0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--layer_step", type=int, default=2)
    ap.add_argument("--dataset", default="vqa_rad")
    ap.add_argument("--max_pixels", type=int, default=0, help="match vLLM --mm-processor-kwargs; 0 = processor default")
    a = ap.parse_args()

    recs = json.load(open(a.responses))
    recs = [r for r in recs if r["with_image"]["response"] and r["without_image"]["response"]]
    if a.limit > 0:
        recs = recs[: a.limit]
    items = {} if a.dataset == "slake" else {i["unique_id"]: i for i in gen._load_dataset_items(ROOT, a.dataset)}

    proc = AutoProcessor.from_pretrained(MODEL, **({"max_pixels": a.max_pixels} if a.max_pixels else {}))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    n_layers = model.config.text_config.num_hidden_layers
    layers = list(range(a.layer_step, n_layers + 1, a.layer_step))
    if layers[-1] != n_layers:
        layers.append(n_layers)
    tok, cfg = proc.tokenizer, model.config
    skip_ids = torch.tensor(sorted(set(tok.all_special_ids) | {
        cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id}),
        device=model.device)

    acts = np.empty((len(recs), 2, 2, len(layers), cfg.text_config.hidden_size), np.float16)
    for i, r in enumerate(tqdm(recs, desc="extract")):
        raw = [open(r["image_path"], "rb").read()] if "image_path" in r else items[r["unique_id"]]["images"]
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raw]
        acts[i, 0] = features(model, proc, r, imgs, r["with_image"]["response"], layers, skip_ids)
        acts[i, 1] = features(model, proc, r, [], r["without_image"]["response"], layers, skip_ids)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "acts.npy", acts)
    json.dump({"keys": [[r["unique_id"], r["variant_id"]] for r in recs], "layers": layers,
               "cond": ["with_image", "without_image"], "pool": ["last_prompt", "text_mean"],
               "model": MODEL}, open(out / "meta.json", "w"))
    print("saved", acts.shape, out)


if __name__ == "__main__":
    main()
