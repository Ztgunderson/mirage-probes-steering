"""Lean steering test of the two directions compare.py liked, on seed 0's split (so it lines up with runs/slake/dose):
  mirage all   type-centred mirage - abstain on base no-image states, every train text (vs ~7 in the fold vectors)
  image        with-image minus no-image state, paired over train texts (subtracting it pushes toward "no image")
  random       isotropic, same sizes
Each at --shares % of the median layer state norm, on top of the honesty prompt, on seed 0's 114 test texts
(no image) + --n_image with-image records (accuracy cost). Rows -> <run>/dose_try/rows.json (resumes arm by arm).

  python -m miragelab.try_dirs --run runs/slake [--smoke 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import analysis as A
from .dose import split, run_arms
from .extract import MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slake")
    ap.add_argument("--layer", type=int, default=34, help="seed 0's layer, so arms compare with dose/test_rows.json")
    ap.add_argument("--shares", default="10,20")
    ap.add_argument("--n_image", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    df, acts, meta = A.load(a.run)
    idx, lab, _, _ = A.unique_texts(df)
    texts, cat = df.question.map(A.norm_text).values[idx], df.category.values[idx]
    fit, dose, test = split(texts, lab, 0)
    train = fit | dose
    j = meta["layers"].index(a.layer)
    S = np.asarray(acts[idx, 1, 0, j], np.float32)          # no image, last prompt token
    Si = np.asarray(acts[idx, 0, 0, j], np.float32)         # with image
    unit = lambda v: v / np.linalg.norm(v)
    Sc = A.center_by(S, cat, train)
    m, ab = train & (lab == "mirage"), train & (lab == "abstain")
    dirs = {"mirage all": unit(Sc[m].mean(0) - Sc[ab].mean(0)),
            "image": unit((Si[train] - S[train]).mean(0)),
            "random": unit(np.random.default_rng(0).normal(size=S.shape[1]))}
    nrm = float(np.median(np.linalg.norm(S[train], axis=1)))
    print(f"layer {a.layer} | train {train.sum()} texts ({m.sum()} mirage, {ab.sum()} abstain) test {test.sum()} "
          f"| median norm {nrm:.1f} | cos(mirage, image) {dirs['mirage all'] @ dirs['image']:.2f}", flush=True)

    arms = [("prompt", "prompt", 0, True, [])] + [
        (f"prompt + {d} {sh}%", d, sh, True, [(v, sh / 100 * nrm)]) for sh in map(int, a.shares.split(",")) for d, v in dirs.items()]
    allrecs = {(r["unique_id"], r["variant_id"]): r for r in json.load(open(Path(a.run) / "responses.json"))}
    recs = [allrecs[tuple(meta["keys"][i])] for i in idx[test]]
    rows = np.where(df.question.map(A.norm_text).isin(set(texts[test])).values)[0]
    img = [allrecs[tuple(meta["keys"][i])] for i in np.random.default_rng(0).permutation(rows)[: a.n_image]]
    if a.smoke:
        recs, img, arms = recs[: a.smoke], img[: a.smoke], arms[:2]

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    out = Path(a.run) / ("dose_try_smoke" if a.smoke else "dose_try")
    out.mkdir(exist_ok=True)
    run_arms(model, proc, arms, recs, img, out / "rows.json", a)


if __name__ == "__main__":
    main()
