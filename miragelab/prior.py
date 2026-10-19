"""Text-only prior strength: how answerable is each question WITHOUT looking?

For every (question, paraphrase) in a run's responses.json, ask the model with an
honest prompt (no image, answer from knowledge, reply with only the answer) and
read the first-token distribution over the valid answers (Yes/No or option
letters). p_correct = renormalised probability of the ground-truth answer.
No reasoning, no sampling: a coin-flip greedy answer can't masquerade as a prior.

Output: <run>/prior.json  [{uid, variant, p_correct, coverage, k}]
"""
import argparse, json, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SYSTEM = ("No image is available for this question. Answer using only general medical "
          "knowledge and the text of the question. Reply with only {fmt}.")


def choices(rec):
    if rec["task_type"] == "mcq":
        return sorted({l.split(".")[0].strip().upper() for l in rec["options"].splitlines() if l[:1].isalpha()})
    return ["YES", "NO"]


def answer_probs(top_logprobs, valid):
    """{answer: prob} over `valid`, from one token's top_logprobs, renormalised.
    Token variants (' yes', 'Yes', 'A', ' A.') collapse onto the answer; returns
    (probs, coverage) where coverage = raw mass that landed on valid answers."""
    mass = {v: 0.0 for v in valid}
    for t in top_logprobs:
        key = t["token"].strip().strip(".:)[]*").upper()
        if key in mass:
            mass[key] += math.exp(t["logprob"])
    cov = sum(mass.values())
    return ({k: v / cov for k, v in mass.items()} if cov > 0 else mass), cov


def score(rec, url, model):
    valid = choices(rec)
    fmt = "Yes or No" if valid == ["YES", "NO"] else "the letter of one option"
    body = {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": SYSTEM.format(fmt=fmt)},
                         {"role": "user", "content": rec["prompt_text"]}]}
    out = requests.post(f"{url}/chat/completions", json=body, timeout=120).json()
    top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    probs, cov = answer_probs(top, valid)
    gt = str(rec["ground_truth_normalized"] or "").upper()
    return {"uid": rec["unique_id"], "variant": rec["variant_id"], "k": len(valid),
            "p_correct": probs.get(gt, float("nan")), "coverage": cov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-vl-4b")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    recs = json.load(open(Path(a.run) / "responses.json"))
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda r: score(r, a.url, a.model), recs))
    json.dump(rows, open(Path(a.run) / "prior.json", "w"))
    print(len(rows), "scored; mean coverage", sum(r["coverage"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
