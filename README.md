# mirage-probes-steering

- A recreation of Ben-Levi et al., *Mirage Probes: How Vision Models Fake Visual Understanding* ([arXiv 2606.13870](https://arxiv.org/abs/2606.13870)), on Qwen3-VL-4B-Instruct, followed by a steering-vector extension: steering the no-image pre-answer state toward abstaining on open-ended medical VQA (SLAKE).
- The original method, prompts, generator, labels and mutations are from the authors' code, [danielbenlevi/mirage_probes](https://github.com/danielbenlevi/mirage_probes). That repo is included unmodified as a git submodule at `upstream/`, pinned to commit `06c8d2d`. Upstream has no LICENSE file, so its code is referenced here, not copied.
- This repo holds only the new work: the `miragelab` package, its tests, the review notebooks, the overnight queue scripts, and one patch to upstream.
- Write-up: `docs/index.html` (GitHub Pages, served from `docs/`).

## Layout

| path | contents |
|---|---|
| `miragelab/` | extraction, labels/probes (`analysis`), SLAKE generation (`slake`), steering (`steer`, `dose`, `try_dirs`, `compare`), pressure-prompt states (`pressure`), text-only prior (`prior`), cross-model transfer (`transfer`), self-report (`introspect`), LLM-judge sample (`judge`) |
| `tests/` | `test_miragelab.py` (CPU, no model loads); `dryrun.py` (CPU dry run on a real `runs/slake`, model stubbed) |
| `review/` | executed notebooks 01–05 (outputs kept) |
| `scripts/queue/` | `chain-*.sh` overnight queues as they were run (they `cd` into the original bench checkout and call `production.miragelab.*`) |
| `patches/vlm_model_registry-qwen3_vl_4b.patch` | adds `qwen3_vl_4b_instruct` to upstream's `vlm_model_registry.py` (needed only to regenerate VQA-RAD / MedXpertQA-MM responses with the upstream generator) |
| `upstream/` | submodule: danielbenlevi/mirage_probes @ `06c8d2d` |
| `docs/` | Pages site: `index.html`, `img/` (figures exported from the notebooks) |

Not in git: `runs/` (outputs, ~2.5 GB), `data/`, `raw_data/` (datasets), Jupyter token/pid files.

## Paths

- `MIRAGE_ROOT`: data root holding `runs/`, `raw_data/`, `data/`. Default: the current working directory. All `--run runs/...` defaults are relative to it.
- `MIRAGE_UPSTREAM`: upstream checkout used for `scripts.data.gen_mutations_get_responses`. Default: `upstream/` next to `miragelab/`.
- Notebooks: run from `review/`; they read `MIRAGE_ROOT` (default `..`).

## Setup

```bash
git clone --recurse-submodules <this repo>
cd mirage-probes-steering
python -m venv .venv && . .venv/bin/activate
pip install torch transformers accelerate scikit-learn numpy pandas pillow requests tqdm matplotlib phate pytest jupyter
```

- Versions used: torch 2.11.0, transformers 5.16.1, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, phate 2.0.0, accelerate 1.14.0 (Jetson Orin, Python 3.10).
- Response generation (`miragelab.slake`, upstream generator) calls an OpenAI-compatible vLLM server serving Qwen3-VL-4B-Instruct as `qwen3-vl-4b` at `http://127.0.0.1:8000/v1`.
- Datasets expected under `MIRAGE_ROOT`: `raw_data/slake/{train,validation,test}.json` + `raw_data/slake/imgs/`; `data/vqa_rad.parquet`, `data/medxpertqa_mm.parquet` (upstream loaders, `upstream/scripts/data/`).
- Upstream's `data/final_data/*_responses.json` / `*_contrastive.json` are Git LFS files and are not needed here.

## Run

```bash
python -m pytest -q tests                      # CPU, ~10 s
export MIRAGE_ROOT=/path/to/data-root          # holds runs/, raw_data/, data/
cd "$MIRAGE_ROOT"

# Part 1: VQA-RAD recreation (upstream generator; args from runs/full/run_metadata.json, exact invocation not recorded)
git -C <repo>/upstream apply ../patches/vlm_model_registry-qwen3_vl_4b.patch
python <repo>/upstream/scripts/data/gen_mutations_get_responses.py --datasets vqa_rad --vlm qwen3_vl_4b_instruct \
  --vllm_model qwen3-vl-4b --vllm_base_url http://127.0.0.1:8000/v1 \
  --reuse_mutations_path <repo>/upstream/data/final_data/mutations.json --save_path runs/full
python -c "from miragelab.analysis import subset; print(subset('runs/full'))"
python -m miragelab.extract --responses runs/full/responses_subset.json --out runs/full/acts

# Part 2: SLAKE generation, states, steering
python -m miragelab.slake --out runs/slake
python -m miragelab.extract --dataset slake --responses runs/slake/responses.json --out runs/slake/acts
python -m miragelab.steer --run runs/slake
python -m miragelab.pressure --run runs/slake
python -m miragelab.dose --run runs/slake --stage {fit,dose,test,sweep,pool} [--seed N] [--vectors_from 0] [--model {think4b,instruct2b,cosmos2b}]
python -m miragelab.introspect --run runs/slake
python -m miragelab.transfer --stage {gen,steer} --model {instruct2b,cosmos2b}
```

- `<repo>` = path to this checkout; run with `PYTHONPATH=<repo>` (or from the repo root with `MIRAGE_ROOT` set).
- Each module's `--help` lists its arguments; `scripts/queue/chain-*.sh` show the exact sequences and timeouts that produced `runs/`.
- Notebooks: `cd review && MIRAGE_ROOT=... jupyter lab`.

## Results

- `docs/index.html` (scaffold, numbers with sources) and the executed notebooks in `review/`.

## License

- None chosen yet. Upstream has no license either.
