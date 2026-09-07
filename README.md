# ml

Learning AI/ML from the ground up — not just calling APIs, but understanding what happens inside.

I build things (a photo-services company, web tools, AI agents). I can ship with AI assistance.
What I could not do was judge whether a model's output was actually right.
This repo is me fixing that.

## Approach

Top-down. Run a complete, working system first — then go back and dig into the parts.
Theory-first order (math → theory → implementation) is where most people quit.

## Progress

| Phase | Focus | Status |
|---|---|---|
| **0** | Run two finished systems end to end | **In progress** |
| 1 | Rebuild them, understand each step | — |
| 2 | Classical ML on real data | — |
| 3 | Deep learning | — |
| 4 | LLMs, evals, RAG, tools | — |
| 5 | Ship something people pay for | — |

## phase0

One full loop with scikit-learn: load data → inspect → train/test split → fit → predict → evaluate → feature importance.

The point was not accuracy. It was seeing the shape of the loop once, by hand.

- `01_first_loop.ipynb`

## Setup

```bash
cd phase0
uv sync
uv run jupyter lab
```

## Stack

Python 3.12 · uv · scikit-learn · pandas · numpy · matplotlib · Jupyter
