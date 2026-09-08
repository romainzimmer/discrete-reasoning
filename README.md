# discrete-reasoning

Experiments on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

## Model

**Mixer-looped** sudoku solver: embed grid digits → looped MLP-Mixer (`h_{t+1} = M(h_t + P)`) → unembed to logits.

- **`--dim`**: embedding / mixer hidden dimension (D); channel-mix uses SwiGLU with `H = round(4·D·2/3)` aligned to 256 (TRM default)
- **`--num-blocks`**: mixer layers per inner step (not a flat FFN stack)
- **`--train-inner-iters`**: looped inner steps per outer commit during training
- **`--train-outer-iters`**: argmax commits per puzzle during training

Runs save `args.model: mixer-looped`. Old checkpoints from before this migration cannot be loaded by `eval`.

## Setup

```bash
uv sync
```

## Download dataset

```bash
uv run download-dataset
```

Writes `data/train.csv` and `data/test.csv` (~798 MB).

## Train and visualize runs

```bash
uv run train \
  --min-rating 0 --max-rating 0 \
  --max-samples 100 --epochs 30 \
  --dim 512 --num-blocks 2 \
  --train-inner-iters 5 --train-outer-iters 10 \
  --eval-inner-iters 5 --eval-outer-iters 10 \
  --train-batch-size 8
# Add --no-augment to disable on-the-fly training augmentations (ablation)
uv run python -m http.server 8000
```

Open http://localhost:8000/viz/

Training uses **train** / **validation** / **test** splits: validation is held out from `train.csv`, test comes from `test.csv`. Charts show train vs validation; test metrics are reported separately. Trajectory viz shows one grid per **outer** rollout commit.

## Python usage

```python
from data import load_split, puzzle_to_tensor, answer_to_tensor

row = load_split("test")[0]
x = puzzle_to_tensor(row["question"])  # (9, 9), 0 = empty
y = answer_to_tensor(row["answer"])    # (9, 9), 1-9
```

## References

- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/html/2510.04871v1) — MLP-Mixer blocks
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/pdf/2311.12424) — looped update `Y_{t+1} = M(Y_t + P)`
