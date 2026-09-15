# Method

Model architecture, training loop, and test-time compute. For commands and hyperparameter defaults see [cli.md](cli.md).

Looped MLP-Mixer sudoku solver on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme). The model iterates a shared mixer stack over an outer commit loop, carries detached cell memory between outer steps, and learns when to halt with a TRM-inspired halt head.

Inner update: `z_t = M(P + h_t)` with shared weights across inner steps (equivalently `h_{t+1}` from the final inner state, detached between outer commits).

## Notation

| Symbol | Meaning |
|--------|---------|
| `x`, `y` | Clue grid and solution (`0` = empty, `1–9` = digit) |
| `P` | Encoded input (digit + clue/non-clue type embeddings) |
| `h_t`, `z_t` | Inner cell memory and pre-readout mixer state at inner step `t` |
| `M` | Shared mixer stack of `L` `MixerBlock` layers |
| `D`, `L` | Hidden dimension and mixer blocks per inner step |
| `T_in` | Inner mixer steps per outer round |
| `T_out_train`, `T_out_eval` | Max outer commits during training refill vs val/test/viz |
| `B` | Parallel training slots |
| `λ_h` | Halt loss weight |
| `N_try` | Max random restarts per puzzle at eval |
| `p_{gt,g}` | Per-puzzle GT reveal probability at seed/refill |
| `p_{gt,g}^{cap}` | Adaptive upper cap on `p_{gt,g}` for rating group `g` |
| random init | Optional: fill unrevealed non-clue cells with random digits 0–9 (default: empty) |
| `g` | Rating group `0…4` (quintile bins on train `rating`; metrics only) |

## Dataset

Each row has an 81-char puzzle string (`question`), solved grid (`answer`), difficulty `rating`, and `source`. Digits are encoded as integers `0…9`.

- `data/train.csv` and `data/test.csv` (see [getting-started.md](getting-started.md))
- Validation is held out from `train.csv`; `test.csv` is held-out eval only
- Optional filters: rating range, random subsample size, fixed seed

**Training augmentations** (enabled by default): digit relabeling, 90° rotations, band/stack permutations within the 3×3 block structure. Val, test, and viz use unaugmented puzzles starting from clues only.

## Model

`MixerNextStateModel` maps a partial grid to per-cell digit logits and a scalar halt logit. One small MLP-Mixer stack is reused many times instead of stacking depth statically.

### Input encoding

Each cell gets digit embedding (vocab size 10) + clue-type embedding (clue vs non-clue). The 9×9 grid is flattened to length 81. Clue cells are pinned in the decoded prediction.

### Looped MLP-Mixer stack

One inner step:

```
z_t = M(P + h_t)
```

`M` is `L` pre-norm `MixerBlock` layers: RMSNorm → token-mix `Linear(81, 81)` across cells → channel-mix SwiGLU with hidden width `round(4·D·2/3)` aligned to 256.

Triple readouts from final `z_t`:

- **Memory**: `LN_m(z_t)` reshaped to `(9, 9, D)`, detached, carried to next outer step
- **Logits**: `Unembed(LN_o(z_t))` over digits `0…9`
- **Halt**: linear head on mean-pooled `LN_a(z_t)`

Same weights for all `T_in` inner steps; only `h_t` changes within an outer round.

### Outer commit semantics

- **Outer loop**: grid commits
- **Inner loop**: mixer refinements per commit

At outer step `k ≥ 1`: apply pending candidate from step `k-1` (full-grid argmax, clues pinned) → run `T_in` inner steps → store new candidate (committed at start of next outer step) → update memory from final inner state.

Decode: clue cells keep clue value; other cells use argmax over logits.

## Training

`B` parallel puzzle slots. Each optimizer step = one outer round on every slot → backward → refill finished slots.

### Losses

- **Cell loss**: masked cross-entropy on non-clue cells with non-zero target
- **Halt target**: 1 iff pre-commit prediction equals full solution
- **Total**: `L = L_cell + λ_h · L_halt`

**Deep supervision** (default): average cell and halt loss over all inner steps; halt accuracy and done logic use the final step. Without deep supervision, only the final inner step contributes to the loss.

### Slot completion

Training slot finishes when **(halt predicted AND grid correct)** OR `T_out_train` reached. Refill with a new random training puzzle. One epoch = a fixed number of optimizer steps (one outer round each).

### GT reveal initialization

Puzzles are grouped by difficulty into **5 rating quintiles** for per-group train/val puzzle accuracy logging and adaptive GT reveal:

| Group | Rating range |
|-------|----------------|
| G0 | 0–1 |
| G1 | 2–11 |
| G2 | 12–23 |
| G3 | 24–38 |
| G4 | 39+ |

Each group `g` has its own adaptive cap **`p_{gt,g}^{cap}`**. At slot **seed/refill**, sample **`p_{gt,g} ~ U[0, p_{gt,g}^{cap}]`** once per puzzle. For each non-clue cell independently:

- with probability `p_{gt,g}`: initialize to ground truth
- otherwise: leave empty (`0`), or a random digit `0…9` when `--random-init` is set

Revealed cells are **not** pinned as clues; loss still requires correct predictions on them. Val/test use clues only (empty non-clue cells by default, or random fill with `--random-init`) and **no GT reveal**.

Three training init modes:

| Flag | Behavior |
|------|----------|
| (default) | Adaptive per-group cap `p_{gt,g}^{cap}`; sample `p_{gt,g} ~ U[0, p_{gt,g}^{cap}]` per puzzle |
| `--no-adaptive-gt-reveal` | Fixed cap 1.0: sample `p_gt ~ U[0, 1]` per puzzle at each seed/refill |
| `--no-gt-reveal` | No GT reveal; clues + empty or random non-clue cells only |

**Adaptive controller** (default). Each group maintains a logit `ℓ_g` with cap `p_{gt,g}^{cap} = σ(ℓ_g)`. Default init: `p_{gt,g}^{cap} = 0.5`.

At the **end of each epoch**, update from that epoch’s **done-only train puzzle accuracy** per group (`acc_g`, or skip if no completions in group `g`):

```
ℓ_g ← γ · ℓ_g + η · (0.5 − acc_g)
```

- **Target** `0.5`: if the model solves too few done puzzles in a group, `p_{gt,g}^{cap}` rises (more GT hints); if too many, it falls.
- **Decay** `γ`: `--curriculum-logit-decay` (default `0.99`); prevents logits drifting unbounded.
- **Step** `η`: `--curriculum-logit-step` (default `1.0`).

Checkpoints store per-group cap logits (`curriculum_p_gt_logits`) and mean cap (`curriculum_p_gt_cap`); `resume` restores them or reconstructs state from `history.json`.

Seed/refill RNG (GT reveal, random digit fill) uses a fixed run seed for reproducibility.

### Optimization

AdamW (weight decay on weights, not biases). Mixed precision on CUDA when available. Checkpoints store weights, optimizer, and hyperparameters.

## Test-time compute

Same inner/outer structure at eval, different stopping rules, optional restarts.

### Inner steps (`T_in`)

Mixer applications per outer commit. More inner steps = more recurrent depth without new parameters.

### Outer commits (`T_out_eval`)

Cap on outer commits at eval. Stops on halt **or** cap (halt alone suffices, grid need not be correct). Training uses a separate, typically lower `T_out_train`.

Metrics: cell acc, puzzle acc, halt acc, avg outer steps, halt rate.

### Random restarts (`N_try`)

Non-clue cells start empty by default, or with puzzle-seeded random digits when `--random-init` is set. With `N_try > 1`, rerun full rollout from fresh init until first halt, or keep last attempt after `N_try` tries. Tries do not share memory.

### Compute sweeps

One-dimensional ablations at eval: sweep inner steps, max outer commits, or restart count while holding the other settings fixed. Results are written to the run's `history.json`.

## Code map

| File | Role |
|------|------|
| `src/model.py` | Looped MLP-Mixer |
| `src/rating_groups.py` | Rating quintile bins for metrics |
| `src/curriculum.py` | Adaptive per-group `p_gt` controller |
| `src/rollout.py` | Train/eval rollouts |
| `src/train.py` | Training loop, val, checkpoints |
| `src/eval.py` | Test eval and compute sweeps |
