# Method

Looped MLP-Mixer sudoku solver on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme). The model iterates a shared mixer stack over an outer commit loop, carries detached cell memory between outer steps, and learns when to halt with a TRM-inspired halt head.

## Notation

| Symbol | Meaning |
|--------|---------|
| `x`, `y` | Clue grid and solution (`0` = empty, `1–9` = digit) |
| `P` | Encoded input (digit + clue/non-clue type embeddings) |
| `h_t`, `z_t` | Inner cell memory and pre-readout mixer state at inner step `t` |
| `M` | Shared mixer stack of `L` `MixerBlock` layers |
| `D`, `L` | Hidden dim (`--dim`) and blocks per inner step (`--num-blocks`) |
| `T_in`, `T_out` | Inner steps per outer round (`--inner-iters`) and max outer commits |
| `B` | Parallel training slots (`--train-batch-size`) |
| `λ_h` | Halt loss weight (`--halt-loss-weight`) |
| `N_try` | Max random restarts per puzzle at eval (`--max-tries`) |

## Dataset

Each row has an 81-char puzzle string (`question`), solved grid (`answer`), difficulty `rating`, and `source`. Digits are encoded as integers `0…9`.

- `data/train.csv` / `data/test.csv` from `uv run download-dataset`
- Validation is held out from `train.csv`; `test.csv` is held-out eval only
- Filter with `--min-rating`, `--max-rating`, `--max-samples`

**Training augmentations** (on by default, `--no-augment` to disable): digit relabeling, 90° rotations, band/stack permutations within the 3×3 block structure. Val, test, and viz use unaugmented puzzles starting from clues only.

## Model

`MixerNextStateModel` (`looped-mixer`) maps a partial grid to per-cell digit logits and a scalar halt logit. One small MLP-Mixer stack is reused many times instead of stacking depth statically.

### Input encoding

Each cell gets digit embedding (vocab size 10) + clue-type embedding (clue vs non-clue). The 9×9 grid is flattened to length 81. Clue cells are pinned in the decoded prediction.

### Looped MLP-Mixer stack

One inner step:

```
z_t = M(P + h_t)
```

`M` is `L` pre-norm `MixerBlock` layers: RMSNorm → token-mix `Linear(81, 81)` across cells → channel-mix SwiGLU with hidden width `round(4·D·2/3)` aligned to 256.

Triple readouts from final `z_t`:

- **Memory** — `LN_m(z_t)` reshaped to `(9, 9, D)`, detached, carried to next outer step
- **Logits** — `Unembed(LN_o(z_t))` over digits `0…9`
- **Halt** — linear head on mean-pooled `LN_a(z_t)`

Same weights for all `T_in` inner steps; only `h_t` changes within an outer round.

### Outer commit semantics

- **Outer loop** — grid commits
- **Inner loop** — mixer refinements per commit

At outer step `k ≥ 1`: apply pending candidate from step `k-1` (full-grid argmax, clues pinned) → run `T_in` inner steps → store new candidate (committed at start of next outer step) → update memory from final inner state.

Decode: clue cells keep clue value; other cells use argmax over logits.

## Training

`B` parallel puzzle slots. Each optimizer step = one outer round on every slot → backward → refill finished slots.

### Losses

- **Cell loss** — masked cross-entropy on non-clue cells with non-zero target
- **Halt target** — 1 iff pre-commit prediction equals full solution
- **Total** — `L = L_cell + λ_h · L_halt`

**Deep supervision** (default): average cell and halt loss over all inner steps; halt accuracy and done logic use the final step. `--no-deep-supervision` uses final step only.

### Slot completion

Training slot finishes when **(halt predicted AND grid correct)** OR `train-max-outer-iters` reached. Refill with a new random training puzzle. One epoch = `--batches-per-epoch` optimizer steps (one outer round each).

### Curriculum initialization

Default at seed/refill: each non-clue cell reveals ground truth with probability `p_gt ~ U[0, 1 - acc]`, where `acc` is EMA (α=½) of done-only train puzzle accuracy. Revealed cells are not pinned as clues; loss still requires correct predictions on them. Other non-clue cells get random digits `0…9`. Val/test start from clues + random non-clue fill. `--no-curriculum-training` disables.

### Optimization

AdamW (weight decay on weights, not biases). AMP enabled by default (`--no-amp` to disable). Checkpoints store weights, optimizer, hyperparameters, curriculum EMA.

## Test-time compute

Same inner/outer structure at eval, different stopping rules, optional restarts.

### `--inner-iters`

Mixer applications per outer commit. More inner steps = more recurrent depth without new parameters.

### `--max-outer-iters` / `--eval-max-outer-iters`

Cap on outer commits. **Eval** stops on halt **or** cap (halt alone suffices, grid need not be correct). Training uses separate `--train-max-outer-iters`.

Metrics: cell acc, puzzle acc, halt acc, avg outer steps, halt rate.

### `--max-tries`

Non-clue cells initialized with puzzle-seeded random digits. With `N_try > 1`, rerun full rollout from fresh init until first halt, or keep last attempt after `N_try` tries. Tries do not share memory.

### `--sweep`

One-dimensional ablations via `eval`: inner `1…N` step 1 (outer=30, tries=10); outer `10…N` step 10 (inner=3, tries=10); tries `10…N` step 10 (inner=3, outer=30). Results in `runs/<id>/history.json`.

## Code map

| File | Role |
|------|------|
| `src/model.py` | Looped MLP-Mixer |
| `src/rollout.py` | Train/eval rollouts |
| `src/train.py` | Training loop, val, checkpoints |
| `src/eval.py` | Test eval and compute sweeps |
