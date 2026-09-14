# Runs & viz

Training writes one directory per run under `runs/` (gitignored).

## Layout

```
runs/<run-id>/
  history.json          # args + per-epoch train/val metrics
  manifest.json         # which trajectories exist per split/epoch
  last.pt / best.pt     # checkpoints
  epochs/0005.pt        # optional per-epoch checkpoints
  trajectories/
    validation/
      epoch_0005/
        puzzle_0000.json
  profile/
    trace.json          # optional PyTorch profiler output
```

### history.json

- `args`: full training config (`model: looped-mixer`, hyperparameters)
- `epochs[]`: `train_*` and `val_*` metrics per epoch
- `test`: optional block written by `eval` (metrics + sweeps)

### Trajectory JSON

One file per puzzle. `predictions` / `states` hold one 81-char grid string per **outer** commit. Clue cells stay fixed; the model fills non-clue cells via argmax each outer step.

## Viz

After training, from the **repo root**:

```bash
uv run python -m http.server 8000
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/)

**Important:** serve the project root, not `viz/`. The page loads runs via `../runs`; serving from inside `viz/` breaks run discovery.

The UI shows:

- Train / validation metric charts per epoch
- Test sweep charts (after `uv run eval --sweep`)
- Trajectory player (input grid, prediction grid, outer-step slider)
