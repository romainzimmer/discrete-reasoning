from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch

from model import MixerNextStateModel
from train import (
    best_val_cell_acc_for_resume,
    curriculum_p_gt_for_resume,
    load_last_checkpoint,
    optimizer_param_groups,
    require_run_args,
    train_run,
    validate_resume_epochs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training from runs/<id>/last.pt")
    parser.add_argument("run_dir", type=Path, help="Existing run directory")
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Total epochs for the run (replaces the saved value; must exceed completed epoch)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    device = torch.device(cli.device)
    ckpt = load_last_checkpoint(run_dir, device)
    run_args = require_run_args(ckpt, source=str(run_dir / "last.pt"))
    completed_epoch = int(ckpt["epoch"])
    validate_resume_epochs(completed_epoch, cli.epochs)

    args = Namespace(**run_args)
    args.epochs = cli.epochs
    args.device = cli.device

    model = MixerNextStateModel(dim=args.dim, num_blocks=args.num_blocks).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, weight_decay=args.weight_decay),
        lr=args.lr,
    )
    optimizer.load_state_dict(ckpt["optimizer"])

    curriculum_p_gt = (
        curriculum_p_gt_for_resume(
            ckpt,
            run_dir,
            max_outer_iters=args.train_max_outer_iters,
        )
        if not getattr(args, "no_curriculum_training", False)
        else None
    )
    print(f"Resuming {run_dir.name} from epoch {completed_epoch + 1}/{cli.epochs}")
    train_run(
        run_dir,
        args,
        start_epoch=completed_epoch + 1,
        model=model,
        optimizer=optimizer,
        best_val_cell_acc=best_val_cell_acc_for_resume(ckpt, run_dir),
        initial_curriculum_p_gt=curriculum_p_gt,
    )


if __name__ == "__main__":
    main()
