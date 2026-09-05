from __future__ import annotations

import shutil

from huggingface_hub import hf_hub_download

from data import DATA_DIR, TEST_CSV, TRAIN_CSV

REPO = "sapientinc/sudoku-extreme"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} ...")
    for filename, dest in [("train.csv", TRAIN_CSV), ("test.csv", TEST_CSV)]:
        print(f"  {filename} -> {dest}")
        cached = hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset")
        shutil.copy(cached, dest)
    print("Done.")


if __name__ == "__main__":
    main()
