"""Train the robot detector on hand labels (data/datasets/hand, repeated) plus pseudo labels.

usage: python scripts/train_mixed.py NAME [--base models/autoscout-robots.pt] [--repeat 6] [--epochs 40]
Saves candidate weights and an evaluation report under data/models/experiments/NAME/.
Deploys only if held-out precision and recall do not regress and F1 improves.
"""
import argparse

from autoscout import config
from autoscout.train.train_robots import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default=str(config.ROOT / "models" / "autoscout-robots.pt"))
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    out = train(base=a.base, epochs=a.epochs, imgsz=a.imgsz, name=a.name, repeat=a.repeat)
    print("weights:", out)


if __name__ == "__main__":  # dataloader workers re-import this file on Windows
    main()
