import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    presets = json.loads((root / "presets.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(presets))
    parser.add_argument("--assets", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = presets[args.task]
    entry = root / preset["entry"]
    output = args.output.resolve()
    work = output / "work"
    data = args.assets.resolve() / "data"
    metrics = output / "results"
    command = [sys.executable, "-u", str(entry), *preset["args"], "--gpu", args.gpu]
    if args.task == "tiny-imagenet":
        command += ["--root", str(work), "--output", str(metrics)]
    else:
        command += ["--data_root", str(data), "--output_dir", str(metrics),
                    "--run_name", args.task]
    if args.dry_run:
        print(json.dumps(command))
        return 0

    if args.task == "tiny-imagenet":
        weights = args.assets.resolve() / "vit_base_patch16_224_in21k.pth"
        if not weights.is_file() or not (data / "tiny-imagenet-200" / "train").is_dir():
            parser.error("Place Tiny-ImageNet and the ViT-B/16 weights under --assets as described in README.md")

    output.mkdir(parents=True, exist_ok=False)
    work.mkdir()
    data.mkdir(parents=True, exist_ok=True)
    (work / "data").symlink_to(data, target_is_directory=True)
    for partition in (root / "partitions").glob("num_workers_*"):
        (work / partition.name).symlink_to(partition)
    if args.task == "tiny-imagenet":
        (work / weights.name).symlink_to(weights)

    env = dict(os.environ)
    env.pop("FEDRSAM_RAY_ADDRESS", None)
    env.pop("RAY_ADDRESS", None)
    env.update(
        PYTHONPATH=os.pathsep.join([str(entry.parent), str(root / "src")]),
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        FEDRSAM_TINY_CLASS_ORDER=str(root / "partitions" / "tiny_class_order.json"),
    )
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    return subprocess.call(command, cwd=str(work), env=env)


if __name__ == "__main__":
    sys.exit(main())
