

from pathlib import Path
import argparse
import os
import tempfile

import torch
from ultralytics import YOLO
import yaml

'''
how to get files to the runpod machine:
scp -P 46597 -i ~/.ssh/id_ed25519 ~/documents/GitHub/taking-stock-yolo/yolo_runpod_train.py root@38.65.239.12:/root/yolo_runpod_train.py


apt update && apt install -y unzip
python -m pip install --upgrade pip && python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 && python -m pip install ultralytics==8.4.30 pyyaml
'''

'''
tmux stuff

apt update
apt install -y tmux

Paste this once on the pod:
cat > ~/.tmux.conf <<'EOF'
set -g mouse on
set -g history-limit 100000
setw -g mode-keys vi
set -g status-bg black
set -g status-fg white
EOF

tmux start-server
tmux new -d -s train
tmux source-file ~/.tmux.conf
tmux attach -t train

Daily use:
tmux new -s train
python yolo_runpod_train.py 2>&1 | tee train.log

Leave it running safely:
Press Ctrl-b, then d

Come back later:
tmux attach -t train
'''


# Profile-based defaults so you can switch pod configs with --profile.
CONFIG_PROFILES = {
    "runpod_3090": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "name": "takingstock_x_3090_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "",  # empty means auto-detect (CUDA -> MPS -> CPU)
        "workers": 0,
        "cache": "disk",
        "amp": False,
        "cudnn": False,
        "freeze": True,
        "augment": True,
    },
    "runpod_4x4090": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 32,
        "name": "takingstock_c36_v1_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "0,1,2,3",
        "workers": 16,
        "cache": "disk",
        "amp": True,
        "cudnn": True,
        "freeze": True,
        "augment": True,
    },
    "runpod_3x4090": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 24,
        "name": "takingstock_c36_v1_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "0,1,2",
        "workers": 12,
        "cache": "disk",
        "amp": True,
        "cudnn": True,
        "freeze": True,
        "augment": True,
    },
}

DEFAULT_PROFILE = "runpod_3x4090"


def train_once(model, *, data, epochs, imgsz, batch, name, patience, device, workers, project, augment, cache, amp):
    return model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        patience=patience,
        device=device,
        workers=workers,
        project=project,
        exist_ok=True,
        augment=augment,
        cache=cache,
        amp=amp,
    )


def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f"model.{x}." for x in range(num_freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False
    print(f"{num_freeze} layers are frozen.")


def resolve_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on RunPod or local machine.")
    parser.add_argument(
        "--profile",
        choices=sorted(CONFIG_PROFILES.keys()),
        default=DEFAULT_PROFILE,
        help="Named training profile.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model checkpoint or model yaml path.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--batch",
        default=None,
        help="Batch size. Use -1 for auto-batch.",
    )
    parser.add_argument(
        "--name",
        default=None,
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Output project directory.",
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (examples: 0, cpu, mps, 0,1).",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--cache",
        default=None,
        choices=["ram", "disk", "False", "false"],
        help="Dataset caching strategy. disk is safer than ram on pods.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP mixed precision.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP mixed precision.")

    cudnn_group = parser.add_mutually_exclusive_group()
    cudnn_group.add_argument("--cudnn", dest="cudnn", action="store_true", help="Enable cuDNN kernels.")
    cudnn_group.add_argument("--no-cudnn", dest="cudnn", action="store_false", help="Disable cuDNN kernels.")

    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze",
        dest="freeze",
        action="store_true",
        help="Freeze first 10 backbone layers.",
    )
    freeze_group.add_argument(
        "--no-freeze",
        dest="freeze",
        action="store_false",
        help="Disable freezing first 10 backbone layers.",
    )

    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument("--augment", dest="augment", action="store_true", help="Enable augmentations.")
    augment_group.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentations.")

    parser.set_defaults(amp=None, cudnn=None, freeze=None, augment=None)
    return parser.parse_args()


def resolve_config(args):
    cfg = dict(CONFIG_PROFILES[args.profile])
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def normalize_data_yaml(data_path: Path) -> str:
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    configured_root = data_cfg.get("path")
    if not configured_root:
        return str(data_path)

    configured_root_path = Path(configured_root)
    if configured_root_path.exists():
        return str(data_path)

    # If the configured dataset root is invalid on this machine, remap to folder containing data.yaml.
    data_cfg["path"] = str(data_path.parent)
    fd, temp_path = tempfile.mkstemp(prefix="runpod_data_", suffix=".yaml")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    print(
        f"Rewriting data.yaml path from '{configured_root}' to '{data_path.parent}' for this run only."
    )
    return temp_path


def main():
    args = parse_args()
    cfg = resolve_config(args)

    device = resolve_device(cfg["device"] or None)
    augment = cfg["augment"]

    batch = int(cfg["batch"]) if str(cfg["batch"]).lstrip("-").isdigit() else cfg["batch"]
    cache = False if str(cfg["cache"]).lower() == "false" else cfg["cache"]

    model_path = Path(cfg["model"])
    data_path = Path(cfg["data"])
    project_path = Path(cfg["project"])
    data_yaml_for_run = normalize_data_yaml(data_path)

    model = YOLO(str(model_path))

    if cfg["freeze"]:
        model.add_callback("on_train_start", freeze_layer)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = cfg["cudnn"]
        # Prefer deterministic, lower-fragmentation behavior in containerized training.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False

    print(f"Profile: {args.profile}")
    print(f"Using device: {device}")
    print(f"AMP enabled: {cfg['amp']}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Training data: {data_path.resolve() if data_path.exists() else data_path}")
    print(f"Saving runs to: {project_path.resolve()}")

    try:
        train_once(
            model,
            data=data_yaml_for_run,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=batch,
            name=cfg["name"],
            patience=cfg["patience"],
            device=device,
            workers=cfg["workers"],
            project=str(project_path),
            augment=augment,
            cache=cache,
            amp=cfg["amp"],
        )
    except RuntimeError as e:
        if "CUDNN_STATUS_NOT_INITIALIZED" not in str(e):
            raise

        safe_batch = max(1, batch // 2) if isinstance(batch, int) and batch > 1 else 1
        print("Encountered CUDNN_STATUS_NOT_INITIALIZED. Retrying once with safer settings...")
        print(f"Retry settings: batch={safe_batch}, workers=0, cache=False, amp=False")
        torch.cuda.empty_cache()

        train_once(
            model,
            data=data_yaml_for_run,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=safe_batch,
            name=cfg["name"],
            patience=cfg["patience"],
            device=device,
            workers=0,
            project=str(project_path),
            augment=augment,
            cache=False,
            amp=False,
        )


if __name__ == "__main__":
    main()

