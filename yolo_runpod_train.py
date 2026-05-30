

from pathlib import Path
import argparse
import os
import tempfile

import torch
from ultralytics import YOLO
import yaml

'''
how to get files to the runpod machine:
scp -P 32990 -i ~/.ssh/id_ed25519 ~/documents/GitHub/taking-stock-yolo/yolo_check_dataset.py root@38.80.152.148:/root/yolo_check_dataset.py

zip goes to workspace on network storage
scp -P 32990 -i ~/.ssh/id_ed25519 ~/documents/GitHub/taking-stock-yolo/yolo_dataset.zip root@38.80.152.148:/workspace/yolo_dataset.zip

ssh root@38.80.152.148 -p 32990 -i ~/.ssh/id_ed25519

MOVE DATASET FROM WORK OT ROOT
unzip on network storage: 
cd /workspace
unzip yolo_dataset.zip
cp -r /workspace/yolo_dataset /root/yolo_dataset

MOVE MODEL BACK TO WORKSPACE
cp -r /root/runs/detect/runs/takingstock_c36_v1_yolo26x/ /workspace/takingstock_c36_v1_yolo26x/

DOWNLOAD MODEL
scp -r -P 10162 -i ~/.ssh/id_ed25519 root@203.57.40.220:/workspace/runs/detect/runs/takingstock_c36_v1_yolo26x ~/documents/GitHub/taking-stock-yolo/runs/takingstock_c36_v1_yolo26x


-- erase cache files after a failed run
pkill -f yolo_runpod_train.py || true
pkill -f ultralytics || true
find /root/yolo_dataset -type f -name "*.npy" -delete
find /root/yolo_dataset -type f -name "*.cache" -delete

'''

'''
ENV setup

apt update && apt install -y unzip
python -m pip install --upgrade pip && python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 && python -m pip install ultralytics==8.4.30 pyyaml


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
python yolo_runpod_train.py --profile runpod_4x4090 2>&1 | tee train.log
python yolo_runpod_train.py --pilot # for 3 epoch test

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
    "runpod_6x4090": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 72,
        "name": "takingstock_c36_v1_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "0,1,2,3,4,5",
        "workers": 8,
        "cache": "disk",
        "amp": True,
        "cudnn": True,
        "freeze": True,
        "augment": True,
    },
    "runpod_4xh200_sxm": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 32,
        "name": "takingstock_c45_h200_4x_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "0,1,2,3",
        "workers": 24,
        "cache": "disk",
        "amp": True,
        "cudnn": True,
        "freeze": True,
        "augment": True,
    },
    "runpod_8xh200_sxm": {
        "model": "yolo26x.pt",
        "data": "yolo_dataset/data.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 64,
        "name": "takingstock_c45_h200_8x_yolo26x",
        "project": "runs",
        "patience": 20,
        "device": "0,1,2,3,4,5,6,7",
        "workers": 32,
        "cache": "disk",
        "amp": True,
        "cudnn": True,
        "freeze": True,
        "augment": True,
    },
}

ACTIVE_PROFILE = "runpod_4xh200_sxm"
DEFAULT_PROFILE = ACTIVE_PROFILE


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

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a short pilot using the selected profile before committing to the full overnight run.",
    )
    parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=3,
        help="Epoch count for pilot mode.",
    )

    parser.set_defaults(amp=None, cudnn=None, freeze=None, augment=None)
    return parser.parse_args()


def resolve_config(args):
    if DEFAULT_PROFILE not in CONFIG_PROFILES:
        raise ValueError(f"ACTIVE_PROFILE '{DEFAULT_PROFILE}' is not defined in CONFIG_PROFILES")

    cfg = dict(CONFIG_PROFILES[args.profile])
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def describe_cuda_environment() -> None:
    if not torch.cuda.is_available():
        print("CUDA available: False")
        return

    print("CUDA available: True")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")


def resolve_run_settings(cfg, *, pilot: bool, pilot_epochs: int):
    run_name = cfg["name"]
    run_epochs = cfg["epochs"]
    run_patience = cfg["patience"]

    if pilot:
        run_epochs = max(1, min(run_epochs, pilot_epochs))
        run_patience = max(1, min(run_patience, run_epochs))
        run_name = f"{run_name}_pilot"

    return run_name, run_epochs, run_patience


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
    run_name, run_epochs, run_patience = resolve_run_settings(
        cfg,
        pilot=args.pilot,
        pilot_epochs=args.pilot_epochs,
    )

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
    print(f"Active profile default: {DEFAULT_PROFILE}")
    print(f"Pilot mode: {args.pilot}")
    print(f"Using device: {device}")
    print(f"Resolved batch: {batch}")
    print(f"Resolved workers: {cfg['workers']}")
    print(f"AMP enabled: {cfg['amp']}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Training data: {data_path.resolve() if data_path.exists() else data_path}")
    print(f"Saving runs to: {project_path.resolve()}")
    print(f"Run name: {run_name}")
    print(f"Run epochs: {run_epochs}")
    print(f"Run patience: {run_patience}")
    describe_cuda_environment()

    try:
        train_once(
            model,
            data=data_yaml_for_run,
            epochs=run_epochs,
            imgsz=cfg["imgsz"],
            batch=batch,
            name=run_name,
            patience=run_patience,
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
            epochs=run_epochs,
            imgsz=cfg["imgsz"],
            batch=safe_batch,
            name=run_name,
            patience=run_patience,
            device=device,
            workers=0,
            project=str(project_path),
            augment=augment,
            cache=False,
            amp=False,
        )


if __name__ == "__main__":
    main()

