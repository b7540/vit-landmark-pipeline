from __future__ import annotations
import json
import math
import os
import sys
import tarfile
import threading
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    HAS_TORCH = True

    class TorchDS(Dataset):
        def __init__(self, items, size):
            self.items = items
            self.tf = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self): return len(self.items)

        def __getitem__(self, i):
            path, y = self.items[i]
            return self.tf(Image.open(path).convert("RGB")), y

except ImportError:
    HAS_TORCH = False
    Dataset = object

DEFAULT_DATA_ROOT = Path(r"F:\Landmark")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

@dataclass
class Config:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    image_size: int = 224
    batch_size: int = 32
    backbone: str = "vit_b_16"  
    max_train_samples: Optional[int] = None
    resume_checkpoint: Optional[Path] = None  # NEW: Added support for loading models

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "metadata"

    @property
    def train_dir(self) -> Path:
        return self.data_root / "train"

def load_training_samples(config: Config):
    csv_path = config.metadata_dir / "train.csv"
    if not csv_path.exists():
        print(f"[Error] Missing {csv_path}. Run 'Prepare Data' first.")
        return None, None, None

    print("Scanning extracted images on disk... this might take a moment.")
    found_files = list(config.train_dir.rglob("*.jpg"))
    if not found_files:
        print(f"[Error] No JPGs found under {config.train_dir}. Run 'Prepare Data' first.")
        return None, None, None

    id_to_path = {p.stem: p for p in found_files}
    print(f"Found {len(found_files)} images on disk. Matching with labels from CSV...")
    
    df = pd.read_csv(csv_path)
    df_valid = df[df['id'].isin(id_to_path.keys())]
    
    if config.max_train_samples:
        df_valid = df_valid.head(config.max_train_samples)

    samples = []
    for row in df_valid.itertuples(index=False):
        samples.append((id_to_path[str(row.id)], int(row.landmark_id)))

    if not samples:
        print(f"[Error] Images were found on disk, but they didn't match any IDs in {csv_path}.")
        return None, None, None

    labels = sorted({lid for _, lid in samples})
    label_map = {lid: i for i, lid in enumerate(labels)}
    inverse = {i: lid for lid, i in label_map.items()}
    indexed = [(p, label_map[lid]) for p, lid in samples]
    
    print(f"Successfully matched {len(indexed)} training samples!")
    return indexed, label_map, inverse

def create_label_formatter(metadata_dir: Path):
    hier_path = metadata_dir / "train_label_to_hierarchical.csv"
    cat_path = metadata_dir / "train_label_to_category.csv"
    
    hier_map, cat_map = {}, {}
    
    if hier_path.exists():
        df_hier = pd.read_csv(hier_path)
        hier_map = dict(zip(df_hier['landmark_id'], df_hier['hierarchical_label']))
        
    if cat_path.exists():
        df_cat = pd.read_csv(cat_path)
        cat_map = dict(zip(df_cat['landmark_id'], df_cat['category']))

    def format_label(landmark_id: int) -> str:
        parts = [f"landmark_id={landmark_id}"]
        if landmark_id in hier_map: parts.append(str(hier_map[landmark_id]))
        if landmark_id in cat_map: parts.append(str(cat_map[landmark_id]))
        return " | ".join(parts)
        
    return format_label

def collect_images(patterns: list[str]) -> list[Path]:
    paths = []
    for item in patterns:
        p = Path(item)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(sorted(p.glob(ext)))
        elif p.is_file():
            paths.append(p)
    return paths

def save_label_meta(checkpoint_dir: Path, name: str, label_map: dict, backbone: str, framework: str):
    meta = {
        "framework": framework,
        "num_classes": len(label_map),
        "label_map": {str(k): v for k, v in label_map.items()},
        "backbone": backbone,
        "image_size": 224,
    }
    path = checkpoint_dir / name
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path

def cmd_prepare(config: Config, skip_tar: bool) -> None:
    print("--- Starting Data Preparation ---")
    meta = config.metadata_dir
    meta.mkdir(parents=True, exist_ok=True)
    dest = meta / "train.csv"

    if not dest.exists():
        zpath = None
        for name in ("train.csv.zip", "train.csv.ZIP"):
            candidate = config.data_root / name
            if candidate.is_file():
                zpath = candidate
                break
        if zpath:
            print(f"Extracting {zpath.name} -> train.csv")
            with zipfile.ZipFile(zpath) as zf:
                member = next(m for m in zf.namelist() if m.lower().endswith("train.csv"))
                dest.write_bytes(zf.read(member))
        else:
            print(f"[Abort] Put train.csv or train.csv.zip in {config.data_root}")
            return

    print(f"Labels CSV ready: {dest}")
    if skip_tar:
        print("Data preparation finished (Tar files skipped).")
        return

    train_dir = config.train_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    tar_files = sorted(config.data_root.glob("images_*.tar"))
    
    if not tar_files:
        print("No images_*.tar files found to extract.")

    for tar_path in tar_files:
        marker = train_dir / f".extracted_{tar_path.stem}"
        if marker.exists():
            print(f"Skip (already extracted): {tar_path.name}")
            continue
        print(f"Extracting {tar_path.name} ...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(train_dir)
        marker.touch()

    n = sum(1 for _ in train_dir.rglob("*.jpg"))
    print(f"Data Preparation Ready: {n} images in {train_dir}")

def train_pytorch(config: Config, epochs: int, lr: float, val_fraction: float, progress_callback=None) -> Optional[Path]:
    if not HAS_TORCH:
        print("[Error] PyTorch or Torchvision is not installed.")
        return None

    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import models

    samples, label_map, _ = load_training_samples(config)
    if samples is None: return None

    ds = TorchDS(samples, config.image_size)
    n_val = max(1, int(len(ds) * val_fraction))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = min(4, os.cpu_count() or 1)
    use_pin = True if device.type == "cuda" else False 
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=workers, pin_memory=use_pin)

    net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    in_features = net.heads.head.in_features
    net.heads.head = nn.Linear(in_features, len(label_map))

    # NEW: Load Model (Resume Training)
    if config.resume_checkpoint and config.resume_checkpoint.exists():
        print(f"Loading checkpoint from {config.resume_checkpoint} to resume training...")
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model_state"])

    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.checkpoint_dir / "best.pt"
    best_acc = 0.0
    print(f"\n[PyTorch Training - ViT] {len(samples)} images, {len(label_map)} classes, device={device}")

    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        net.train()
        t_ok = t_n = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            crit(out, y).backward()
            opt.step()
            t_ok += (out.argmax(1) == y).sum().item()
            t_n += len(y)

            if progress_callback:
                percent = ((i + 1) / total_batches) * 100
                progress_callback(percent, f"PyTorch Epoch {epoch}/{epochs} | Training Batch {i+1}/{total_batches} ({percent:.1f}%)")

        if progress_callback:
            progress_callback(100, f"PyTorch Epoch {epoch}/{epochs} | Running Validation...")

        net.eval()
        v_ok = v_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_ok += (net(x).argmax(1) == y).sum().item()
                v_n += len(y)

        tr_acc = t_ok / max(t_n, 1)
        va_acc = v_ok / max(v_n, 1)
        print(f"  Epoch {epoch}/{epochs} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": net.state_dict(),
                "label_map": label_map,
                "backbone": config.backbone,
                "image_size": config.image_size,
                "num_classes": len(label_map),
            }, ckpt_path)

    save_label_meta(config.checkpoint_dir, "label_map_pt.json", label_map, config.backbone, "pytorch")
    print(f"[PyTorch] Process completed. Model saved to: {ckpt_path}")
    if progress_callback:
        progress_callback(100, "PyTorch Training Complete!")
    return ckpt_path

def train_tensorflow(config: Config, epochs: int, lr: float, val_fraction: float, progress_callback=None) -> Optional[Path]:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        print("[Error] TensorFlow or tensorflow_hub is not installed.")
        return None

    samples, label_map, _ = load_training_samples(config)
    if samples is None: return None

    paths = [str(p) for p, _ in samples]
    labels = [y for _, y in samples]
    n = len(paths)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).shuffle(n, seed=42)
    train_ds, val_ds = ds.take(n_train), ds.skip(n_train)

    def load_fn(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [config.image_size, config.image_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = train_ds.map(load_fn, tf.data.AUTOTUNE).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(load_fn, tf.data.AUTOTUNE).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    # NEW: Load Model (Resume Training)
    if config.resume_checkpoint and config.resume_checkpoint.exists():
        print(f"Loading checkpoint from {config.resume_checkpoint} to resume training...")
        model = tf.keras.models.load_model(str(config.resume_checkpoint), custom_objects={'KerasLayer': hub.KerasLayer})
    else:
        vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
        print("Downloading/Loading ViT backbone from TF Hub...")
        base = hub.KerasLayer(vit_url, trainable=True)
        inp = tf.keras.Input(shape=(224, 224, 3))
        x = base(inp)
        out = tf.keras.layers.Dense(len(label_map), activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.checkpoint_dir / "best_tf.keras"
    print(f"\n[TensorFlow Training - ViT] {len(samples)} images, {len(label_map)} classes")

    total_batches = math.ceil(n_train / config.batch_size)
    
    class GUIProgressCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if progress_callback:
                percent = ((batch + 1) / total_batches) * 100
                progress_callback(percent, f"TF Training | Batch {batch+1}/{total_batches} ({percent:.1f}%)")
                
        def on_epoch_begin(self, epoch, logs=None):
            if progress_callback:
                progress_callback(0, f"TF Epoch {epoch+1}/{epochs} | Starting...")
                
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(100, f"TF Epoch {epoch+1}/{epochs} | Running Validation...")

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, mode="max")
    ]
    if progress_callback:
        callbacks_list.append(GUIProgressCallback())

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
    )
    
    save_label_meta(config.checkpoint_dir, "label_map_tf.json", label_map, config.backbone, "tensorflow")
    print(f"[TensorFlow] Process completed. Model saved to: {ckpt_path}")
    if progress_callback:
        progress_callback(100, "TensorFlow Training Complete!")
    return ckpt_path

def predict_pytorch(config: Config, images: list[Path], top_k: int, checkpoint: Optional[Path]) -> None:
    if not HAS_TORCH:
        print("[Error] PyTorch/Torchvision missing.")
        return

    import torch.nn as nn
    from torchvision import models

    ckpt_path = checkpoint or config.checkpoint_dir / "best.pt"
    if not ckpt_path.is_file():
        print(f"[Error] Missing checkpoint file: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    inverse = {v: int(k) for k, v in ckpt["label_map"].items()}
    size = ckpt.get("image_size", 224)

    net = models.vit_b_16(weights=None)
    in_features = net.heads.head.in_features
    net.heads.head = nn.Linear(in_features, ckpt["num_classes"])
    
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    tf_pipeline = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    format_label = create_label_formatter(config.metadata_dir)

    for path in images:
        if not path.is_file(): continue
        x = tf_pipeline(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)
        k = min(top_k, len(probs[0]))
        vals, idxs = probs[0].topk(k)
        print(f"\n[PyTorch Prediction] {path.name}")
        for p, i in zip(vals.tolist(), idxs.tolist()):
            print(f"  {p:.3f}  {format_label(inverse[i])}")

def predict_tensorflow(config: Config, images: list[Path], top_k: int, checkpoint: Optional[Path]) -> None:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        print("[Error] TensorFlow or tensorflow_hub missing.")
        return

    ckpt_path = checkpoint or config.checkpoint_dir / "best_tf.keras"
    if not ckpt_path.is_file():
        print(f"[Error] Missing checkpoint file: {ckpt_path}")
        return

    meta_path = config.checkpoint_dir / "label_map_tf.json"
    if not meta_path.is_file():
        print(f"[Error] Missing TensorFlow configuration mapping metadata file.")
        return
        
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inverse = {int(v): int(k) for k, v in meta["label_map"].items()}
    size = meta.get("image_size", 224)
    
    model = tf.keras.models.load_model(str(ckpt_path), custom_objects={'KerasLayer': hub.KerasLayer})

    format_label = create_label_formatter(config.metadata_dir)

    batch_arr = []
    valid_paths = []
    
    for path in images:
        if not path.is_file(): continue
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = np.array(img, dtype=np.float32)
        arr = arr / 255.0
        batch_arr.append(arr)
        valid_paths.append(path)
        
    if batch_arr:
        batch_arr = np.array(batch_arr)
        batch_probs = model.predict(batch_arr, verbose=0)
        
        for path, probs in zip(valid_paths, batch_probs):
            k = min(top_k, len(probs))
            idxs = np.argsort(probs)[-k:][::-1]
            print(f"\n[TensorFlow Prediction] {path.name}")
            for i in idxs:
                print(f"  {probs[i]:.3f}  {format_label(inverse[int(i)])}")

class ConsoleRedirector:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
    def write(self, string: str):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

class LandmarkGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Google Landmark Recognition Pipeline Engine (ViT Edition)")
        self.root.geometry("900x850")
        
        self.selected_predict_files: list[str] = []

        config_frame = ttk.LabelFrame(root, text=" Global Path Configuration ")
        config_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(config_frame, text="Data Root Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.data_root_var = tk.StringVar(value=str(DEFAULT_DATA_ROOT))
        ttk.Entry(config_frame, textvariable=self.data_root_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse...", command=self.browse_data_root).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(config_frame, text="Checkpoint Dir:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ckpt_dir_var = tk.StringVar(value=str(DEFAULT_CHECKPOINT_DIR))
        ttk.Entry(config_frame, textvariable=self.ckpt_dir_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse...", command=self.browse_ckpt_dir).grid(row=1, column=2, padx=5, pady=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_prepare = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_predict = ttk.Frame(self.notebook)
        self.tab_viewer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook) # NEW TAB

        self.notebook.add(self.tab_prepare, text="  1. Prepare Data  ")
        self.notebook.add(self.tab_train, text="  2. Train Models  ")
        self.notebook.add(self.tab_predict, text="  3. Run Predictions  ")
        self.notebook.add(self.tab_viewer, text="  4. Model Viewer  ")
        self.notebook.add(self.tab_manage, text="  5. Manage Models (Download)  ")

        self.setup_prepare_tab()
        self.setup_train_tab()
        self.setup_predict_tab()
        self.setup_viewer_tab()
        self.setup_manage_tab()

        console_frame = ttk.LabelFrame(root, text=" Live Execution Console Window Logging ")
        console_frame.pack(fill="both", padx=10, pady=5, expand=True)

        self.console_text = tk.Text(console_frame, wrap="word", height=12, bg="#1e1e1e", fg="#ffffff", insertbackground="white")
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        sys.stdout = ConsoleRedirector(self.console_text)
        sys.stderr = ConsoleRedirector(self.console_text)

        print("System initialized. Select an action tab above to begin pipeline operations.")

    def get_current_config(self) -> Config:
        return Config(
            data_root=Path(self.data_root_var.get().strip()),
            checkpoint_dir=Path(self.ckpt_dir_var.get().strip())
        )

    def browse_data_root(self):
        d = filedialog.askdirectory(initialdir=self.data_root_var.get())
        if d: self.data_root_var.set(d)

    def browse_ckpt_dir(self):
        d = filedialog.askdirectory(initialdir=self.ckpt_dir_var.get())
        if d: self.ckpt_dir_var.set(d)
        
    def browse_file_override(self, target_var: tk.StringVar):
        f = filedialog.askopenfilename()
        if f: target_var.set(f)

    def update_progress_ui(self, percent: float, text: str):
        def task():
            self.progress_var.set(percent)
            self.progress_label.config(text=text)
        self.root.after(0, task)

    def setup_prepare_tab(self):
        frame = self.tab_prepare
        self.skip_tar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Skip Images (.tar file extraction) - Process Metadata CSV only", variable=self.skip_tar_var).pack(anchor="w", padx=20, pady=20)
        self.btn_prep = ttk.Button(frame, text="Execute Data Pipeline Preparation Sequences", command=self.start_prepare)
        self.btn_prep.pack(padx=20, pady=10)

    def start_prepare(self):
        self.btn_prep.config(state="disabled")
        config = self.get_current_config()
        skip_tar = self.skip_tar_var.get()
        def run():
            try:
                cmd_prepare(config, skip_tar)
            except Exception as e:
                print(f"\n[Execution Exception Error]: {e}")
            finally:
                self.btn_prep.config(state="normal")
        threading.Thread(target=run, daemon=True).start()

    def setup_train_tab(self):
        frame = self.tab_train
        grid = ttk.Frame(frame)
        grid.pack(anchor="w", padx=20, pady=10)
        ttk.Label(grid, text="Framework Target:").grid(row=0, column=0, sticky="w", pady=5)
        self.train_framework_var = tk.StringVar(value="both")
        ttk.Combobox(grid, textvariable=self.train_framework_var, values=["pytorch", "tensorflow", "both"], state="readonly").grid(row=0, column=1, pady=5, padx=10)
        ttk.Label(grid, text="Training Epochs:").grid(row=1, column=0, sticky="w", pady=5)
        self.epochs_var = tk.IntVar(value=5)
        ttk.Entry(grid, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(grid, text="Learning Rate:").grid(row=2, column=0, sticky="w", pady=5)
        self.lr_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(grid, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(grid, text="Batch Size:").grid(row=3, column=0, sticky="w", pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(grid, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(grid, text="Max Train Samples (Blank for all):").grid(row=4, column=0, sticky="w", pady=5)
        self.max_samples_var = tk.StringVar(value="")
        ttk.Entry(grid, textvariable=self.max_samples_var, width=10).grid(row=4, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(grid, text="Validation Fraction:").grid(row=5, column=0, sticky="w", pady=5)
        self.val_fraction_var = tk.DoubleVar(value=0.1)
        ttk.Entry(grid, textvariable=self.val_fraction_var, width=10).grid(row=5, column=1, sticky="w", pady=5, padx=10)
        
        # NEW: Resume Checkpoint Picker
        ttk.Label(grid, text="Load Checkpoint to Resume Training (Optional):").grid(row=6, column=0, sticky="w", pady=5)
        self.resume_ckpt_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.resume_ckpt_var, width=30).grid(row=6, column=1, sticky="w", pady=5, padx=10)
        ttk.Button(grid, text="Browse...", command=lambda: self.browse_file_override(self.resume_ckpt_var)).grid(row=6, column=2, pady=5)

        self.btn_train = ttk.Button(frame, text="Launch Neural Network Model Optimization Training Loop Process", command=self.start_training)
        self.btn_train.pack(padx=20, pady=10, anchor="w")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100, length=450, mode='determinate')
        self.progress_bar.pack(padx=20, pady=5, anchor="w")
        self.progress_label = ttk.Label(frame, text="Status: Ready to train", font=("Segoe UI", 9, "italic"))
        self.progress_label.pack(padx=20, pady=0, anchor="w")

    def start_training(self):
        self.btn_train.config(state="disabled")
        config = self.get_current_config()
        try:
            config.batch_size = int(self.batch_size_var.get())
            ms = self.max_samples_var.get().strip()
            config.max_train_samples = int(ms) if ms else None
            if self.resume_ckpt_var.get().strip():
                config.resume_checkpoint = Path(self.resume_ckpt_var.get().strip())
        except ValueError:
            print("[Form Entry Input Error] Batch size and samples parameters must resolve to numerical values.")
            self.btn_train.config(state="normal")
            return
        framework = self.train_framework_var.get()
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        val_fraction = self.val_fraction_var.get()
        self.update_progress_ui(0, "Initializing dataloaders...")
        def run():
            try:
                if framework in ("pytorch", "both"):
                    train_pytorch(config, epochs, lr, val_fraction, progress_callback=self.update_progress_ui)
                if framework in ("tensorflow", "both"):
                    train_tensorflow(config, epochs, lr, val_fraction, progress_callback=self.update_progress_ui)
            except Exception as e:
                print(f"\n[Execution Training Exception Error Context Failure]: {e}")
                self.update_progress_ui(0, "Training Failed (See Console)")
            finally:
                self.btn_train.config(state="normal")
        threading.Thread(target=run, daemon=True).start()

    def setup_predict_tab(self):
        frame = self.tab_predict
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill="x", padx=20, pady=10)
        ttk.Button(file_frame, text="Select Target Images to Analyze...", command=self.select_predict_images).pack(side="left")
        self.lbl_selected_count = ttk.Label(file_frame, text="No images loaded.")
        self.lbl_selected_count.pack(side="left", padx=15)
        grid = ttk.Frame(frame)
        grid.pack(anchor="w", padx=20, pady=10)
        ttk.Label(grid, text="Framework Engine Inference:").grid(row=0, column=0, sticky="w", pady=5)
        self.predict_framework_var = tk.StringVar(value="both")
        ttk.Combobox(grid, textvariable=self.predict_framework_var, values=["pytorch", "tensorflow", "both"], state="readonly").grid(row=0, column=1, pady=5, padx=10)
        ttk.Label(grid, text="Top K Probabilities List Metrics:").grid(row=1, column=0, sticky="w", pady=5)
        self.top_k_var = tk.IntVar(value=5)
        ttk.Entry(grid, textvariable=self.top_k_var, width=8).grid(row=1, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(grid, text="Custom PyTorch Checkpoint Path (Optional Override):").grid(row=2, column=0, sticky="w", pady=5)
        self.custom_pt_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.custom_pt_var, width=40).grid(row=2, column=1, pady=5, padx=10)
        ttk.Button(grid, text="Browse File...", command=lambda: self.browse_file_override(self.custom_pt_var)).grid(row=2, column=2, pady=5)
        ttk.Label(grid, text="Custom TensorFlow Checkpoint Path (Optional Override):").grid(row=3, column=0, sticky="w", pady=5)
        self.custom_tf_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.custom_tf_var, width=40).grid(row=3, column=1, pady=5, padx=10)
        ttk.Button(grid, text="Browse File...", command=lambda: self.browse_file_override(self.custom_tf_var)).grid(row=3, column=2, pady=5)
        self.btn_predict = ttk.Button(frame, text="Execute Prediction Matrix Evaluation", command=self.start_predict)
        self.btn_predict.pack(padx=20, pady=15, anchor="w")

    def select_predict_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.selected_predict_files = list(files)
            self.lbl_selected_count.config(text=f"Selected {len(files)} target files for evaluation processing queues.")

    def start_predict(self):
        if not self.selected_predict_files:
            messagebox.showwarning("Missing Inputs Target", "Please select at least one input validation image prior to triggering evaluation engine calculations.")
            return
        self.btn_predict.config(state="disabled")
        config = self.get_current_config()
        paths = collect_images(self.selected_predict_files)
        framework = self.predict_framework_var.get()
        top_k = self.top_k_var.get()
        pt_ckpt = Path(self.custom_pt_var.get().strip()) if self.custom_pt_var.get().strip() else None
        tf_ckpt = Path(self.custom_tf_var.get().strip()) if self.custom_tf_var.get().strip() else None
        def run():
            try:
                if framework in ("pytorch", "both"):
                    predict_pytorch(config, paths, top_k, pt_ckpt)
                if framework in ("tensorflow", "both"):
                    predict_tensorflow(config, paths, top_k, tf_ckpt)
            except Exception as e:
                print(f"\n[Prediction Routine Processing Execution Failure]: {e}")
            finally:
                self.btn_predict.config(state="normal")
        threading.Thread(target=run, daemon=True).start()

    def setup_viewer_tab(self):
        frame = self.tab_viewer
        ttk.Label(frame, text="Inspect Trained Neural Network Architecture", font=("Segoe UI", 12, "bold")).pack(pady=20)
        ttk.Label(frame, text="This will load the saved model files and print their internal mathematical structure to the console below.").pack(pady=5)
        self.btn_view_pt = ttk.Button(frame, text="Inspect PyTorch Model (best.pt)", command=self.view_pytorch_model)
        self.btn_view_pt.pack(pady=10)
        self.btn_view_tf = ttk.Button(frame, text="Inspect TensorFlow Model (best_tf.keras)", command=self.view_tensorflow_model)
        self.btn_view_tf.pack(pady=10)

    def view_pytorch_model(self):
        self.btn_view_pt.config(state="disabled")
        def run():
            print("\n==================================================")
            print(" INSPECTING PYTORCH BRAIN (best.pt)")
            print("==================================================")
            try:
                if not HAS_TORCH:
                    print("Error: PyTorch is not installed.")
                    return
                import torch.nn as nn
                from torchvision import models
                ckpt_path = self.get_current_config().checkpoint_dir / "best.pt"
                if not ckpt_path.exists():
                    print(f"Error: Could not find {ckpt_path}. Have you trained the model yet?")
                    return
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                pt_model = models.vit_b_16(weights=None)
                in_features = pt_model.heads.head.in_features
                pt_model.heads.head = nn.Linear(in_features, ckpt["num_classes"])
                pt_model.load_state_dict(ckpt["model_state"])
                print(pt_model)
                print(f"\nTotal PyTorch classes (Landmarks) learned: {ckpt['num_classes']}")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
            finally:
                self.root.after(0, lambda: self.btn_view_pt.config(state="normal"))
        threading.Thread(target=run, daemon=True).start()

    def view_tensorflow_model(self):
        self.btn_view_tf.config(state="disabled")
        def run():
            print("\n==================================================")
            print(" INSPECTING TENSORFLOW BRAIN (best_tf.keras)")
            print("==================================================")
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                ckpt_path = self.get_current_config().checkpoint_dir / "best_tf.keras"
                if not ckpt_path.exists():
                    print(f"Error: Could not find {ckpt_path}. Have you trained the model yet?")
                    return
                tf_model = tf.keras.models.load_model(str(ckpt_path), custom_objects={'KerasLayer': hub.KerasLayer})
                tf_model.summary(print_fn=print)
            except Exception as e:
                print(f"Error loading TensorFlow model: {e}")
            finally:
                self.root.after(0, lambda: self.btn_view_tf.config(state="normal"))
        threading.Thread(target=run, daemon=True).start()

    # NEW: Setup Manage / Download Tab
    def setup_manage_tab(self):
        frame = self.tab_manage
        ttk.Label(frame, text="Download Remote Checkpoints", font=("Segoe UI", 12, "bold")).pack(pady=20)
        ttk.Label(frame, text="Enter a direct URL to a pre-trained model to download it directly into your local Checkpoint Directory.").pack(pady=5)
        
        dl_frame = ttk.Frame(frame)
        dl_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(dl_frame, text="Direct File URL:").grid(row=0, column=0, sticky="w", pady=5)
        self.dl_url_var = tk.StringVar()
        ttk.Entry(dl_frame, textvariable=self.dl_url_var, width=70).grid(row=0, column=1, pady=5, padx=10)
        
        ttk.Label(dl_frame, text="Save As (Filename):").grid(row=1, column=0, sticky="w", pady=5)
        self.dl_name_var = tk.StringVar(value="downloaded_model.pt")
        ttk.Entry(dl_frame, textvariable=self.dl_name_var, width=30).grid(row=1, column=1, sticky="w", pady=5, padx=10)
        
        self.btn_download = ttk.Button(dl_frame, text="Download Model Checkpoint", command=self.download_model)
        self.btn_download.grid(row=2, column=1, sticky="w", pady=15, padx=10)

    # NEW: Download Model Logic
    def download_model(self):
        url = self.dl_url_var.get().strip()
        name = self.dl_name_var.get().strip()
        if not url or not name:
            print("[Error] Please provide both a URL and a valid filename.")
            return
            
        config = self.get_current_config()
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dest = config.checkpoint_dir / name
        self.btn_download.config(state="disabled")
        
        def run():
            try:
                print(f"\n[Download Initiated] Target: {url}")
                print(f"Saving to: {dest}")
                urllib.request.urlretrieve(url, dest)
                print("[Success] Model downloaded completely!")
            except Exception as e:
                print(f"\n[Download Failed] Error: {e}")
            finally:
                self.btn_download.config(state="normal")
                
        threading.Thread(target=run, daemon=True).start()

def main() -> None:
    root = tk.Tk()
    app = LandmarkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
