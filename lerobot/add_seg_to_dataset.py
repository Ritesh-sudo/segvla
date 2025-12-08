import shutil
from pathlib import Path
import sys
from io import BytesIO
import math
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
)

DATASET_PATH = Path("./libero")
IMAGE_COLS = ["observation.images.image", "observation.images.image2"]
MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
DEVICE = "cuda"
BATCH_SIZE = 64
NUM_CLASSES = 150

def seg_map_to_bytes(seg_map: np.ndarray) -> bytes:
    """Convert segmentation map (H, W) uint8 to raw bytes.

    We store as raw bytes instead of PNG because:
    1. HuggingFace's 'image' dtype will decode PNG to PIL and apply transforms
    2. transforms.ToTensor() converts grayscale to float in [0,1] range
    3. We need raw uint8 class indices (0-149) for one-hot conversion
    """
    return seg_map.tobytes()


def load_seg_model():
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    return proc, model


def run_seg(proc, model, imgs):
    """Run segmentation, return class indices (B, H, W) uint8."""
    inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.float16):
        outputs = model(**inputs)

    target_sizes = [(img.shape[0], img.shape[1]) for img in imgs]
    seg_maps = proc.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    
    seg_maps = torch.stack(seg_maps, dim=0).cpu().numpy().astype(np.uint8)
    return seg_maps


def to_numpy_image(entry):
    """Convert various image formats to numpy array (H, W, 3) uint8."""
    if isinstance(entry, dict):
        if entry.get("bytes") is not None:
            with BytesIO(entry["bytes"]) as buf:
                img = Image.open(buf).convert("RGB")
        elif entry.get("path") is not None:
            img = Image.open(entry["path"]).convert("RGB")
        else:
            raise ValueError("Unsupported image dict format")
    elif isinstance(entry, (str, Path)):
        img = Image.open(entry).convert("RGB")
    else:
        img = entry
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def augment_parquet(parquet_path: Path, image_cols, proc, model):
    """Process a single parquet file IN PLACE and add segmentation columns."""

    # Read entire file once
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Check if segmentation columns already exist in the file
    for col in image_cols:
        seg_col = f"{col}_segmentation"
        if seg_col in df.columns:
            print(f"[skip] {seg_col} already exists in {parquet_path.name}")
            return

    # Process each image column
    for col in image_cols:
        seg_col = f"{col}_segmentation"

        # Decode all images at once
        imgs = [to_numpy_image(x) for x in df[col]]
        num_imgs = len(imgs)
        h, w = imgs[0].shape[:2]

        # Pre-allocate segmentation array
        seg_maps = np.empty((num_imgs, h, w), dtype=np.uint8)

        # Process in batches
        for i in range(0, num_imgs, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, num_imgs)
            mini = imgs[i:end_idx]
            seg_maps[i:end_idx] = run_seg(proc, model, mini)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # Convert to raw bytes format (not PNG - see seg_map_to_bytes docstring)
        seg_entries = []
        for idx, seg in enumerate(seg_maps):
            seg_bytes = seg_map_to_bytes(seg)
            path_str = df[col].iloc[idx]['path']

            seg_entries.append({
                'bytes': seg_bytes,
                'path': path_str,
                'shape': [h, w]  # Store shape for reconstruction
            })

        df[seg_col] = seg_entries

    # Write back to parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path)
    # print(f"[updated] {parquet_path}")


def update_info_json(dataset_path: Path, image_cols: list, img_shape: tuple = (256, 256)):
    """Update info.json with segmentation feature definitions."""
    info_path = dataset_path / "meta" / "info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    for col in image_cols:
        seg_col = f"{col}_segmentation"
        # Store metadata about segmentation format
        # NOTE: Stored as raw bytes (seg_map.tobytes()), not as PNG
        # This preserves uint8 class indices for one-hot conversion during training
        info["features"][seg_col] = {
            "dtype": "binary",  # Raw bytes, not 'image' type
            "shape": list(img_shape),
            "names": ["height", "width"],
            "fps": 10.0,
            "encoding": "raw_bytes",
            "num_classes": NUM_CLASSES,
            "description": "Semantic segmentation class indices stored as raw uint8 bytes"
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"[updated] {info_path}")


def main():
    print(f"Processing dataset at: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)

    data_dir = DATASET_PATH / "data"
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    proc, model = load_seg_model()

    chunk_dirs = sorted(data_dir.glob("chunk-*"))
    if not chunk_dirs:
        print(f"Error: No chunk directories found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_dirs)} chunk directories")

    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("file-*.parquet"))
        print(f"\nProcessing {chunk_dir.name} with {len(parquet_files)} parquet files")

        for pq_file in tqdm(parquet_files, desc=f"chunk {chunk_dir.name}"):
            file_num = int(str(pq_file).split("/")[3].split("-")[1].split(".")[0])
            if file_num < 12:
                continue
            
            augment_parquet(pq_file, IMAGE_COLS, proc, model)

    update_info_json(DATASET_PATH, IMAGE_COLS)

    print(f"\n✓ Done! Dataset augmented with segmentation data at: {DATASET_PATH}")


if __name__ == "__main__":
    main()