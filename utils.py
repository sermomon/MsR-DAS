
import os
import re
import csv
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def log(msg: str, show_memory: bool = True) -> None:
    """
    Print a timestamped log message to stdout.

    Optionally includes current process memory usage if psutil is available.
    If psutil is not installed, memory reporting is silently skipped regardless
    of show_memory.

    Parameters
    ----------
    msg : str
        Message to log.
    show_memory : bool, optional
        Whether to include memory usage in the log output (default: True).
        Has no effect if psutil is not installed.
    """

    timestamp = datetime.now().strftime("%H:%M:%S")

    if show_memory and _HAS_PSUTIL:
        mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f"  [{timestamp}] [{mem_gb:.2f}GB] {msg}")
    else:
        print(f"  [{timestamp}] {msg}")

def filename_to_date(filepath):
    """
    Extract and format datetime from DAS filename
    
    Extracts the datetime string from a DAS data filename following the pattern
    YYYY-MM-DDTHHMMSSZ and converts it to a human-readable format.
    
    Parameters:
    -----------
    filepath : str
        Full path to the DAS data file. Expected filename format includes
        a datetime string like '2021-11-02T103545Z'
    
    Returns:
    --------
    str
        Formatted datetime string as 'YYYY-MM-DD  HH:MM:SS UTC' if pattern
        is found, otherwise returns the original filename without extension
    """
    
    name = os.path.splitext(os.path.basename(filepath))[0]
    
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6}Z)', name)
    
    if match:
        dt = datetime.strptime(match.group(1), '%Y-%m-%dT%H%M%SZ')
        return dt.strftime('%Y-%m-%d  %H:%M:%S UTC')
    
    return name

def generate_dataset_report(
    dataset_path: str,
    output_csv: str = "dataset_report.csv",
    class_folders: dict = None,
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"),
) -> str:
    """
    Generate a CSV report of image filenames and their corresponding class labels

    Traverses a dataset directory structured into class subfolders, collects all
    image files with their relative paths and class labels, and writes the results
    to a CSV file. Also prints a summary to the console showing the image count
    and percentage per class, useful for detecting dataset imbalance.

    Expected dataset structure:
        dataset_path/
        ├── 0/
        │   ├── image_a.jpg
        │   └── image_b.png
        └── 1/
            ├── image_c.jpg
            └── image_d.png

    Parameters:
    -----------
    dataset_path : str
        Path to the root dataset folder containing one subfolder per class
        (e.g. 'C:/carpeta/run01/all').
    output_csv : str, optional
        File path for the generated CSV report. Defaults to 'dataset_report.csv'
        in the current working directory.
    class_folders : dict, optional
        Mapping from subfolder name to class label
        (e.g. {"0": 0, "1": 1}). If None, the subfolder name is used
        directly as the class label.
    image_extensions : tuple of str, optional
        Lowercase file extensions to treat as images. Defaults to
        ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp').

    Returns:
    --------
    str
        Absolute path of the generated CSV file.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {dataset_path}")

    records = []
    class_counts = {}

    for class_dir in class_dirs:
        folder_name = class_dir.name
        label = class_folders[folder_name] if class_folders and folder_name in class_folders else folder_name

        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        class_counts[label] = len(images)

        for img_path in sorted(images):
            records.append({
                "filename": img_path.name,
                "relative_path": str(img_path.relative_to(dataset_path)),
                "class": label,
            })

    if not records:
        raise ValueError("No images found in any class subfolder.")

    output_csv = Path(output_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "relative_path", "class"])
        writer.writeheader()
        writer.writerows(records)

    print(f"{'='*50}")
    print(f"  REPORT GENERATED: {output_csv.resolve()}")
    print(f"{'='*50}")
    print(f"  Total images  : {len(records)}")
    print(f"  Classes found : {len(class_counts)}")
    print()
    for cls, count in class_counts.items():
        pct = count / len(records) * 100
        print(f"    Class '{cls}': {count} images ({pct:.1f}%)")
    print(f"{'='*50}")

    return str(output_csv.resolve())

def reorganize_dataset(
    source_dir: str,
    csv_path: str,
    output_dir: str,
    copy: bool = True,
) -> None:
    """
    Reorganize image files into class subfolders based on a dataset CSV report

    Reads a CSV file generated by generate_dataset_report and moves or copies
    each image from a flat source directory into a structured output directory
    where each subfolder corresponds to a class label.

    Expected CSV columns:
        filename     : image file name (e.g. 'img_001.jpg')
        relative_path: original relative path within the dataset
        class        : class label used as the destination subfolder name

    Output structure:
        output_dir/
        ├── 0/
        │   ├── img_001.jpg
        │   └── img_002.jpg
        └── 1/
            ├── img_003.jpg
            └── img_004.jpg

    Parameters:
    -----------
    source_dir : str
        Path to the folder containing all source images (flat or nested).
    csv_path : str
        Path to the CSV file produced by generate_dataset_report.
    output_dir : str
        Path to the destination folder where the reorganized dataset will be saved.
    copy : bool, optional
        If True (default), files are copied to the output directory, leaving the
        source intact. If False, files are moved instead.

    Returns:
    --------
    None
    """
    source_dir = Path(source_dir)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"filename", "class"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        records = list(reader)

    if not records:
        raise ValueError("No records found in the CSV file.")

    source_index = {f.name: f for f in source_dir.rglob("*") if f.is_file()}

    action = shutil.copy2 if copy else shutil.move
    action_label = "Copied" if copy else "Moved"

    ok = 0
    skipped = 0

    print(f"{'='*50}")
    print(f"  SOURCE      : {source_dir}")
    print(f"  CSV         : {csv_path}")
    print(f"  DESTINATION : {output_dir}")
    print(f"  ACTION      : {'Copy' if copy else 'Move'}")
    print(f"{'='*50}")

    for record in records:
        filename = record["filename"].strip()
        label = record["class"].strip()

        src = source_index.get(filename)

        if src is None:
            print(f"  [SKIPPED] File not found in source: {filename}")
            skipped += 1
            continue

        dest_folder = output_dir / str(label)
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / filename

        if dest.exists():
            print(f"  [SKIPPED] Already exists at destination: {dest}")
            skipped += 1
            continue

        action(src, dest)
        ok += 1

    print(f"\n  {action_label} : {ok} files")
    print(f"  Skipped  : {skipped} files")
    print(f"{'='*50}")