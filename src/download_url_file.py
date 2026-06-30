#!/usr/bin/env python3

from pathlib import Path
import argparse
import requests
from tqdm import tqdm


def download_url_file(url: str, output_dir: str | Path) -> Path:
    """
    Download a file from a URL into the specified directory.

    Parameters
    ----------
    url : str
        URL of the file to download.
    output_dir : str | Path
        Destination directory.

    Returns
    -------
    Path
        Path to the downloaded (or existing) file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(url).name
    filepath = output_dir / filename

    if filepath.exists():
        print(f"✓ File already exists:\n{filepath}")
        return filepath

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    print(f"✓ Download completed:\n{filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL.")

    parser.add_argument(
        "--url",
        required=True,
        help="URL of the file to download.",
    )

    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory where the file will be stored (default: ./results).",
    )

    args = parser.parse_args()

    download_url_file(args.url, args.output_dir)


if __name__ == "__main__":
    main()