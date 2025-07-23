import os
import requests
import tarfile
from tqdm import tqdm

# Get the absolute path to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Compute the grandparent directory
GRANDPARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Target directory: MRVD in grandparent directory
OUTPUT_DIR = os.path.join(GRANDPARENT_DIR, "MRVD")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URLs to download
FILES_TO_DOWNLOAD = {
    "YouTubeClips.tar": "https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar",
    "AllVideoDescriptions.txt": "https://www.cs.utexas.edu/~ml/clamp/videoDescription/AllVideoDescriptions.txt"
}

def download_file(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def extract_tar(tar_path, extract_to):
    print(f"Extracting {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")

if __name__ == "__main__":
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Preparing to download {filename}...")
        download_file(url, dest_path)

        if filename.endswith(".tar"):
            extract_tar(dest_path, OUTPUT_DIR)