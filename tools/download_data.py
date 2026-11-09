"""Download the MovieLens latest-small dataset (about 1 MB) into ./data.

The GroupLens licence asks that the data not be redistributed, so the CSV files are
not committed to this repository. Run this once before using the project.
"""
import io
import os
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    for name in archive.namelist():
        if name.endswith(".csv"):
            target = os.path.join(DEST, os.path.basename(name))
            with open(target, "wb") as f:
                f.write(archive.read(name))
            print("  saved", target)
    print("Done.")


if __name__ == "__main__":
    main()
