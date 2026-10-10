"""
Download the competition data and the original dataset into ./data.

Needs a Kaggle API token (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY)
and the competition rules accepted at
https://www.kaggle.com/competitions/playground-series-s6e10/rules

Resulting layout (src/train.py finds the files by these folder names):
    data/playground-series-s6e10/{train,test,sample_submission}.csv
    data/airline-passenger-satisfaction/{train,test}.csv
"""
import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COMPETITION = "playground-series-s6e10"
ORIGINAL = "teejmahal20/airline-passenger-satisfaction"


def unzip_all(folder):
    for name in os.listdir(folder):
        if name.endswith(".zip"):
            path = os.path.join(folder, name)
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
            os.remove(path)


def main():
    api = KaggleApi()
    api.authenticate()

    comp_dir = os.path.join(ROOT, COMPETITION)
    os.makedirs(comp_dir, exist_ok=True)
    print(f"Downloading competition data -> {comp_dir}")
    api.competition_download_files(COMPETITION, path=comp_dir, quiet=False)
    unzip_all(comp_dir)

    orig_dir = os.path.join(ROOT, "airline-passenger-satisfaction")
    os.makedirs(orig_dir, exist_ok=True)
    print(f"Downloading original dataset -> {orig_dir}")
    api.dataset_download_files(ORIGINAL, path=orig_dir, unzip=True, quiet=False)

    for d in (comp_dir, orig_dir):
        print(d, sorted(os.listdir(d)))


if __name__ == "__main__":
    main()
