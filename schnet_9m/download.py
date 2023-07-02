import os
from jsonargparse import CLI


def download(root: str):
    """
    Downloads the QM1B dataset to a local folder

    Args:
        root (str): the root folder for storing a local copy of QM1B
    """
    args = ["--recursive", "--no-sign-request", "--exclude", "*raw-split*"]
    args = " ".join(args)
    source = "s3://graphcore-research-public/qm1b-dataset/"
    os.system(f"aws s3 cp {source} {root} {args}")


if __name__ == "__main__":
    CLI(download)
