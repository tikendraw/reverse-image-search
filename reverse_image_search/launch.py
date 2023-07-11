import os
from pathlib import Path


def launch():
    os.system(f"streamlit run {Path(__file__).parent/'app.py'}")


def main():
    print(Path(__file__).parent / "app.py")


if __name__ == "__main__":
    main()
