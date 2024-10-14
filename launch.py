import os
from pathlib import Path


def launch():
    command = f"streamlit run {Path(__file__).parent/'v2/app.py'}"
    print("command: ", command)
    os.system(command)


def main():
    print(Path(__file__).parent / "v2/app.py")


if __name__ == "__main__":
    main()
    launch()
