from pathlib import Path

# This finds the directory where config.py lives (MatchMyJob/src)
# then takes the .parent to get to the root (MatchMyJob/)
ROOT_DIR = Path(__file__).resolve().parent.parent
print(f"Root directory: {ROOT_DIR}")

# Now you can define your sub-directories easily
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Example of a specific file path
USER_INPUT_FILE = RAW_DATA_DIR / "user_survey_input.csv"