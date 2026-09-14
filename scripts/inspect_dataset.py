"""
Phase 1, Step 1: Inspect the raw twcs.csv dataset.

Loads the dataset, reports shape, columns, dtypes, missing values,
sample rows, and basic text statistics.
"""
import csv
import sys
from pathlib import Path

DATA_PATH = Path("data/raw/twcs.csv")


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please download twcs.csv and place it in data/raw/")
        sys.exit(1)

    # --- Basic file info ---
    file_size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # --- Count rows without loading full dataset into memory ---
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)
    print(f"Total rows (excl header): {row_count:,}")
    print(f"Columns: {header}")

    # --- Load a sample to inspect ---
    import pandas as pd

    print("\n--- Loading first 50,000 rows for inspection ---")
    df_sample = pd.read_csv(DATA_PATH, nrows=50_000, dtype=str)

    print(f"\nShape of sample: {df_sample.shape}")
    print(f"\nColumn dtypes:\n{df_sample.dtypes}")

    # --- Missing values in sample ---
    missing = df_sample.isnull().sum()
    missing_pct = (missing / len(df_sample) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
    print(f"\nMissing values in first 50K rows:\n{missing_df}")

    # --- Text column stats ---
    if "text" in df_sample.columns:
        text_lengths = df_sample["text"].str.len()
        print(f"\nText length stats:")
        print(f"  Mean: {text_lengths.mean():.0f}")
        print(f"  Median: {text_lengths.median():.0f}")
        print(f"  Min: {text_lengths.min():.0f}")
        print(f"  Max: {text_lengths.max():.0f}")
        print(f"  Empty/NaN text: {df_sample['text'].isna().sum()}")

    # --- Unique values for key columns ---
    for col in ["author_id", "inbound", "in_response_to_tweet_id", "conversation_id"]:
        if col in df_sample.columns:
            nunique = df_sample[col].nunique()
            print(f"\n{col}: {nunique:,} unique values")
            if col == "inbound":
                print(f"  Value counts:\n{df_sample[col].value_counts().to_string()}")

    # --- Sample rows ---
    print("\n--- Sample rows (first 5) ---")
    print(df_sample.head().to_string())

    # --- Check for conversation_id column ---
    if "conversation_id" in df_sample.columns:
        print(f"\nconversation_id exists. Unique conversations in sample: "
              f"{df_sample['conversation_id'].nunique():,}")
    else:
        print("\nNo 'conversation_id' column found. Thread reconstruction will be needed.")

    print("\nDone. Inspection complete.")


if __name__ == "__main__":
    main()
