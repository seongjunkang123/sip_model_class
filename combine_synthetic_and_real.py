import pandas as pd
from pathlib import Path


def get_project_root() -> Path:
    """Project root is the parent of sip_model_class."""
    return Path(__file__).resolve().parent.parent


def load_synthetic_data(synthetic_dir: Path) -> pd.DataFrame:
    pattern = "synthetic_data_*.csv"
    files = sorted(synthetic_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {synthetic_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def combine_synthetic_and_real(
    synthetic_dir: Path | None = None,
    real_data_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    root = get_project_root()
    synthetic_dir = synthetic_dir or root / "sip_model_gen" / "synthetic_data"
    real_data_path = real_data_path or root / "sip_data" / "res1" / "combined_data.csv"

    real_df = pd.read_csv(real_data_path)
    synthetic_df = load_synthetic_data(synthetic_dir)

    # Ensure column alignment (same order)
    if list(real_df.columns) != list(synthetic_df.columns):
        common = [c for c in real_df.columns if c in synthetic_df.columns]
        real_df = real_df[common]
        synthetic_df = synthetic_df[common]

    combined = pd.concat([real_df, synthetic_df], ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return combined


if __name__ == "__main__":
    root = get_project_root()
    out = root / "sip_data" / "res1" / "combined_synthetic_and_real.csv"
    df = combine_synthetic_and_real(output_path=out)
    print(f"Combined shape: {df.shape}")
    print(f"Real + synthetic saved to: {out}")
