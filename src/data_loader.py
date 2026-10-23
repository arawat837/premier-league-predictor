"""Data loading and standardization module for Premier League match data.

Downloads historical match CSVs directly from Football-Data.co.uk,
standardizes schemas, cleans missing/malformed rows, and maps target results.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd

# Base URL pattern for Football-Data.co.uk Premier League (division E0)
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

# Seasons to download: 2020-21 through 2025-26
SEASONS: Dict[str, str] = {
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

# Target columns required by specification
REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
]

# Numeric columns to cast
NUMERIC_COLUMNS = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]

# Mapping for full-time result: 0: Away Win (A), 1: Draw (D), 2: Home Win (H)
FTR_MAPPING = {
    "A": 0,
    "D": 1,
    "H": 2,
}


def download_season(
    season_code: str,
    output_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 30,
) -> Optional[Path]:
    """Download a single season CSV from Football-Data.co.uk.

    Args:
        season_code: The 4-digit season code used by Football-Data (e.g. '2021').
        output_dir: Path where raw CSV should be saved.
        filename: Optional output filename. Defaults to 'E0_{season_code}.csv'.
        timeout: Request timeout in seconds.

    Returns:
        Path to the saved CSV file, or None if download failed (e.g., 404).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"E0_{season_code}.csv"
    output_path = output_dir / filename

    url = BASE_URL.format(season_code=season_code)
    print(f"Downloading {url} -> {output_path}...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f"Successfully saved {output_path} ({len(response.content)} bytes)")
            return output_path
        elif response.status_code == 404:
            print(f"Warning: Season {season_code} returned 404 (file not available yet on server).")
            return None
        else:
            response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error downloading season {season_code}: {exc}")
        return None


def download_all_seasons(
    seasons: Optional[Dict[str, str]] = None,
    raw_dir: Optional[Path] = None,
) -> List[Path]:
    """Download all configured Premier League seasons to raw data directory.

    Args:
        seasons: Mapping from season label to code. Defaults to SEASONS.
        raw_dir: Output directory. Defaults to 'data/raw'.

    Returns:
        List of Paths to successfully downloaded CSV files.
    """
    if seasons is None:
        seasons = SEASONS
    if raw_dir is None:
        raw_dir = Path("data") / "raw"

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: List[Path] = []
    for label, code in seasons.items():
        dest_filename = f"E0_{code}.csv"
        saved_file = download_season(code, raw_dir, filename=dest_filename)
        if saved_file and saved_file.exists():
            downloaded_files.append(saved_file)

    return downloaded_files


def standardize_dataframe(df: pd.DataFrame, source_season: Optional[str] = None) -> pd.DataFrame:
    """Standardize and clean a single raw match DataFrame.

    - Strips whitespace from column names and string columns
    - Validates required columns exist
    - Drops empty/unplayed fixtures and invalid target values
    - Parses Date to datetime objects
    - Casts numeric stats to int
    - Maps FTR: Away Win (A) -> 0, Draw (D) -> 1, Home Win (H) -> 2

    Args:
        df: Raw DataFrame from CSV.
        source_season: Optional season label for metadata.

    Returns:
        Standardized DataFrame containing only REQUIRED_COLUMNS.
    """
    # Clean column names
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Verify all required columns exist
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Select only required columns
    standardized = df[REQUIRED_COLUMNS].copy()

    # Strip string columns
    for str_col in ["HomeTeam", "AwayTeam", "FTR"]:
        if str_col in standardized.columns:
            standardized[str_col] = standardized[str_col].astype(str).str.strip()

    # Drop unplayed matches or rows missing key fields
    standardized = standardized.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    standardized = standardized[standardized["FTR"].isin(["A", "D", "H"])]

    # Parse Date (day-first format common in Football-Data UK: DD/MM/YYYY or DD/MM/YY)
    standardized["Date"] = pd.to_datetime(
        standardized["Date"], dayfirst=True, format="mixed"
    )

    # Cast numeric columns to integer
    for num_col in NUMERIC_COLUMNS:
        standardized[num_col] = pd.to_numeric(
            standardized[num_col], errors="coerce"
        ).fillna(0).astype(int)

    # Map target FTR to integers: 0 for Away (A), 1 for Draw (D), 2 for Home (H)
    standardized["FTR"] = standardized["FTR"].map(FTR_MAPPING).astype(int)

    return standardized


def load_and_standardize_all(
    raw_dir: Optional[Path] = None,
    seasons: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load all downloaded raw CSVs, standardize each, combine, and sort chronologically.

    Args:
        raw_dir: Path to directory containing raw CSVs. Defaults to 'data/raw'.
        seasons: Mapping from season label to code. Defaults to SEASONS.

    Returns:
        Combined and chronologically sorted DataFrame with standardized schema.
    """
    if raw_dir is None:
        raw_dir = Path("data") / "raw"
    if seasons is None:
        seasons = SEASONS

    raw_dir = Path(raw_dir)
    dfs: List[pd.DataFrame] = []

    for label, code in seasons.items():
        file_path = raw_dir / f"E0_{code}.csv"
        if not file_path.exists():
            print(f"Notice: Raw CSV {file_path} not found. Skipping {label}.")
            continue

        try:
            # Football-Data CSVs may have latin1 encoding
            raw_df = pd.read_csv(file_path, encoding="latin1")
            clean_df = standardize_dataframe(raw_df, source_season=label)
            print(f"Loaded {label} ({file_path.name}): {len(clean_df)} fixtures standardized.")
            dfs.append(clean_df)
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")

    if not dfs:
        raise RuntimeError(f"No match CSVs found or successfully parsed in {raw_dir}")

    # Combine all seasons
    combined = pd.concat(dfs, ignore_index=True)

    # Drop duplicate matches if any
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])

    # Sort strictly chronologically (anti-leakage requirement)
    combined = combined.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True)

    return combined


API_TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa",
    "Brighton & Hove Albion FC": "Brighton",
    "West Ham United FC": "West Ham",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Nottingham Forest FC": "Nott'm Forest",
    "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester",
    "Southampton FC": "Southampton",
    "Leeds United FC": "Leeds",
    "Burnley FC": "Burnley",
    "Sunderland AFC": "Sunderland",
}


def get_upcoming_fixtures(
    api_key: Optional[str] = None,
    csv_path: Optional[Path] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch the next scheduled Premier League gameweek fixtures.

    Attempts to fetch live scheduled fixtures using the football-data.org API
    if an API key is provided or present in the environment (FOOTBALL_DATA_ORG_KEY
    or FOOTBALL_DATA_API_KEY). Falls back to the schedule CSV in data/raw/upcoming_fixtures.csv
    or built-in gameweek schedule.

    Args:
        api_key: Optional football-data.org API token.
        csv_path: Path to fallback upcoming fixtures CSV. Defaults to 'data/raw/upcoming_fixtures.csv'.
        limit: Maximum number of fixtures to return.

    Returns:
        DataFrame with columns: ['Matchday', 'Date', 'HomeTeam', 'AwayTeam']
    """
    key = api_key or os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")

    if key:
        try:
            url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
            headers = {"X-Auth-Token": key}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                matches = payload.get("matches", [])
                if matches:
                    rows = []
                    # Filter for next upcoming matchday
                    target_matchday = matches[0].get("matchday")
                    for m in matches:
                        if target_matchday and m.get("matchday") != target_matchday and len(rows) >= limit:
                            break
                        h_raw = m.get("homeTeam", {}).get("name", "")
                        a_raw = m.get("awayTeam", {}).get("name", "")
                        h_team = API_TEAM_NAME_MAP.get(h_raw, h_raw)
                        a_team = API_TEAM_NAME_MAP.get(a_raw, a_raw)
                        utc_date = m.get("utcDate", "")
                        matchday = m.get("matchday", target_matchday or 1)
                        rows.append({
                            "Matchday": matchday,
                            "Date": utc_date,
                            "HomeTeam": h_team,
                            "AwayTeam": a_team,
                        })
                        if len(rows) >= limit:
                            break
                    if rows:
                        df_sched = pd.DataFrame(rows)
                        df_sched["Date"] = pd.to_datetime(df_sched["Date"]).dt.strftime("%Y-%m-%d %H:%M")
                        return df_sched
        except Exception as exc:
            print(f"Notice: football-data.org API fetch failed ({exc}). Using schedule fallback.")

    # Fallback to schedule CSV
    if csv_path is None:
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / "data" / "raw" / "upcoming_fixtures.csv"

    csv_path = Path(csv_path)
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)
        return df_csv.head(limit)

    # Built-in fallback schedule
    default_fixtures = [
        {"Matchday": 10, "Date": "2026-10-24 12:30", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"},
        {"Matchday": 10, "Date": "2026-10-24 15:00", "HomeTeam": "Bournemouth", "AwayTeam": "Brentford"},
        {"Matchday": 10, "Date": "2026-10-24 15:00", "HomeTeam": "Brighton", "AwayTeam": "Wolves"},
        {"Matchday": 10, "Date": "2026-10-24 15:00", "HomeTeam": "Crystal Palace", "AwayTeam": "Fulham"},
        {"Matchday": 10, "Date": "2026-10-24 15:00", "HomeTeam": "Everton", "AwayTeam": "West Ham"},
        {"Matchday": 10, "Date": "2026-10-24 17:30", "HomeTeam": "Man City", "AwayTeam": "Liverpool"},
        {"Matchday": 10, "Date": "2026-10-25 14:00", "HomeTeam": "Tottenham", "AwayTeam": "Aston Villa"},
        {"Matchday": 10, "Date": "2026-10-25 14:00", "HomeTeam": "Nott'm Forest", "AwayTeam": "Ipswich"},
        {"Matchday": 10, "Date": "2026-10-25 16:30", "HomeTeam": "Man United", "AwayTeam": "Newcastle"},
        {"Matchday": 10, "Date": "2026-10-26 20:00", "HomeTeam": "Leicester", "AwayTeam": "Southampton"},
    ]
    return pd.DataFrame(default_fixtures).head(limit)


def get_data(
    raw_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Convenience entrypoint to ensure data is downloaded, standardized, and returned.

    Args:
        raw_dir: Path to raw directory.
        force_download: If True, redownload CSVs even if present.

    Returns:
        Standardized, chronologically sorted DataFrame.
    """
    if raw_dir is None:
        raw_dir = Path("data") / "raw"
    raw_dir = Path(raw_dir)

    # Check if download is needed
    existing = list(raw_dir.glob("E0_*.csv"))
    if force_download or not existing:
        print("Raw files missing or force_download requested. Starting download...")
        download_all_seasons(raw_dir=raw_dir)

    df = load_and_standardize_all(raw_dir=raw_dir)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Premier League Data Ingestion & Standardization (Phase 1)")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    raw_directory = project_root / "data" / "raw"

    print(f"Raw data target directory: {raw_directory}")
    print(f"Downloading seasons: {list(SEASONS.keys())}")
    download_all_seasons(raw_dir=raw_directory)

    print("\nVerifying raw files in data/raw/:")
    for f in sorted(raw_directory.glob("*.csv")):
        print(f" - {f.name} ({f.stat().st_size:,} bytes)")

    print("\nStandardizing and compiling dataset...")
    data = load_and_standardize_all(raw_dir=raw_directory)

    print("\n" + "=" * 60)
    print("DATA SUMMARY & VERIFICATION")
    print("=" * 60)
    print(f"Total Matches Loaded : {len(data):,}")
    print(f"Dataset Shape        : {data.shape}")
    print(f"Columns              : {list(data.columns)}")
    print(f"Date Range           : {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    print("\nTarget (FTR) Distribution:")
    counts = data["FTR"].value_counts().sort_index()
    labels = {0: "0 (Away Win)", 1: "1 (Draw)", 2: "2 (Home Win)"}
    for val, count in counts.items():
        pct = (count / len(data)) * 100
        print(f"  {labels.get(val, val)}: {count:4d} ({pct:5.1f}%)")

    print("\nFirst 5 Rows:")
    print(data.head())
    print("\nLast 5 Rows:")
    print(data.tail())
    print("\nData Types:")
    print(data.dtypes)
    print("=" * 60)
    print("Phase 1 Data Ingestion completed successfully.")
