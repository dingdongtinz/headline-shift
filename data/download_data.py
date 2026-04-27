#!/usr/bin/env python3
"""
Download and prepare datasets for the headline_shift project.

Datasets:
  1. QBias  – already cloned into data/qbias/ (AllSides headlines with bias labels)
  2. Kaggle 4.5M headlines – requires kaggle API key.
     Falls back to generating realistic synthetic sample data for demo.

Run:
    python data/download_data.py
"""

import os, sys, csv, random, datetime, subprocess, zipfile
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    DATA_DIR, PROCESSED_DIR, TARGET_PUBLICATIONS, YEAR_RANGE, set_seed
)

set_seed(42)

# ── 1. QBias ───────────────────────────────────────────────────────────────
QBIAS_RAW = os.path.join(DATA_DIR, "qbias", "allsides_balanced_news_headlines-texts.csv")
QBIAS_OUT = os.path.join(PROCESSED_DIR, "qbias_clean.csv")


def process_qbias():
    """Clean QBias data: keep heading (headline), source, bias_rating."""
    print("[1/3] Processing QBias data …")
    if not os.path.exists(QBIAS_RAW):
        print("  ⚠ QBias CSV not found. Run: git clone https://github.com/irgroup/Qbias data/qbias")
        return False

    df = pd.read_csv(QBIAS_RAW)
    # keep only the columns we need
    df = df[["heading", "source", "bias_rating"]].rename(
        columns={"heading": "headline", "bias_rating": "label"}
    )
    df = df.dropna(subset=["headline", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["left", "center", "right"])]
    df.to_csv(QBIAS_OUT, index=False)
    print(f"  ✓ Saved {len(df)} records → {QBIAS_OUT}")
    return True


# ── 2. Kaggle headlines ───────────────────────────────────────────────────
KAGGLE_DATASET = "jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites"
KAGGLE_ZIP = os.path.join(DATA_DIR, "45m-headlines-from-2007-2022-10-largest-sites.zip")
HEADLINES_OUT = os.path.join(PROCESSED_DIR, "headlines_filtered.csv")


def try_kaggle_download() -> bool:
    """Attempt to download via kaggle CLI. Returns True on success."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", DATA_DIR],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and os.path.exists(KAGGLE_ZIP):
            print("  ✓ Kaggle download succeeded.")
            return True
        print(f"  ⚠ Kaggle CLI failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ⚠ Kaggle CLI unavailable: {e}")
    return False


def extract_and_filter_kaggle():
    """Unzip kaggle data, filter to target pubs and year range."""
    print("  Extracting & filtering Kaggle data …")
    with zipfile.ZipFile(KAGGLE_ZIP) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            print("  ⚠ No CSV found in zip")
            return False
        # read in chunks because it's huge
        frames = []
        for name in csv_names:
            with zf.open(name) as f:
                for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
                    # try common column names
                    pub_col = None
                    for c in ["publication", "source", "publisher"]:
                        if c in chunk.columns:
                            pub_col = c
                            break
                    date_col = None
                    for c in ["date", "publish_date", "published"]:
                        if c in chunk.columns:
                            date_col = c
                            break
                    headline_col = None
                    for c in ["title", "headline", "heading"]:
                        if c in chunk.columns:
                            headline_col = c
                            break
                    if not all([pub_col, date_col, headline_col]):
                        continue
                    sub = chunk[[pub_col, date_col, headline_col]].rename(
                        columns={pub_col: "publication", date_col: "date", headline_col: "headline"}
                    )
                    # filter publications (fuzzy match)
                    pub_map = {
                        "cnn": "CNN", "fox": "Fox News", "fox news": "Fox News",
                        "washington post": "Washington Post", "new york times": "New York Times",
                        "nyt": "New York Times", "nytimes": "New York Times",
                    }
                    sub["publication"] = sub["publication"].str.lower().str.strip().map(
                        lambda x: next((v for k, v in pub_map.items() if k in str(x)), None)
                    )
                    sub = sub.dropna(subset=["publication"])
                    # filter years
                    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                    sub = sub.dropna(subset=["date"])
                    sub = sub[(sub["date"].dt.year >= YEAR_RANGE[0]) & (sub["date"].dt.year <= YEAR_RANGE[1])]
                    if len(sub) > 0:
                        frames.append(sub)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(HEADLINES_OUT, index=False)
            print(f"  ✓ Filtered {len(df)} headlines → {HEADLINES_OUT}")
            return True
    return False


# ── 3. Synthetic fallback ─────────────────────────────────────────────────
SAMPLE_HEADLINES = {
    "CNN": [
        "Breaking: Climate Change Report Warns of Dire Consequences",
        "Trump Administration Faces Backlash Over Immigration Policy",
        "Healthcare Reform Battle Heats Up in Congress",
        "Gun Violence Epidemic Sparks Renewed Calls for Action",
        "Economy Shows Signs of Recovery Amid Pandemic",
        "Racial Justice Protests Sweep the Nation",
        "Supreme Court Rules on Landmark Voting Rights Case",
        "Infrastructure Bill Passes After Months of Negotiations",
        "Tech Giants Face Antitrust Scrutiny",
        "Biden Administration Announces New Climate Targets",
        "Deadly Shooting Highlights Gun Control Debate",
        "January 6 Committee Releases Damning Findings",
        "COVID-19 Vaccine Rollout Faces Distribution Challenges",
        "Democrats Push for Expanded Social Safety Net",
        "Experts Warn of Growing Inequality in America",
    ],
    "Fox News": [
        "Biden's Border Crisis Reaches New Heights",
        "Small Business Owners Crushed by New Regulations",
        "Americans Fed Up With Rising Gas Prices Under Biden",
        "Parents Fight Back Against Critical Race Theory in Schools",
        "Second Amendment Rights Under Attack by Democrats",
        "Inflation Surge Threatens American Families",
        "Big Tech Censorship of Conservative Voices Continues",
        "Law Enforcement Officers Demoralized by Defund Movement",
        "Energy Independence Sacrificed for Green New Deal",
        "Illegal Immigration Costs Taxpayers Billions",
        "Cancel Culture Claims Another Victim",
        "Democrats' Spending Spree Will Bankrupt America",
        "Hunter Biden Laptop Scandal Media Coverup Exposed",
        "Religious Liberty Under Siege From Radical Left",
        "Trump Rally Draws Massive Crowds Across Nation",
    ],
    "Washington Post": [
        "Analysis: Democracy Faces Its Greatest Test in Decades",
        "The Quiet Crisis: How Climate Change Is Reshaping American Cities",
        "Investigation: Corporate Lobbying Undermines Environmental Rules",
        "Fact Checker: Political Claims on Immigration Miss the Mark",
        "How Gerrymandering Threatens Fair Representation",
        "Inside the Battle for the Future of the Republican Party",
        "Education Gaps Widen as Pandemic Disrupts Learning",
        "The Filibuster Debate: What's Really at Stake",
        "Tech Regulation Stalls as Industry Influence Grows",
        "How Disinformation Campaigns Are Eroding Public Trust",
        "Pentagon Papers for a New Era: Government Secrecy Examined",
        "Wealth Gap Between Richest and Poorest Americans Hits Record",
        "Police Reform Measures Face Resistance in State Legislatures",
        "Global Democracy in Retreat as Authoritarianism Rises",
        "Voting Rights Act Protections Weakened by Court Rulings",
    ],
    "New York Times": [
        "A Reckoning on Race Forces America to Confront Its Past",
        "The Pandemic Changed How Americans Work. There's No Going Back.",
        "Climate Scientists Issue Urgent Warning on Tipping Points",
        "Inside the Growing Movement to Restrict Abortion Access",
        "How Social Media Algorithms Fuel Political Polarization",
        "The Struggle for Affordable Housing in America's Cities",
        "Investigating the Roots of the January 6 Capitol Breach",
        "Mental Health Crisis Among Young Americans Deepens",
        "The Widening Political Divide: Can America Find Common Ground?",
        "Global Supply Chain Disruptions Reshape the Economy",
        "Rural America Faces a Growing Healthcare Desert",
        "The Rise of Political Violence Threatens Democratic Norms",
        "How Misinformation Became a Public Health Crisis",
        "Tech Giants Wield Unprecedented Power Over Public Discourse",
        "Immigration Reform Remains Elusive Despite Bipartisan Support",
    ],
}


def generate_synthetic_data(n_per_pub: int = 5000):
    """Create synthetic headline data that mimics the real dataset structure."""
    print("[2/3] Generating synthetic headline data for demo …")

    topics = [
        "politics", "economy", "healthcare", "immigration", "climate",
        "education", "technology", "crime", "foreign_policy", "social_issues",
    ]
    rows = []
    for pub in TARGET_PUBLICATIONS:
        templates = SAMPLE_HEADLINES[pub]
        for i in range(n_per_pub):
            # random date in range
            year = random.randint(YEAR_RANGE[0], YEAR_RANGE[1])
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.date(year, month, day)
            headline = random.choice(templates)
            # add slight variation
            variations = [
                "", " - Report", " | Opinion", " (Updated)", "",
                " as Tensions Rise", " Amid Growing Concerns", "",
                " in Latest Development", " Sources Say",
            ]
            headline = headline + random.choice(variations)
            rows.append({
                "publication": pub,
                "date": date.isoformat(),
                "headline": headline,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(HEADLINES_OUT, index=False)
    print(f"  ✓ Generated {len(df)} synthetic headlines → {HEADLINES_OUT}")
    return True


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Headline Shift – Data Download & Preparation")
    print("=" * 60)

    # Step 1: QBias
    process_qbias()

    # Step 2: Kaggle (or synthetic fallback)
    if os.path.exists(HEADLINES_OUT):
        print(f"[2/3] Headlines already processed at {HEADLINES_OUT}, skipping.")
    else:
        print("[2/3] Attempting Kaggle download …")
        if try_kaggle_download() and os.path.exists(KAGGLE_ZIP):
            if not extract_and_filter_kaggle():
                generate_synthetic_data()
        else:
            generate_synthetic_data()

    # Step 3: summary
    print("\n[3/3] Data summary:")
    if os.path.exists(QBIAS_OUT):
        df = pd.read_csv(QBIAS_OUT)
        print(f"  QBias:     {len(df)} rows  | Labels: {df['label'].value_counts().to_dict()}")
    if os.path.exists(HEADLINES_OUT):
        df = pd.read_csv(HEADLINES_OUT)
        print(f"  Headlines: {len(df)} rows  | Pubs: {df['publication'].value_counts().to_dict()}")
    print("\n✓ Data preparation complete.")


if __name__ == "__main__":
    main()
