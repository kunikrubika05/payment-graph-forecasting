import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data/samples"

def analyze_file(filepath):
    df = pd.read_csv(filepath)
    print(f"\n{'='*60}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(f"\nBasic stats:")
    print(df.describe())

    print(f"\nUnique entities:")
    print(f"  SRC_ID: {df['SRC_ID'].nunique():,}")
    print(f"  DST_ID: {df['DST_ID'].nunique():,}")
    all_nodes = pd.concat([df['SRC_ID'], df['DST_ID']]).unique()
    print(f"  Total unique: {len(all_nodes):,}")

    print(f"\nValue stats (BTC):")
    btc = df['VALUE_SATOSHI'] / 1e8
    print(f"  Sum: {btc.sum():,.2f}")
    print(f"  Mean: {btc.mean():.6f}")
    print(f"  Median: {btc.median():.6f}")
    print(f"  Max: {btc.max():,.2f}")

    dst_zero = len(df[df['DST_ID'] == 0])
    self_loops = len(df[df['SRC_ID'] == df['DST_ID']])
    print(f"\nSpecial edges:")
    print(f"  To entity 0: {dst_zero:,}")
    print(f"  Self-loops: {self_loops:,}")

    if 'TIMESTAMP' in df.columns:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
        print(f"\nTime range:")
        print(f"  Start: {df['datetime'].min()}")
        print(f"  End: {df['datetime'].max()}")

    out_deg = df.groupby('SRC_ID').size()
    in_deg = df.groupby('DST_ID').size()
    print(f"\nDegree distribution:")
    print(f"  Out-degree: mean={out_deg.mean():.2f}, max={out_deg.max()}")
    print(f"  In-degree: mean={in_deg.mean():.2f}, max={in_deg.max()}")

    return df

files = sorted(DATA_DIR.glob("*.csv"))
dfs = {}
for f in files:
    dfs[f.name] = analyze_file(f)

print(f"\n{'='*60}")
print("COMBINED ANALYSIS")
print(f"{'='*60}")

stream_dfs = [dfs[k] for k in dfs if 'stream' in k]
snapshot_dfs = [dfs[k] for k in dfs if 'snapshot' in k]

if stream_dfs:
    stream = pd.concat(stream_dfs)
    print(f"\nStream graph (combined):")
    print(f"  Transactions: {len(stream):,}")
    print(f"  Unique nodes: {len(pd.concat([stream['SRC_ID'], stream['DST_ID']]).unique()):,}")
    print(f"  Total BTC: {stream['VALUE_SATOSHI'].sum() / 1e8:,.2f}")

if snapshot_dfs:
    snapshot = pd.concat(snapshot_dfs)
    print(f"\nSnapshots (combined):")
    print(f"  Edges: {len(snapshot):,}")
    print(f"  Unique nodes: {len(pd.concat([snapshot['SRC_ID'], snapshot['DST_ID']]).unique()):,}")
    print(f"  Total BTC: {snapshot['VALUE_SATOSHI'].sum() / 1e8:,.2f}")

n = len(pd.concat([snapshot['SRC_ID'], snapshot['DST_ID']]).unique())
e = len(snapshot)
density = e / (n * (n - 1))
print(f"\nGraph density: {density:.2e}")
