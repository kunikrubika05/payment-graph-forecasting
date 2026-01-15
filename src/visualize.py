import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from collections import Counter

plt.rcParams["figure.figsize"] = 11, 7
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 16

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/samples"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

stream08 = pd.read_csv(DATA_DIR / "orbitaal-stream_graph-2016_07_08.csv")
stream09 = pd.read_csv(DATA_DIR / "orbitaal-stream_graph-2016_07_09.csv")
snap08 = pd.read_csv(DATA_DIR / "orbitaal-snapshot-2016_07_08.csv")
snap09 = pd.read_csv(DATA_DIR / "orbitaal-snapshot-2016_07_09.csv")
stream = pd.concat([stream08, stream09])
snapshot = pd.concat([snap08, snap09])

fig, ax = plt.subplots()
out_deg = snapshot.groupby('SRC_ID').size()
counts = Counter(out_deg)
x, y = zip(*sorted(counts.items()))
ax.loglog(x, y, 'o', alpha=0.6, color='#2E86AB')
ax.set_xlabel('Out-degree (k)')
ax.set_ylabel('P(k)')
ax.set_title('Degree Distribution')
ax.grid()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_degree_distribution.png", dpi=150)
plt.close()
print("Saved: 01_degree_distribution.png")

fig, ax = plt.subplots()
stream['hour'] = pd.to_datetime(stream['TIMESTAMP'], unit='s').dt.hour
hourly = stream.groupby('hour').size()
ax.bar(hourly.index, hourly.values, color='#A23B72', edgecolor='white')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Transactions')
ax.set_title('Daily Activity Pattern')
ax.set_xticks(range(0, 24, 3))
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_hourly_activity.png", dpi=150)
plt.close()
print("Saved: 02_hourly_activity.png")

fig, ax = plt.subplots()
btc = snapshot['VALUE_SATOSHI'] / 1e8
ax.hist(np.log10(btc + 1e-10), bins=80, color='#F18F01', edgecolor='white', alpha=0.8)
ax.axvline(np.log10(btc.median()), color='red', linestyle='--', label=f'Median: {btc.median():.4f} BTC')
ax.set_xlabel('log10(BTC)')
ax.set_ylabel('Edge Count')
ax.set_title('Transaction Value Distribution')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_value_distribution.png", dpi=150)
plt.close()
print("Saved: 03_value_distribution.png")

fig, ax = plt.subplots()
stream['datetime'] = pd.to_datetime(stream['TIMESTAMP'], unit='s')
stream['time_bin'] = stream['datetime'].dt.floor('30min')
timeline = stream.groupby('time_bin').size()
ax.plot(timeline.index, timeline.values, color='#2E86AB')
ax.fill_between(timeline.index, timeline.values, alpha=0.3, color='#2E86AB')
ax.set_xlabel('Time')
ax.set_ylabel('Transactions / 30 min')
ax.set_title('Transaction Rate')
ax.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_temporal.png", dpi=150)
plt.close()
print("Saved: 04_temporal.png")

fig, ax = plt.subplots()
nodes_08 = set(stream08['SRC_ID']).union(set(stream08['DST_ID']))
nodes_09 = set(stream09['SRC_ID']).union(set(stream09['DST_ID']))
common = len(nodes_08 & nodes_09)
only_08 = len(nodes_08 - nodes_09)
only_09 = len(nodes_09 - nodes_08)
sizes = [only_08, common, only_09]
labels = ['Only July 8', 'Both days', 'Only July 9']
colors = ['#2E86AB', '#A23B72', '#F18F01']
ax.bar(labels, sizes, color=colors, edgecolor='white')
ax.set_ylabel('Number of Entities')
ax.set_title('Entity Overlap Between Days')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_entity_overlap.png", dpi=150)
plt.close()
print("Saved: 05_entity_overlap.png")

fig, ax = plt.subplots()
snap_clean = snapshot[(snapshot['DST_ID'] != 0) & (snapshot['SRC_ID'] != snapshot['DST_ID'])]
top_nodes = pd.concat([snap_clean['SRC_ID'], snap_clean['DST_ID']]).value_counts().head(30).index
edges = snap_clean[snap_clean['SRC_ID'].isin(top_nodes) & snap_clean['DST_ID'].isin(top_nodes)]
G = nx.DiGraph()
for _, row in edges.iterrows():
    G.add_edge(row['SRC_ID'], row['DST_ID'], weight=row['VALUE_SATOSHI']/1e8)
if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    degrees = dict(G.degree())
    node_sizes = [100 + degrees.get(n, 0) * 30 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#2E86AB', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True, arrowsize=8, ax=ax)
ax.set_title(f'Network Subgraph ({len(G.nodes())} nodes)')
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_network_subgraph.png", dpi=150)
plt.close()
print("Saved: 06_network_subgraph.png")

print("\nDone!")
