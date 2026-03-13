# Evaluation Protocols for Temporal Link Prediction on Dynamic Transaction Graphs

## 1. Background and Tasks

Temporal link prediction (dynamic link prediction) asks, given a temporal graph observed up to time \(t\), to predict whether an edge \((s,d)\) will appear (or with what score) at a future time \(t' > t\). This is usually framed either as:

- **Binary classification** over sampled positive/negative edges, evaluated with AUROC / AP.
- **Ranking** of candidate destinations for a given source (and time), evaluated with MRR / Hits@K / Precision@K.

Recent benchmark work (DGB, TGB, ROLAND) and analyses of TGN/DyRep have converged toward **ranking-style evaluation with multiple negatives per positive** and away from binary AUROC with one random negative per positive, which is now seen as too optimistic for sparse temporal graphs.[^1][^2][^3]


## 2. Negative Sampling: What Is “Standard” Now?

### 2.1 Legacy “standard”: single random negative per positive

Many early temporal GNN papers, including TGN and several pre-TGB benchmarks, evaluated dynamic link prediction as a binary classification task with **one negative edge per positive**, sampled uniformly from all node pairs at the same time. This strategy tends to generate very easy negatives in sparse graphs and leads to near-perfect AUROC/AP for many models, obscuring real differences.[^2][^3][^1]

DyRep’s original evaluation effectively ranks the true destination against *all* other candidate destinations, but training still uses negative sampling in the likelihood, and comparisons to baselines in that era often used relatively small candidate sets and AUROC/AP.[^4]


### 2.2 TGB (NeurIPS 2023): mix of historical and random negatives

For **dynamic link property prediction** (their link prediction task), TGB explicitly replaces binary AUROC with a **ranking protocol** and a more realistic negative sampler:[^1]

- For each positive edge \(e_p = (s, d, t)\), they **fix the source \(s\) and time \(t\)** and sample \(q\) negative destinations.
- The candidate set is \(\{d\} \cup \{d_1, \dots, d_q\}\), and the model is evaluated by the rank of \(d\) within this set.
- Negatives are sampled as a **50/50 mix of**:
  - **Historical negatives**: edges that occurred in training but are *not present at current time* \(t\) (i.e., “edges that used to exist but do not exist now”).[^2][^1]
  - **Random negatives**: destination nodes drawn uniformly from the node set.[^1]
- If insufficient historical negatives exist at time \(t\), the remainder are filled by random negatives.[^1]

This is directly inspired by DGB’s historical negative sampling, which was proposed as a harder and more realistic alternative to pure random sampling.[^5][^2]

For small graphs (tgbl-wiki), TGB sets \(q\) such that **all possible destinations** for a source are used (full ranking); for larger graphs (e.g., tgbl-review: 100 negatives per positive), \(q\) is chosen as a trade-off between evaluation completeness and runtime.[^1]


### 2.3 DGB and follow‑ups: random vs historical vs inductive vs hard negatives

DGB formalizes three families of negative sampling strategies for evaluation:[^5][^2]

- **Random NS**: sample from all node pairs that are not edges at time \(t\).
- **Historical NS**: sample from edges that occurred at earlier times but are *absent* at time \(t\); tests whether the model can decide *when* a recurrent edge reappears.[^2][^5]
- **Inductive NS**: sample from edges that only appear in the *future* part of the sequence; tests generalization to edges never seen during training.[^5][^2]

They show that random-only NS massively overestimates performance and that historical/inductive negatives change model rankings.[^3][^2]

Later work further refines **hard negative sampling** for *training*:

- **ENS (Enhanced Negative Sampling)** gradually increases negative difficulty by exploiting historical dependence and temporal proximity preferences in dynamic graphs.[^6][^7]
- **Nearest-neighbour–based hard negatives** for temporal GNNs sample negatives with similar embeddings instead of uniform random, improving training but not necessarily changing the evaluation protocol itself.[^8][^9]
- **Recently Popular Negative Sampling (RP‑NS)** for TGN/DyRep emphasizes negatives among globally popular destinations, to combat saturated scores and better match real‑world recommendation scenarios.[^10]

Cornell et al. (2025) caution that *embedding-based* hard negatives and inconsistent sampled metrics can make evaluation brittle and difficult to interpret, and recommend focusing on well‑defined random/historical/inductive schemes instead of ad‑hoc “hard” negatives.[^11][^12]


### 2.4 ROLAND: large‑candidate random negatives

ROLAND evaluates future link prediction on snapshot‑based dynamic graphs (including Bitcoin OTC/Alpha trust networks) using a pure **random negative sampler** but with large candidate sets:[^13]

- For each positive edge \((u,v)\) at snapshot \(t+1\), they sample **10,001 negative destinations** for source \(u\) (100 for the largest dataset due to memory).[^13]
- They then compute the rank of \(v\) among these 10,002 candidates and average the reciprocal rank across all positives (MRR).

This is a more stringent random‑only evaluation than the old 1‑negative protocols but does **not** incorporate historical or inductive negatives.


### 2.5 Summary: what to do in practice

For a new dynamic transaction graph (Bitcoin‑like), current best practice, consistent with TGB and DGB, is:

- **Evaluation**:
  - Treat the task as a **ranking problem** with multiple negatives per positive.
  - For each \((s,d,t)\), fix \(s,t\) and sample \(q\) candidate negative destinations.
  - Use a **mixture of historical and random negatives**; optionally add an inductive split if probing generalization.
- **Training**:
  - Use random/historical negatives, or one of the principled hard‑negative schemes (ENS, RP‑NS) if needed; but keep the *evaluation* sampler simple and well‑documented.

Pure random one‑negative evaluation with AUROC/AP should be considered legacy unless you need comparability with older work.


## 3. Candidate Generation Strategies

### 3.1 Per‑source candidate sets

Nearly all modern TLP protocols (TGB, DGB, ROLAND, DyRep) define candidates **per source node and time**:

- For each positive event \((s, d, t)\), define a *query* \(q = (s,t)\).
- Build a candidate set \(C(q) = \{d\} \cup N^{-}(q)\), where \(N^{-}(q)\) is a set of negative destinations for that source and time.
- Score all \(d' \in C(q)\) and compute the rank of the true destination.[^4][^13][^1]

This is conceptually equivalent to recommendation‑style evaluation: for user \(s\) at time \(t\), rank a list of items (destinations) and check where the actually chosen item lands.


### 3.2 Full vs sampled candidate sets

- **Full ranking (all destinations)** is used when the graph is small enough. TGB’s tgbl-wiki dataset evaluates each event against *all possible destinations*.[^1]
- **Sampled ranking** is standard on large graphs:
  - TGB uses a dataset‑specific number of negatives \(q\) (e.g., 100 negatives per positive on tgbl-review), chosen to balance runtime and resolution.[^1]
  - ROLAND uses 10,001 negatives per positive (or 100 on BSI‑ZK) for future link prediction.[^13]
  - DGB uses tens to hundreds of negatives depending on the protocol and dataset.[^3][^2]

For transaction graphs with millions of addresses (e.g., TGB’s tgbl-coin stablecoin transaction data), full ranking is infeasible, so a sampled candidate set is the norm.[^1]


### 3.3 Popularity‑aware candidate sets for transaction graphs

On transaction/social graphs with strong global temporal dynamics, pure random negatives under‑represent *popular* destinations. Daniluk & Dąbrowski (NeurIPS 2023 workshop) show that a trivial “recently popular nodes” baseline (PopTrack) that ignores the source node and only ranks by recent global destination frequency achieves **MRR ≈ 0.725 on tgbl-coin and 0.729 on tgbl-comment**, beating TGN, DyRep and EdgeBank under TGB’s default sampler.[^10]

To stress‑test models under such conditions, they propose evaluating also on **top‑N popular‑destination candidate sets**:

- For each query, restrict candidates to the top 20/100/500 most recently popular destinations and compute MRR within that set (MRR\(_{\text{top}N}\)).[^10]
- Under this harder setting, PopTrack collapses, while non‑contrastive models or improved negative sampling (RP‑NS) help TGN/DyRep perform better on popular destinations.[^10]

For Bitcoin‑like graphs, where global popularity spikes (exchanges, mixers, large services) are common, including a **popularity‑focused candidate variant** in addition to TGB‑style sampling is advisable.


## 4. Precision@K and Hits@K: Global vs Per‑Source

### 4.1 How metrics are actually computed in key papers

- **DyRep**: for dynamic link prediction they replace the true destination with all other entities, rank by model score, and report Mean Average Rank (MAR) and Hits@10 across *test events*. Metrics are computed **per query (event)** and then averaged, not as a single global ranking.[^4]

- **TGB**: for link prediction they use *filtered* Mean Reciprocal Rank (MRR) as the primary metric, computing the reciprocal rank for each positive edge against its sampled negatives, then averaging across all test edges. This is equivalent to per‑query evaluation; although they do not report Precision@K, the natural definition would follow the same per‑query pattern.[^1]

- **ROLAND**: defines future link prediction with per‑snapshot queries and computes MRR by averaging per‑query reciprocal ranks over all snapshots.[^13]

- **DGB**: for AUROC/AP, they still treat the task as a pooled binary classification problem, but for ranking‑style metrics (e.g., their EXHAUSTIVE evaluation) they operate per query (per source–time pair) and aggregate.[^14][^2]


### 4.2 Critique of “combined nodes predictions” / global metrics

Cornell et al. (2025) explicitly highlight “combined nodes predictions” as a key problem in current TLP evaluation:[^12][^11]

- Many works compute metrics over *all* scored edges together (e.g., pooling predictions from all sources), which implicitly assumes equal base rates across sources and lets high‑degree nodes dominate.
- They argue this mirrors long‑standing concerns in recommender systems, where global metrics can be skewed toward heavy users or popular items.
- Their survey table shows that several benchmarks (e.g., DGB) combine nodes, while TGB avoids this issue for its link prediction task.[^12]

The recommendation from this line of work is to compute ranking metrics **per source (or per query) and then average**, rather than globally pooling predictions.


### 4.3 Recommended protocol for Precision@K / Hits@K

Given the above, for temporal link prediction on transaction graphs:

- Define evaluation at the level of **queries** \(q = (s,t)\) or individual positive edges.
- For each query, rank its candidate set \(C(q)\) and compute:
  - Hits@K: indicator that the true destination is in the top K.
  - Precision@K or Recall@K: ratio of true positives in the top K when multiple positives per query exist.
- Average the metric **over queries** (macro averaging), optionally weighting queries equally or by application‑specific importance.

Computing precision@K “globally” over all scored edges (pooling sources) is not common in recent TGL benchmarks and is specifically discouraged by the evaluation‑oriented literature.[^11][^12]


## 5. Baseline Results on Temporal Graph Benchmarks

### 5.1 TGB transaction‑like datasets (tgbl-coin, tgbl-comment)

On TGB’s **dynamic link property prediction** task, using their historical+random negative sampler and MRR metric, the reported results on transaction‑like datasets are:[^15][^16][^1]

| Dataset | Model / Baseline | Test MRR |
|--------|------------------|----------|
| tgbl-coin (stablecoin tx) | DyRep | 0.452 ± 0.046[^1] |
| | TGN | 0.586 ± 0.037[^1] |
| | EdgeBank\(_{tw}\) (temporal memory heuristic) | 0.580[^1] |
| | EdgeBank\(_{∞}\) | 0.359[^1] |
| | Preferential Attachment (PA) | 0.481[^15] |
| | PA restricted to recent edges (PA\(_{rec}\)) | 0.584[^15] |
| | PopTrack (recently popular nodes, no learning) | 0.725[^10] |

| Dataset | Model / Baseline | Test MRR |
|--------|------------------|----------|
| tgbl-comment (Reddit reply network) | DyRep | 0.289 ± 0.033[^1] |
| | TGN | 0.379 ± 0.021[^1] |
| | EdgeBank\(_{tw}\) | 0.149[^1] |
| | EdgeBank\(_{∞}\) | 0.129[^1] |
| | Common Neighbours (CN) | 0.131[^15] |
| | CN\(_{rec}\), AA\(_{rec}\), RA\(_{rec}\) (recent neighbours) | ≈ 0.242–0.245[^15] |
| | PopTrack | 0.729[^10] |

A few observations relevant for Bitcoin‑like graphs:

- Simple *non‑learned* heuristics (PA, CN/AA/RA on recent edges) already reach **MRR ≈ 0.48–0.58** on tgbl-coin, comparable to TGN/DyRep.[^15][^1]
- Purely global popularity (PopTrack) can reach **MRR ≈ 0.72–0.73** on both tgbl-coin and tgbl-comment under TGB’s default sampler, highlighting how strong global temporal dynamics are in these datasets.[^10]

These numbers provide realistic targets for any logistic‑regression or CatBoost model built on simple temporal/structural features: beating PA/PA\(_{rec}\), EdgeBank, and ideally PopTrack under the same sampler should be the bar.


### 5.2 ROLAND on Bitcoin OTC / Alpha trust graphs

ROLAND reports future link prediction performance (MRR, random negatives, large candidate sets) on Bitcoin OTC and Bitcoin Alpha trust networks.[^13]

Under the fixed‑split setting (train on first 90% snapshots, test on last 10%), the reported test MRRs are approximately:

- **Bitcoin‑OTC**:
  - Best baseline (EvolveGCN variants): MRR ≈ 0.152.[^13]
  - ROLAND‑GRU: MRR ≈ 0.220, a **~74% relative improvement**.[^13]
- **Bitcoin‑Alpha**:
  - Best baseline: MRR ≈ 0.201.[^13]
  - ROLAND‑GRU: MRR ≈ 0.289, a **~44% relative improvement**.[^13]

These are **trust graphs**, not raw UTXO/transaction graphs, but structurally similar to entity‑level transaction networks.


### 5.3 DyRep small‑scale benchmarks

On the original DyRep datasets (Social Evolution, Github), evaluated with per‑query ranking (MAR) and Hits@10:[^4]

- On a dense **Social Evolution** network, DyRep achieves **Hits@10 ≈ 0.79** and significantly better MAR than Know‑Evolve, DynGEM, GraphSAGE, and node2vec on both communication and association prediction.[^4]
- On a sparser **Github** network, DyRep’s Hits@10 is ≈ 0.32 for communication links, again outperforming baselines.[^4]

Absolute numbers are not directly comparable to TGB (different datasets and protocols), but they give a sense of the performance regime for small, well‑instrumented dynamic networks.


### 5.4 Logistic regression / CatBoost baselines

There is **no widely adopted “standard” MRR / Hits@K baseline for logistic regression or CatBoost** on temporal link prediction benchmarks:

- Major TGL benchmarks (DGB, TGB, ROLAND) focus on neural temporal GNNs plus **non‑parametric heuristics** (EdgeBank, PA, CN, PopTrack) as baselines, not tree/linear models on handcrafted features.[^16][^2][^1]
- Gradient‑boosted trees and logistic regression are popular on *node classification* tasks on Bitcoin transaction graphs (e.g., Elliptic AML detection, where GNNs are compared against XGBoost, etc.), but these operate on node‑label prediction, not link prediction.[^17][^18][^19]

Given this, the most defensible way to characterize “typical classical baselines” on temporal link prediction is:

- Use **heuristic scores** (PA, CN/AA/RA, EdgeBank, PopTrack) as feature inputs to logistic regression / CatBoost.
- Expect performance to be **similar to or slightly above** the raw heuristic scores shown above, but any specific MRR/Hits@K values need to be empirically measured on the target dataset.


## 6. Bitcoin‑Specific Dynamic Transaction Graphs

Several recent works provide Bitcoin‑scale temporal transaction graphs but often focus on node classification, anomaly detection, or price forecasting rather than standardized TLP:

- **ORBITAAL**: a temporal entity–entity Bitcoin transaction dataset (2009–2021) with both stream and snapshot representations and rich per‑entity attributes, intended as a general resource rather than a fixed benchmark; no canonical link‑prediction protocol is prescribed.[^20]
- **Temporal Graph of Bitcoin Transactions (Jalili 2025)**: constructs a massive temporal heterogeneous graph of Bitcoin transactions (2.4B nodes, 39.7B edges) with tools for sampling and analysis; again, it is a dataset, not a standardized TLP benchmark.[^21][^22]
- Financial crime / AML works (e.g., Balanced‑BiEGCN, RecGNN on Elliptic) focus on node anomaly classification on evolving Bitcoin graphs, using accuracy/F1 rather than link prediction metrics.[^18][^19][^17]

For Bitcoin‑style *link* prediction, the closest standardized setups today are thus:

- TGB’s **tgbl-coin** stablecoin transaction dataset and evaluation pipeline.[^1]
- ROLAND’s Bitcoin OTC/Alpha trust network experiments with random‑negative MRR.[^13]

These strongly suggest adopting TGB’s negative sampling and ranking protocol (Section 2.2) combined with popularity‑aware candidate subsets (Section 3.3) for Bitcoin transaction graphs.


## 7. Practical Recommendations for Your Setup

For temporal link prediction on a dynamic Bitcoin/transaction graph, aligned with TGB, ROLAND, DyRep, and TGN:

1. **Train/test split**: chronological, e.g. 70/15/15 as in TGB; avoid leakage by only using history up to time \(t\) when predicting at \(t' > t\).[^1][^13]
2. **Evaluation setting**: streaming (TGB/TGN style) or live‑update (ROLAND style), where the model can update its internal state with observed test events but cannot backpropagate on them.[^23][^13][^1]
3. **Negative sampling (eval)**:
   - For each positive \((s,d,t)\), fix \(s,t\) and sample \(q\) negatives with a 50/50 mix of historical and random edges when possible, falling back to random when necessary.[^2][^1]
   - Consider an additional **popular‑destinations evaluation** restricted to the top‑N globally popular addresses (MRR\(_{\text{top}N}\)).[^10]
4. **Candidate generation**:
   - Use per‑query candidate sets; full destination sets for small graphs, sampled for large Bitcoin‑scale graphs.
   - Document \(q\) and sampling details clearly.
5. **Metrics**:
   - Primary: **filtered MRR**, as in TGB.[^1]
   - Secondary: **Hits@K / Precision@K**, computed per query (per source–time pair) and averaged.[^11][^4]
   - Optional: AUROC/AP only if you need comparability to legacy TGN/DyRep baselines.
6. **Baselines**:
   - Non‑learned: PA, PA\(_{rec}\), CN/AA/RA and their “recent” variants, EdgeBank, PopTrack.[^15][^10][^1]
   - Neural: TGN, DyRep (and/or ROLAND‑style GNNs) with the same evaluation pipeline.[^23][^13][^1]
   - Classical ML: logistic regression / CatBoost on features such as degree, temporal counts, heuristic scores, and recency; compare their MRR/Hits@K directly to the heuristic baselines above.

This configuration makes your setup comparable to current TGL benchmarks while addressing the known pitfalls around weak negatives and global metrics in temporal link prediction.

---

## References

1. [[PDF] Temporal Graph Benchmark for Machine Learning on Temporal ...](https://cs.stanford.edu/people/jure/pubs/temporal-neurips23.pdf) - For dynamic link property prediction, we sample multiple negative instances per positive edge and en...

2. [[PDF] Towards Better Evaluation for Dynamic Link Prediction - arXiv](https://arxiv.org/pdf/2207.10128.pdf)

3. [Towards Better Evaluation for](https://papers.nips.cc/paper_files/paper/2022/file/d49042a5d49818711c401d34172f9900-Paper-Datasets_and_Benchmarks.pdf)

4. [Representation Learning over Dynamic Graphs](https://arxiv.org/pdf/1803.04051.pdf)

5. [[PDF] Towards Better Evaluation for Dynamic Link Prediction - NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/d49042a5d49818711c401d34172f9900-Paper-Datasets_and_Benchmarks.pdf)

6. [Towards a better negative sampling strategy for dynamic ...](https://openreview.net/forum?id=n1M4V3A3gT) - As dynamic graphs have become indispensable in numerous fields due to their capacity to represent ev...

7. [Towards a better negative sampling strategy for dynamic graphs - PubMed](https://pubmed.ncbi.nlm.nih.gov/38387201/) - As dynamic graphs have become indispensable in numerous fields due to their capacity to represent ev...

8. [Robust Training of Temporal GNNs using Nearest Neighbours based Hard Negatives](https://arxiv.org/html/2402.09239v1)

9. [Robust Training of Temporal GNNs using Nearest Neighbours ...](https://arxiv.org/html/2402.09239)

10. [Temporal graph models fail to capture global temporal](https://openreview.net/pdf?id=Ks94Yn5jqY)

11. [Are We Really Measuring Progress? Transferring Insights from ...](https://openreview.net/pdf?id=S6BfBrrD9L)

12. [Are We Really Measuring Progress? Transferring Insights ...](https://arxiv.org/html/2506.12588v1)

13. [ROLAND: Graph Learning Framework for Dynamic Graphs](https://arxiv.org/pdf/2208.07239.pdf)

14. [Exhaustive Evaluation of Dynamic Link Prediction](https://openreview.net/forum?id=C1YEZ7hmyR) - Dynamic link prediction is a crucial task in the study of evolving graphs, which serve as abstract m...

15. [Link prediction heuristics for temporal graph benchmark](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-141.pdf)

16. [Dynamic Link Prediction: AI-driven Methods and Evaluation | SERP AI](https://serp.ai/posts/dynamic-link-prediction/)

17. [Balanced-BiEGCN: A Bidirectional EvolveGCN with a Class ... - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC12564723/) - Bitcoin transaction anomaly detection is essential for maintaining financial market stability. A sig...

18. [Robust recurrent graph convolutional network approach ...](https://d-nb.info/1322840954/34)

19. [Multimedia Tools and Applications](https://eprints.bournemouth.ac.uk/39075/7/s11042-023-17323-4.pdf)

20. [ORBITAAL: A Temporal Graph Dataset of Bitcoin Entity- ...](https://arxiv.org/abs/2408.14147) - Research on Bitcoin (BTC) transactions is a matter of interest for both economic and network science...

21. [The Temporal Graph of Bitcoin Transactions - OpenReview](https://openreview.net/forum?id=Xs7JM4VGHv) - Since its 2009 genesis block, the Bitcoin network has processed >1.08 billion (B) transactions repre...

22. [[2510.20028] The Temporal Graph of Bitcoin Transactions](https://arxiv.org/abs/2510.20028) - Since its 2009 genesis block, the Bitcoin network has processed >1.08 billion (B) transactions repre...

23. [Temporal Graph Networks for Deep Learning on Dynamic ...](https://arxiv.org/abs/2006.10637) - Graph Neural Networks (GNNs) have recently become increasingly popular due to their ability to learn...

