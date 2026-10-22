# HW6 — Personalized PageRank movie recommendations

LLM4Rec: AI-Assisted Recommender Systems, HSE University. Individual assignment by Anastasia Getman.

Live page: https://getmanyan.github.io/llm4rec-hw6/
Assignment text: [`ASSIGNMENT.md`](ASSIGNMENT.md) (the instructor's `week6/readme.md`, unchanged; renamed because `README.md` and `readme.md` cannot both exist on macOS).
Starter: https://github.com/dryjins/RecSys-LLMs/tree/main/week6 (commit `00c032f`, 13 Oct 2026).

## What is implemented

| File | Function | What it does |
| --- | --- | --- |
| `graph.js` | `buildGraph(train)` | Bipartite user–movie graph from training ratings ≥ 4.0. Nodes = users and movies observed in train (943 + 1,611 = 2,554), users first, no padding node. |
| `graph.js` | `rowNormalize(graph)` | Dense row-normalized `Q` and `danglingMask`, for inspection and the small self-check fixtures only. |
| `graph.js` | `pageRank(alpha, opts)` | Power iteration with uniform restart in TensorFlow.js. |
| `graph.js` | `personalizedPageRank(node, alpha, opts)` | Same, one-hot restart at a user node. |
| `graph.js` | `personalizedPageRankBatch(nodes, alpha, opts)` | (added) PPR for many users at once: one `tf.matMul` per update for a `[2554, B]` matrix; each column stops on its own, so it equals the single-user result. |
| `script.js` | `movieScores`, `recommendTopK`, `recallAtK` | Movie-node scores, Top-K with the split's exclusions (ties by ascending id), macro Recall@10 on the shared cohort. |

Update rule (`alpha` = link-following probability, `1 - alpha` = restart):

```
p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v,   d = mass on zero-degree nodes
```

The square matrix is never built for MovieLens: `(Q^T p)_movies = R^T (p_users / deg_users)` and `(Q^T p)_users = R (p_movies / deg_movies)`, with `1/deg = 0` on zero-degree nodes (their mass goes back through `d * v`). Stop when the L1 change < 1e-6 or after 500 updates.

**MF.** The `mf` method uses my HW5 model, as the assignment asks ("your previous MF"): `mf/hw5_mf_checkpoint.json` is the selected checkpoint of https://github.com/getmanyan/llm4rec-hw5 (SGD, 20 epochs, lr 0.01, λ 0.05, batch 64, seeds 42/43/2026; checkpoint epoch 16). It was produced by `analysis/export_hw5_checkpoint.js`. The starter's own MF reference (`hw6TrainMf`, Adam + L2 on the whole factor matrices) is kept in `script.js` only for the comparison in `analysis/reference_mf.js`: on this split it collapses to the training mean (validation RMSE 1.1830 at every epoch, Recall@10 0.0211).

## Changes to the provided scaffolding (all marked `FIX (HW6)` in `script.js`)

1. `hw6AdapterPPR` passed the raw user id to `personalizedPageRank`, whose contract (and the readme 4.1 harness) uses a node index. On MovieLens that restarts the walk at the next user. It now passes the node index.
2. `hw6HandleRecall` ran the test comparison before the validation alpha sweep, so the final PPR row used the slider value. It now selects alpha on validation first and uses it for the test comparison.
3. `hw6RunTwoUserDemo` always used the two smallest cohort ids. It now uses the two selected users (default: users 10 and 102 from HW5).
4. The dataset summary computed sparsity over `(numUsers + 1) × (maxMovieId + 1)` cells (it counted the unused row/column 0).
5. MF: `hw6EnsureMf` loads the HW5 checkpoint instead of training the reference.
6. Two self-checks added (batched = single PPR; MF checkpoint RMSE). "Run tests" builds the graph and loads MF first, so it can be the first click.

`data.js`, `index.html`, `style.css`, `vendor/d3.min.js`, `u.data`, `u.item` are unchanged.

## Run the page

```
python3 -m http.server 8000      # from this folder; file:// does not work (fetch)
```

Open http://localhost:8000/ . TensorFlow.js 4.22.0 is loaded from jsDelivr, d3 v7 from `vendor/`.
Suggested order: **Run tests** → **Evaluate Recall@10 — all four methods** (validation sweep, then test with the selected alpha) → **Recommend Top-10** (method / user) → **Run two-user demo** → the three **Export** buttons.

## Reproduce every number in the report

```
npm install                          # @tensorflow/tfjs 4.22.0, @tensorflow/tfjs-backend-wasm 4.22.0
node analysis/evaluate.js            # wasm backend; prints the report numbers
node analysis/evaluate.js --backend cpu
node analysis/reference_mf.js > analysis/reference_mf.txt   # the starter's MF reference, ~3 min
```

Output used in the report: `analysis/results.txt` and `analysis/reference_mf.txt`. The script runs the page's own `data.js`, `graph.js` and `script.js` in Node (`analysis/load.js`) and writes the three exports with the page's exporters: `verification.json`, `results.csv`, `recommendations.csv`.

Versions actually used: Node.js 22.22.0, @tensorflow/tfjs 4.22.0, @tensorflow/tfjs-backend-wasm 4.22.0, d3 v7 (vendored by the starter).

## Data

MovieLens 100K (GroupLens, University of Minnesota): F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," ACM TiiS 5(4), 2015, https://doi.org/10.1145/2827872. `u.data` and `u.item` are the files from `week5/`, unchanged.
