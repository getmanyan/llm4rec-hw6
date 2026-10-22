// analysis/reference_mf.js — train the starter's MF reference (hw6TrainMf in
// script.js: Adam, lr 0.01, L2 0.05 on the WHOLE factor matrices added to the
// batch MSE, 30 epochs, batch 512) and report what it learns.
//
// The loss, optimizer, settings and seed are the starter's. One substitution:
// the wasm backend has no gradient kernel for tf.gather (UnsortedSegmentSum)
// and the cpu backend needs hours for it, so inside the page tf.gather(x, idx)
// is replaced by the equivalent tf.matMul(oneHot(idx), x): same rows, same
// gradient (up to float32 rounding). Run:
//   node analysis/reference_mf.js > analysis/reference_mf.txt
"use strict";
const { loadApp } = require("./load.js");

const fmt = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "—");
(async () => {
  const tf = require("@tensorflow/tfjs");
  const gatherRows = (x, indices) => {
    const idx = indices instanceof tf.Tensor ? indices : tf.tensor1d(Int32Array.from(indices), "int32");
    return tf.matMul(tf.oneHot(idx, x.shape[0]).toFloat(), x);
  };
  const app = await loadApp({ backend: "wasm", tfOverrides: { gather: gatherRows } });
  const { run } = app;
  const split = run("HW6_SPLIT");
  const t0 = Date.now();
  // Progress to stderr: hw6TrainMf writes one status line per epoch.
  let last = "";
  const timer = setInterval(() => {
    const s = app.el("status").textContent;
    if (s !== last) { last = s; console.error(`[${((Date.now() - t0) / 1000).toFixed(0)} s] ${s}`); }
  }, 2000);
  const model = await run("hw6TrainMf(HW6_SPLIT.train)");
  clearInterval(timer);
  if (!model) throw new Error("reference training failed");
  const info = run("MF_INFO");
  console.log(`reference: ${info.epochs} epochs, Adam lr ${info.learningRate}, reg ${info.regularization} on the whole matrices, batch ${info.batchSize}, seed ${info.seed}, backend wasm (tf.gather as one-hot matMul); ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  console.log("validation RMSE by epoch: " + info.history.filter((h) => [1, 2, 3, 5, 10, 20, 30].includes(h.epoch)).map((h) => `${h.epoch}: ${fmt(h.valRMSE)}`).join(", "));
  const mu = split.train.reduce((a, r) => a + r.rating, 0) / split.train.length;
  const base = (rows) => Math.sqrt(rows.reduce((a, r) => a + (r.rating - mu) ** 2, 0) / rows.length);
  console.log(`selected epoch ${info.checkpoint}, validation RMSE ${fmt(info.validationRMSE)}; training-mean baseline: validation ${fmt(base(split.val))}, test ${fmt(base(split.test))}; reference test RMSE ${fmt(run("hw6MfRmse(HW6_SPLIT.test)"))}`);
  const norms = run("(function(){ const u = HW6_MF_MODEL.userFactors.dataSync(); const i = HW6_MF_MODEL.itemFactors.dataSync(); let su=0, si=0; for (const x of u) su += x*x; for (const x of i) si += x*x; return [Math.sqrt(su/HW6_MF_MODEL.userFactors.shape[0]), Math.sqrt(si/HW6_MF_MODEL.itemFactors.shape[0])]; })()");
  console.log(`mean vector length: users ${norms[0].toExponential(2)}, movies ${norms[1].toExponential(2)} (HW5 checkpoint: about 1)`);
  const rep = run('recallAtK("mf", 10, "test")');
  const list = run('recommendTopK("mf", 10, 10, "test")');
  const range = run('(function(){ const m = mfScores(10); const s = HW6_STATE.candidateIds.map(function(id){ return m[id]; }); return [Math.min.apply(null, s), Math.max.apply(null, s)]; })()');
  console.log(`reference Recall@10 ${fmt(rep.recall)} over ${rep.cohort.length} users; user 10's scores over all candidates range ${range[0].toFixed(6)} .. ${range[1].toFixed(6)}`);
  console.log(`user 10's reference Top-5: ${list.slice(0, 5).map((r) => `${r.title} (id ${r.movieId}, ${r.score.toFixed(6)})`).join("; ")}`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
