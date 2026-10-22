// analysis/export_hw5_checkpoint.js — how mf/hw5_mf_checkpoint.json was made.
//
// It retrains my HW5 model (https://github.com/getmanyan/llm4rec-hw5) with the
// HW5 page defaults, using HW5's own script.js through HW5's analysis/load.js,
// and saves the selected checkpoint (epoch with the lowest validation RMSE).
// Training is seeded, so the file is reproducible.
//
// Usage:
//   git clone https://github.com/getmanyan/llm4rec-hw5 ../llm4rec-hw5
//   node analysis/export_hw5_checkpoint.js ../llm4rec-hw5     (~10 min on CPU)
"use strict";
const fs = require("fs");
const path = require("path");

const HW5_DIR = path.resolve(process.argv[2] || path.join(__dirname, "..", "..", "llm4rec-hw5"));
const { loadApp } = require(path.join(HW5_DIR, "analysis", "load.js"));

const OPTS = { epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64 };
const OUT = process.argv[3] || path.join(__dirname, "..", "mf", "hw5_mf_checkpoint.json");

(async () => {
  const app = await loadApp();
  const { tf, run } = app;
  const t0 = Date.now();
  run("initializeFactors(trainSet, NUM_FACTORS)");
  const history = run(`trainingHistory = trainMF(trainSet, valSet, ${JSON.stringify(OPTS)})`);
  const best = run("bestCheckpoint = selectBestCheckpoint(trainingHistory)");
  const U = run("model.userFactors.dataSync()");
  const Q = run("model.itemFactors.dataSync()");
  const out = {
    source: "HW5 (getmanyan/llm4rec-hw5) trainMF, SGD, page defaults",
    opts: OPTS,
    factors: 16,
    seeds: { initUsers: 42, initItems: 43, batchOrder: 2026 },
    backend: tf.getBackend(),
    globalMean: run("model.globalMean"),
    checkpointEpoch: best.epoch,
    validationRMSE: best.valRMSE,
    testRMSE: run("computeRMSEFor(testSet)"),
    history: history.map((h) => ({ epoch: h.epoch, trainRMSE: h.trainRMSE, valRMSE: h.valRMSE })),
    userShape: run("model.userFactors.shape"),
    itemShape: run("model.itemFactors.shape"),
    // toPrecision(9) round-trips every float32 value exactly.
    userFactors: Array.from(U, (x) => Number(x.toPrecision(9))),
    itemFactors: Array.from(Q, (x) => Number(x.toPrecision(9))),
  };
  fs.writeFileSync(OUT, JSON.stringify(out));
  console.log(`epoch ${best.epoch}, val ${best.valRMSE.toFixed(4)}, test ${out.testRMSE.toFixed(4)}, ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${OUT}`);
})();
