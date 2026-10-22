// analysis/evaluate.js — every number in the HW6 report.
//
// Usage (from the repository root):
//   npm install                         (once; TensorFlow.js 4.22.0 + the wasm backend)
//   node analysis/evaluate.js           (wasm backend, about 15-20 min on a laptop)
//   node analysis/evaluate.js --backend cpu      (pure-JS kernels, several times slower)
//   node analysis/reference_mf.js > analysis/reference_mf.txt   (the starter's MF reference, cpu)
//
// It runs the page's own data.js, graph.js and script.js (see load.js) and calls
// the same functions the buttons call. Nothing here is random, so a re-run
// prints the same numbers (only timings differ). It also writes the three
// exports the readme asks for, produced by the page's own exporters:
//   verification.json, results.csv, recommendations.csv  (repository root)
"use strict";

const fs = require("fs");
const path = require("path");
const { loadApp } = require("./load.js");

const args = process.argv.slice(2);
const argValue = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
};
const BACKEND = argValue("--backend", "wasm");
const ROOT = path.join(__dirname, "..");
const K = 10;

const fmt = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + "%" : "—");
const line = (s = "") => console.log(s);
const head = (s) => line(`\n=== ${s} ===`);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const median = (xs) => {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};
const timed = (fn) => {
  const t0 = Date.now();
  const value = fn();
  return { value, seconds: (Date.now() - t0) / 1000 };
};

(async () => {
  const app = await loadApp({ backend: BACKEND });
  const { tf, run, el } = app;
  const S = (name) => run(name);

  const movies = S("movies");
  const ratings = S("ratings");
  const split = S("HW6_SPLIT");
  const state = S("HW6_STATE");
  const byId = new Map(movies.map((m) => [m.id, m]));
  const title = (id) => byId.get(id).title.replace("Mis�rables", "Misérables");
  const genres = (id) => byId.get(id).genres;
  const pop = (id) => state.popularity.get(id) || 0;

  line(`HW6 offline evaluation — TensorFlow.js ${tf.version.tfjs}, backend ${app.backend}, node ${process.version}`);

  // ------------------------------------------------------------------ data
  head("1. Data and split");
  const numUsers = S("numUsers");
  const cells = numUsers * movies.length;
  line(`users ${numUsers}, movies ${movies.length}, ratings ${ratings.length}; sparsity ${pct(1 - ratings.length / cells, 2)} of ${numUsers} x ${movies.length}`);
  const total = split.train.length + split.val.length + split.test.length;
  line(`split per user by timestamp: train ${split.train.length} (${pct(split.train.length / total)}), val ${split.val.length} (${pct(split.val.length / total)}), test ${split.test.length} (${pct(split.test.length / total)}); users skipped ${S("SPLIT_EXCLUSIONS").length}`);
  line(`users in train ${state.trainUserIds.length}; movies in train (candidate pool) ${state.candidateIds.length}`);
  const posTrain = split.train.filter((r) => r.rating >= 4).length;
  line(`positive training ratings (>= 4.0): ${posTrain} of ${split.train.length} (${pct(posTrain / split.train.length)}); distinct positive pairs ${state.positivePairs}`);
  line(`users with a positive training edge: ${state.usersWithPositiveTrain.size}`);

  // ----------------------------------------------------------------- graph
  head("2. Graph");
  run("hw6HandleBuildGraph()");
  line(el("status").textContent);
  const graph = S("GRAPH");
  const nU = graph.nUsers;
  const nM = graph.nMovies;
  line(`nodes ${graph.nodeCount} = ${nU} users + ${nM} movies (no padding node); positive edges ${graph.positiveEdges} (each followed in both directions)`);
  line(`zero-degree users: ${graph.zeroDegreeUsers.length} (${graph.zeroDegreeUsers.join(", ")}); zero-degree movies: ${graph.zeroDegreeMovies.length}`);
  const degU = Array.from(graph.degree.slice(0, nU));
  const degM = Array.from(graph.degree.slice(nU));
  line(`user degree: min ${Math.min(...degU)}, median ${median(degU)}, max ${Math.max(...degU)}; movie degree: min ${Math.min(...degM)}, median ${median(degM)}, max ${Math.max(...degM)}`);
  line(`R density ${pct(graph.positiveEdges / (nU * nM), 2)} of ${nU} x ${nM}`);
  const zu = graph.zeroDegreeUsers[0];
  if (zu !== undefined) {
    const rs = split.train.filter((r) => r.userId === zu);
    line(`zero-degree user ${zu}: ${rs.length} training ratings, max ${Math.max(...rs.map((r) => r.rating))}`);
  }

  // ---------------------------------------------------------- verification
  head("3. Verification");
  const memBefore = tf.memory().numTensors;
  const checks = await (async () => {
    await run("hw6EnsureMfForNeed()");
    return run("runSelfChecks()");
  })();
  line(`page self-checks: pass ${checks.passed}, fail ${checks.failed}, pending ${checks.pending}`);
  for (const c of checks.checks) line(`  ${c.status}  ${c.name}${c.detail ? " — " + c.detail : ""}`);

  // 3a. Transition normalization on the MovieLens graph (plain JS, from R).
  let maxRowErr = 0;
  let zeroRows = 0;
  for (let u = 0; u < nU; u++) {
    const d = graph.R[u].reduce((a, b) => a + b, 0);
    if (d === 0) { zeroRows++; continue; }
    maxRowErr = Math.max(maxRowErr, Math.abs(graph.R[u].reduce((a, b) => a + b / d, 0) - 1));
  }
  for (let i = 0; i < nM; i++) {
    let d = 0;
    for (let u = 0; u < nU; u++) d += graph.R[u][i];
    if (d === 0) { zeroRows++; continue; }
    let s = 0;
    for (let u = 0; u < nU; u++) s += graph.R[u][i] / d;
    maxRowErr = Math.max(maxRowErr, Math.abs(s - 1));
  }
  line(`[3a] MovieLens transition: ${graph.nodeCount - zeroRows} positive-degree rows sum to 1 (max error ${maxRowErr.toExponential(1)}), ${zeroRows} zero-degree rows are all zero`);

  // 3b. Global PageRank and PPR on MovieLens: mass, sign, convergence.
  const ALPHAS = S("ALPHA_GRID");
  for (const a of ALPHAS) {
    const pr = run(`pageRank(${a})`);
    const sum = pr.p.reduce((x, y) => x + y, 0);
    const minv = Math.min(...pr.p);
    line(`[3b] global PageRank alpha ${a.toFixed(2)}: ${pr.iterations} iterations, residual ${pr.residual.toExponential(2)}, converged ${pr.converged}, sum ${sum.toFixed(7)}, min ${minv.toExponential(2)}`);
  }
  const node10 = graph.nodeOfUser(10);
  for (const a of ALPHAS) {
    const r = run(`personalizedPageRank(${node10}, ${a})`);
    const sum = r.p.reduce((x, y) => x + y, 0);
    line(`[3b] PPR user 10 (node ${node10}) alpha ${a.toFixed(2)}: ${r.iterations} iterations, residual ${r.residual.toExponential(2)}, converged ${r.converged}, sum ${sum.toFixed(7)}, own-node score ${r.p[node10].toFixed(4)} (restart prob ${(1 - a).toFixed(2)})`);
  }

  // 3c. Zero-degree handling on MovieLens.
  if (zu !== undefined) {
    const z = run(`personalizedPageRank(${graph.nodeOfUser(zu)}, 0.85)`);
    line(`[3c] PPR seeded at zero-degree user ${zu}: sum ${z.p.reduce((x, y) => x + y, 0).toFixed(7)}, mass on the seed ${z.p[graph.nodeOfUser(zu)].toFixed(7)}, ${z.iterations} iteration(s) — the walk cannot leave, all movie scores are 0`);
  }
  {
    const r = run(`personalizedPageRank(${node10}, 0.85)`);
    const zm = graph.zeroDegreeMovies.map((m) => r.p[nU + graph.movieIndexOf.get(m)]);
    const g = run("pageRank(0.85)");
    const zg = graph.zeroDegreeMovies.map((m) => g.p[nU + graph.movieIndexOf.get(m)]);
    const posMin = Math.min(...graph.movieIds.filter((m) => pop(m) > 0).map((m) => g.p[nU + graph.movieIndexOf.get(m)]));
    line(`[3c] ${graph.zeroDegreeMovies.length} zero-degree movies: PPR(user 10) score max ${Math.max(...zm)}; global PageRank score ${zg[0].toExponential(3)} each (restart share only), below every positive-degree movie (min ${posMin.toExponential(3)})`);
  }

  // 3d. Independent float64 implementation (adjacency lists, no TF.js).
  const adjU = Array.from({ length: nU }, (_, u) => graph.R[u].reduce((acc, x, i) => (x ? (acc.push(i), acc) : acc), []));
  const adjM = Array.from({ length: nM }, () => []);
  adjU.forEach((list, u) => list.forEach((i) => adjM[i].push(u)));
  const plainPPR = (vIndex, a) => {
    const n = nU + nM;
    const v = new Float64Array(n);
    if (vIndex === null) v.fill(1 / n); else v[vIndex] = 1;
    let p = Float64Array.from(v);
    let it = 0;
    let res = Infinity;
    while (it < 500) {
      const next = new Float64Array(n);
      let d = 0;
      for (let u = 0; u < nU; u++) {
        const L = adjU[u];
        if (!L.length) { d += p[u]; continue; }
        const share = p[u] / L.length;
        for (const i of L) next[nU + i] += share;
      }
      for (let i = 0; i < nM; i++) {
        const L = adjM[i];
        if (!L.length) { d += p[nU + i]; continue; }
        const share = p[nU + i] / L.length;
        for (const u of L) next[u] += share;
      }
      res = 0;
      for (let j = 0; j < n; j++) {
        next[j] = a * (next[j] + d * v[j]) + (1 - a) * v[j];
        res += Math.abs(next[j] - p[j]);
      }
      p = next;
      it++;
      if (res < 1e-6) break;
    }
    return { p, it, res };
  };
  {
    const g64 = plainPPR(null, 0.85);
    const g32 = run("pageRank(0.85)");
    let md = 0;
    for (let j = 0; j < g64.p.length; j++) md = Math.max(md, Math.abs(g64.p[j] - g32.p[j]));
    line(`[3d] global PageRank alpha 0.85, float64 plain JS vs TF.js: ${g64.it} vs ${g32.iterations} iterations, max |diff| ${md.toExponential(2)}`);
  }

  // 3e. Starter adapter bug: raw user id passed where a node index is expected.
  {
    const right = run(`personalizedPageRank(${node10}, 0.85)`);
    const wrong = run(`personalizedPageRank(10, 0.85)`); // what the starter adapter called
    const seedOfWrong = wrong.p.indexOf(Math.max(...wrong.p.slice(0, nU)));
    line(`[3e] starter adapter: personalizedPageRank(10) restarts at node 10 = user ${graph.userIds[10]}, not user 10 (node ${node10}); highest user-node mass at node ${seedOfWrong} (user ${graph.userIds[seedOfWrong]}). Fixed adapter: own-node mass ${right.p[node10].toFixed(4)}`);
    const last = run(`(function(){ try { return GRAPH.isUserNode(${graph.userIds[nU - 1]}) } catch (e) { return String(e) } })()`);
    line(`[3e] with the raw id, user ${graph.userIds[nU - 1]} would map to node ${graph.userIds[nU - 1]}, which ${last ? "is" : "is NOT"} a user node (it is the first movie node)`);
  }

  // 3f. Memory: repeated runs must not leave tensors behind.
  {
    run("pageRank(0.85)");
    const t0 = tf.memory().numTensors;
    for (let k = 0; k < 5; k++) {
      run("pageRank(0.85)");
      run(`personalizedPageRank(${node10}, 0.85)`);
      run(`personalizedPageRankBatch([0,1,2,3,4,5,6,7], 0.70)`);
      run(`recommendTopK("mf", 10, 10, "test")`);
    }
    const t1 = tf.memory().numTensors;
    line(`[3f] live tensors before / after 5 rounds of PageRank + PPR + batched PPR + MF Top-10: ${t0} / ${t1} (page start: ${memBefore}; the difference is the cached graph tensors and the MF variables)`);
  }

  // ------------------------------------------------------- alpha selection
  head("4. Alpha selection on validation (PPR, Recall@10)");
  const sweep = timed(() => run("hw6RunAlphaSweep()"));
  for (const row of sweep.value.rows) {
    const st = run(`HW6_STATE.pprBatchStats && HW6_STATE.pprBatchStats["${row.alpha.toFixed(2)}"]`);
    line(`alpha ${row.alpha.toFixed(2)}: validation Recall@10 ${fmt(row.recall)} over ${row.cohortSize} users; PPR iterations ${st.minIter}-${st.maxIter}, max residual ${st.maxResidual.toExponential(2)}, all converged ${st.allConverged}${row.alpha === sweep.value.best ? "  <- selected" : ""}`);
  }
  const best = sweep.value.best;
  line(`selected alpha ${best.toFixed(2)} (validation only, ties toward the smaller alpha); sweep time ${sweep.seconds.toFixed(0)} s on ${app.backend}`);
  const valEx = run('hw6CohortExclusions("val")');
  line(`validation cohort ${valEx.cohortSize} of ${valEx.candidateUsers} users with validation ratings; excluded: ${valEx.excludedNoEligible} without an eligible relevant movie, ${valEx.excludedNoPositiveEdge} without a positive training edge; ${valEx.excludedUnseen} relevant ratings on movies absent from train`);
  const valOther = {};
  for (const m of ["popularity", "mf"]) valOther[m] = run(`recallAtK("${m}", 10, "val")`).recall;
  run(`hw6SetAlpha(${best})`);
  valOther.pagerank = run('recallAtK("pagerank", 10, "val")').recall;
  line(`for reference, validation Recall@10 of the other methods: popularity ${fmt(valOther.popularity)}, global PageRank (alpha ${best.toFixed(2)}) ${fmt(valOther.pagerank)}, MF ${fmt(valOther.mf)}`);

  // ------------------------------------------------------------- test eval
  head(`5. Test evaluation (shared cohort, alpha ${best.toFixed(2)} for PageRank and PPR)`);
  run(`hw6SetAlpha(${best})`);
  const testEx = run('hw6CohortExclusions("test")');
  line(`test cohort ${testEx.cohortSize} of ${testEx.candidateUsers} users with test ratings; excluded: ${testEx.excludedNoEligible} users without an eligible relevant movie (of them ${testEx.excludedNoPositiveEdge} without a positive training edge); ${testEx.excludedUnseen} relevant test ratings on ${testEx.unseenIds.length} movies absent from train`);
  const METHODS = ["popularity", "pagerank", "ppr", "mf"];
  const reports = {};
  const lists = {};
  const cohort = run('hw6SharedCohort("test")');
  const top100 = new Set(state.candidateIds.slice().sort((a, b) => pop(b) - pop(a) || a - b).slice(0, 100));
  line("method      Recall@10  users with a hit  hits/relevant  mean list pop.  distinct movies  slots in top-100 pop.  slots pop<=5");
  for (const m of METHODS) {
    const rep = run(`recallAtK("${m}", 10, "test")`);
    reports[m] = rep;
    const L = new Map();
    for (const u of cohort) L.set(u, run(`recommendTopK("${m}", ${u}, 10, "test")`));
    lists[m] = L;
    const slots = [].concat(...[...L.values()]);
    const hits = rep.perUser.reduce((a, r) => a + r.hit, 0);
    const relevant = rep.perUser.reduce((a, r) => a + r.relevant, 0);
    const usersHit = rep.perUser.filter((r) => r.hit > 0).length;
    const meanPop = mean([...L.values()].map((recs) => mean(recs.map((r) => r.popularity))));
    const distinct = new Set(slots.map((r) => r.movieId)).size;
    line(`${m.padEnd(11)} ${fmt(rep.recall).padEnd(10)} ${String(usersHit).padEnd(17)} ${(hits + "/" + relevant).padEnd(14)} ${fmt(meanPop, 1).padEnd(15)} ${String(distinct).padEnd(16)} ${pct(slots.filter((r) => top100.has(r.movieId)).length / slots.length).padEnd(22)} ${pct(slots.filter((r) => r.popularity <= 5).length / slots.length)}`);
  }
  line(`HW5 report for comparison: MF Recall@10 0.0443 over 861 users (HW5 cohort: users with a relevant test movie in the train pool)`);

  // 5b. How alike are the lists?
  const overlap = (a, b) => mean(cohort.map((u) => {
    const s = new Set(lists[a].get(u).map((r) => r.movieId));
    return lists[b].get(u).filter((r) => s.has(r.movieId)).length;
  }));
  line(`mean common movies per user between methods (of 10): pop-PR ${fmt(overlap("popularity", "pagerank"), 2)}, pop-PPR ${fmt(overlap("popularity", "ppr"), 2)}, PR-PPR ${fmt(overlap("pagerank", "ppr"), 2)}, PPR-MF ${fmt(overlap("ppr", "mf"), 2)}, pop-MF ${fmt(overlap("popularity", "mf"), 2)}`);
  // per-user wins
  let pprBetter = 0, mfBetter = 0, tie0 = 0, tiePos = 0;
  const mfHit = new Map(reports.mf.perUser.map((r) => [r.userId, r.hit / r.relevant]));
  for (const r of reports.ppr.perUser) {
    const a = r.hit / r.relevant;
    const b = mfHit.get(r.userId);
    if (a > b) pprBetter++; else if (b > a) mfBetter++; else if (a === 0) tie0++; else tiePos++;
  }
  line(`per user, PPR vs MF recall: PPR higher ${pprBetter}, MF higher ${mfBetter}, both 0 ${tie0}, equal and > 0 ${tiePos}`);
  const hitBoth = cohort.filter((u) => {
    const rel = new Set();
    run(`HW6_STATE.gradesBySplit.test.get(${u})`).forEach((rt, m) => { if (rt >= 4) rel.add(m); });
    const a = lists.ppr.get(u).filter((r) => rel.has(r.movieId)).map((r) => r.movieId);
    const b = new Set(lists.mf.get(u).filter((r) => rel.has(r.movieId)).map((r) => r.movieId));
    return a.some((m) => !b.has(m)) && b.size > 0;
  }).length;
  line(`users for whom PPR and MF each find a relevant movie the other misses: ${hitBoth}`);

  // 5c. Recall by user activity (training positives), PPR vs MF.
  const posCount = new Map();
  for (const r of split.train) if (r.rating >= 4) posCount.set(r.userId, (posCount.get(r.userId) || 0) + 1);
  const bucket = (u) => { const c = posCount.get(u) || 0; return c < 20 ? "<20" : c < 50 ? "20-49" : c < 100 ? "50-99" : ">=100"; };
  const buckets = ["<20", "20-49", "50-99", ">=100"];
  line("recall by number of positive training ratings:");
  for (const b of buckets) {
    const us = new Set(cohort.filter((u) => bucket(u) === b));
    const r = (m) => mean(reports[m].perUser.filter((x) => us.has(x.userId)).map((x) => x.hit / x.relevant));
    line(`  ${b.padEnd(6)} users ${String(us.size).padEnd(4)} popularity ${fmt(r("popularity"))}  PageRank ${fmt(r("pagerank"))}  PPR ${fmt(r("ppr"))}  MF ${fmt(r("mf"))}`);
  }

  // 5d. Independent recall for PPR (float64 plain JS on adjacency lists).
  {
    const t = timed(() => {
      let sum = 0, same = 0;
      for (const u of cohort) {
        const res = plainPPR(graph.nodeOfUser(u), best);
        const excluded = run(`hw6ExclusionSet("test", ${u})`);
        const ranked = state.candidateIds
          .filter((m) => !excluded.has(m))
          .map((m) => [m, res.p[nU + graph.movieIndexOf.get(m)]])
          .sort((x, y) => y[1] - x[1] || x[0] - y[0])
          .slice(0, 10)
          .map((x) => x[0]);
        const rel = [];
        run(`HW6_STATE.gradesBySplit.test.get(${u})`).forEach((rt, m) => { if (rt >= 4 && state.candidateSet.has(m) && !excluded.has(m)) rel.push(m); });
        sum += rel.filter((m) => ranked.includes(m)).length / rel.length;
        const tfList = lists.ppr.get(u).map((r) => r.movieId);
        if (tfList.join(",") === ranked.join(",")) same++;
      }
      return { recall: sum / cohort.length, same };
    });
    line(`[check] PPR Recall@10 from a float64 plain-JS power iteration: ${fmt(t.value.recall)} (TF.js ${fmt(reports.ppr.recall)}); identical Top-10 lists for ${t.value.same} of ${cohort.length} users (${t.seconds.toFixed(0)} s)`);
  }
  // 5e. Exclusion audit across every list.
  {
    let bad = 0, outside = 0, n = 0;
    for (const m of METHODS) for (const u of cohort) {
      const ex = run(`hw6ExclusionSet("test", ${u})`);
      for (const r of lists[m].get(u)) { n++; if (ex.has(r.movieId)) bad++; if (!state.candidateSet.has(r.movieId)) outside++; }
    }
    line(`[check] ${n} test slots (4 methods x ${cohort.length} users): ${bad} already rated in train/val, ${outside} outside the training pool; every list has 10 items: ${METHODS.every((m) => cohort.every((u) => lists[m].get(u).length === 10))}`);
    const zeroScorePPR = cohort.filter((u) => lists.ppr.get(u).some((r) => !(r.score > 0))).length;
    line(`[check] PPR lists that needed a zero-score filler: ${zeroScorePPR}`);
  }

  // ------------------------------------------------------------- two users
  const [UA, UB] = S("HW6_REPORT_USERS");
  head(`6. The two users from HW5: ${UA} and ${UB} (test Top-10, alpha ${best.toFixed(2)})`);
  const histOf = (u) => [...split.train, ...split.val].filter((r) => r.userId === u);
  const topGenres = (rows, n = 3) => {
    const c = new Map();
    for (const r of rows) if (r.rating >= 4) for (const g of genres(r.movieId)) c.set(g, (c.get(g) || 0) + 1);
    return [...c.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, n);
  };
  for (const u of [UA, UB]) {
    const h = histOf(u);
    const tr = split.train.filter((r) => r.userId === u);
    const liked = tr.filter((r) => r.rating >= 4).sort((a, b) => b.rating - a.rating || pop(b.movieId) - pop(a.movieId));
    line(`user ${u}: in cohort ${cohort.includes(u)}; train ${tr.length} ratings (mean ${fmt(mean(tr.map((r) => r.rating)), 2)}), ${liked.length} positive = graph degree ${graph.degree[graph.nodeOfUser(u)]}; train+val ${h.length}`);
    line(`  top genres among his positive train+val ratings: ${topGenres(h).map(([g, c]) => g + " " + c).join(", ")}`);
    line(`  mean training popularity of his positive train movies: ${fmt(mean(liked.map((r) => pop(r.movieId))), 1)}`);
    line(`  examples of 5-star / 4-star train movies: ${liked.slice(0, 8).map((r) => `${title(r.movieId)} [${r.rating}]`).join("; ")}`);
    const test = split.test.filter((r) => r.userId === u);
    line(`  test period: ${test.length} ratings, relevant (>= 4) eligible: ${reports.ppr.perUser.find((x) => x.userId === u)?.relevant ?? 0}: ${test.filter((r) => r.rating >= 4).map((r) => title(r.movieId)).join("; ")}`);
  }
  const userTopG = new Map([UA, UB].map((u) => [u, new Set(topGenres(histOf(u)).map(([g]) => g))]));
  for (const m of METHODS) {
    for (const u of [UA, UB]) {
      const recs = run(`recommendTopK("${m}", ${u}, 10, "test")`);
      const testRated = new Map(split.test.filter((r) => r.userId === u).map((r) => [r.movieId, r.rating]));
      const inTop = recs.filter((r) => r.genres.some((g) => userTopG.get(u).has(g))).length;
      const per = reports[m].perUser.find((x) => x.userId === u);
      line(`\n[${m}] user ${u}: recall ${per ? per.hit + "/" + per.relevant : "n/a"}, mean popularity ${fmt(mean(recs.map((r) => r.popularity)), 1)}, ${inTop}/10 share one of his top-3 genres, ${recs.filter((r) => top100.has(r.movieId)).length}/10 in the 100 most popular`);
      recs.forEach((r, i) => {
        const t = testRated.has(r.movieId) ? `  <- rated ${testRated.get(r.movieId)} in test` : "";
        line(`  ${String(i + 1).padStart(2)}. ${title(r.movieId)} | ${r.genres.join(", ")} | score ${m === "ppr" || m === "pagerank" ? r.score.toExponential(3) : fmt(r.score)} | pop ${r.popularity}${t}`);
      });
    }
    const a = new Set(run(`recommendTopK("${m}", ${UA}, 10, "test")`).map((r) => r.movieId));
    const common = run(`recommendTopK("${m}", ${UB}, 10, "test")`).filter((r) => a.has(r.movieId)).length;
    line(`[${m}] common movies between users ${UA} and ${UB}: ${common} of 10`);
  }
  for (const u of [UA, UB]) {
    const a = new Set(run(`recommendTopK("ppr", ${u}, 10, "test")`).map((r) => r.movieId));
    const c = run(`recommendTopK("mf", ${u}, 10, "test")`).filter((r) => a.has(r.movieId)).length;
    line(`user ${u}: PPR and MF lists share ${c} of 10`);
  }

  // --------------------------------------------------- one factor at a time
  head("7. One factor at a time (readme 6.2)");
  el("user-select").value = String(UA);
  el("seed-user-select").value = String(UB);
  const demo = run("hw6RunTwoUserDemo()");
  if (demo.error) line("demo error: " + demo.error);
  for (const s of demo.settings) {
    line(`${s.label}: user ${s.user}, alpha ${s.alpha.toFixed(2)} — mean popularity ${fmt(mean(s.recs.map((r) => r.popularity)), 1)}, ${s.recs.filter((r) => top100.has(r.movieId)).length}/10 in top-100`);
    s.recs.forEach((r, i) => line(`  ${String(i + 1).padStart(2)}. ${title(r.movieId)} | ${r.genres.join(", ")} | ${r.score.toExponential(3)} | pop ${r.popularity}`));
  }
  for (const c of demo.common) line(`common: ${c.pair}: ${c.shared} of ${c.of}`);

  // ----------------------------------------------- instructor MF reference
  head("8. The starter's MF reference (Adam, L2 on the whole matrices) vs my HW5 checkpoint");
  const mfInfo = run("MF_INFO");
  line(`used for "mf": ${mfInfo.source}; checkpoint epoch ${mfInfo.checkpoint}, validation RMSE ${fmt(mfInfo.validationRMSE)}, test RMSE ${fmt(mfInfo.testRMSE)} (trained on ${mfInfo.backend})`);
  line("the starter's reference trainer (hw6TrainMf) is evaluated separately (about 3 min on wasm):");
  line("it collapses to the training mean; see analysis/reference_mf.txt, made with");
  line("  node analysis/reference_mf.js > analysis/reference_mf.txt");

  // ------------------------------------------------------------- exports
  head("9. Exports (the page's own exporters)");
  run("HW6_STATE.recommendationRows = []; HW6_STATE.recommendationKeys = new Set();");
  // Exactly what the "Evaluate Recall@10" button does: sweep, select, compare.
  el("method").value = "ppr";
  el("user-select").value = String(UA);
  el("seed-user-select").value = String(UB);
  run("hw6SetAlpha(0.85)");
  await run("hw6HandleRecall()");
  line(el("status").textContent);
  for (const m of METHODS) {
    for (const u of [UA, UB]) {
      el("method").value = m;
      el("user-select").value = String(u);
      run("hw6CurrentRecommendations()");
    }
  }
  el("method").value = "ppr";
  el("user-select").value = String(UA);
  el("seed-user-select").value = String(UB);
  run("hw6HandleTwoUserDemo()");
  line(el("status").textContent);
  run("hw6HandleRunPageRank(); hw6HandleRunPPR();");
  await run("hw6HandleRunTests()");
  fs.writeFileSync(path.join(ROOT, "verification.json"), run("serializeVerification()"));
  fs.writeFileSync(path.join(ROOT, "results.csv"), run("serializeResults()"));
  fs.writeFileSync(path.join(ROOT, "recommendations.csv"), run("serializeRecommendations()"));
  const v = JSON.parse(run("serializeVerification()"));
  line(`verification.json: pass ${v.passed}, fail ${v.failed}, pending ${v.pending}; results.csv ${run("serializeResults()").trim().split("\n").length - 1} rows; recommendations.csv ${run("HW6_STATE.recommendationRows.length")} rows`);
  line(`\ndone, backend ${app.backend}`);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
