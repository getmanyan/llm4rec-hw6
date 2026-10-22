// analysis/load.js — runs the page's own data.js, graph.js and script.js in Node.
//
// The browser page and this loader execute the same three files, so every
// number printed by evaluate.js comes from the code that GitHub Pages serves.
// The DOM is replaced by small stub elements (they only store value /
// textContent / innerHTML), fetch() reads files from the repository folder,
// and d3 is not loaded (the graph view is drawing only and reports that d3 is
// missing). TensorFlow.js runs on the backend chosen with --backend:
//   cpu  (default) pure-JavaScript kernels, the same as the page's CPU fallback
//   wasm WebAssembly kernels (@tensorflow/tfjs-backend-wasm), much faster
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE_DIR = path.join(__dirname, "..");

function requireTf(backend) {
  let tf;
  try {
    tf = require("@tensorflow/tfjs");
  } catch (error) {
    console.error("Install TensorFlow.js first:  npm install");
    throw error;
  }
  if (backend === "wasm") require("@tensorflow/tfjs-backend-wasm");
  return tf;
}

/** A DOM element stand-in: it keeps what the page writes into it. */
function makeElement(id) {
  return {
    id,
    value: id === "alpha" ? "0.85" : id === "max-nodes" ? "120" : "",
    textContent: "",
    innerHTML: "",
    className: "",
    style: {},
    children: [],
    addEventListener() {},
    removeEventListener() {},
    appendChild(child) { this.children.push(child); return child; },
    removeChild() {},
    setAttribute() {},
    getAttribute() { return null; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
}

/**
 * Load the page in a VM context.
 *
 * @param {{backend?: string}} [options]
 * @returns {Promise<{tf: Object, run: function(string): *, context: Object, el: function(string): Object}>}
 */
async function loadApp(options) {
  const backend = (options && options.backend) || "cpu";
  const tf = requireTf(backend);
  await tf.setBackend(backend);
  await tf.ready();

  // The page asks for "webgl" first; in Node that backend does not exist. The
  // page's own fallback would pick "cpu"; here the backend chosen above is kept.
  const keepBackend = async () => true;
  const overrides = (options && options.tfOverrides) || {};
  const tfForPage = new Proxy(tf, {
    get: (target, key) => (key === "setBackend" ? keepBackend : key in overrides ? overrides[key] : target[key]),
  });

  const elements = new Map();
  const el = (id) => {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  };
  const fakeDocument = {
    readyState: "complete",
    getElementById: el,
    createElement: (tag) => makeElement("<" + tag + ">"),
    addEventListener() {},
    body: makeElement("body"),
  };
  const fakeFetch = async (name) => {
    const file = path.join(PAGE_DIR, String(name));
    if (!fs.existsSync(file)) return { ok: false, status: 404, text: async () => "" };
    const text = fs.readFileSync(file, "utf8");
    return { ok: true, status: 200, text: async () => text };
  };

  const context = vm.createContext({
    tf: tfForPage,
    console,
    fetch: fakeFetch,
    document: fakeDocument,
    setTimeout,
    clearTimeout,
    Math, Number, Array, Set, Map, WeakMap, Object, String, Boolean, Error, JSON, Date, Infinity, NaN,
    Int32Array, Uint8Array, Float32Array, Float64Array, ArrayBuffer, Promise, Symbol, RegExp,
  });
  context.window = context; // top-level functions become window properties, as in a browser

  for (const file of ["data.js", "graph.js", "script.js"]) {
    const code = fs.readFileSync(path.join(PAGE_DIR, file), "utf8");
    vm.runInContext(code, context, { filename: file });
  }

  // script.js starts hw6Bootstrap() on load; wait until the data is ready.
  for (let i = 0; i < 600; i++) {
    if (vm.runInContext("HW6_STATE.loaded", context)) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  if (!vm.runInContext("HW6_STATE.loaded", context)) {
    throw new Error("page bootstrap did not finish: " + el("status").textContent);
  }

  return {
    tf,
    backend: tf.getBackend(),
    run: (expression) => vm.runInContext(expression, context),
    context,
    el,
  };
}

module.exports = { loadApp };
