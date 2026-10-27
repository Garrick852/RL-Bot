"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { runInNewContext } = require("node:vm");

const html = readFileSync(join(__dirname, "..", "specs", "offline-orchestrator-flow.html"), "utf8");
const context = {};
runInNewContext(html.match(/<script id="flow-model">([\s\S]*?)<\/script>/)[1], context);
const { resolve, median, audit } = context.FlowModel;
const valid = { errors: [], complete: true, frozen: true, checks: ["PASS", "PASS", "PASS"] };
const rows = [7, 42, 99].map(seed => ({ seed, pf: 1.4, sharpe: 1.2, dd: .08, trades: 40, expectancy: 10, complete: true }));

test("readiness precedence, freeze contradiction, and evidence invalidation", () => {
  assert.equal(resolve(valid), "READY_FOR_OFFLINE_EXECUTION");
  assert.equal(resolve({ ...valid, frozen: false }), "DRAFT_INCOMPLETE");
  assert.equal(resolve({ ...valid, complete: false }), "INVALID");
  assert.equal(resolve({ ...valid, checks: ["PASS", "PASS", "UNCHECKED"] }), "EVIDENCE_UNCHECKED");
  assert.equal(resolve({ ...valid, frozen: false, checks: ["PASS", "PASS", "FAIL"] }), "INVALID");
  assert.equal(resolve({ ...valid, errors: ["bad type"] }), "INVALID");
  assert.equal(resolve({ ...valid, frozen: "true" }), "INVALID");
  assert.equal(resolve({ ...valid, checks: ["PASS", "PASS", "ASSUMED"] }), "INVALID");
  assert.equal(resolve({ ...valid, complete: false, frozen: false }), "DRAFT_INCOMPLETE");
});

test("medians are numeric, componentwise, and fail closed on undefined values", () => {
  assert.equal(median([99, 2, 4]), 4);
  assert.equal(median([1, 5, 3, 7]), 4);
  assert.equal(median([1, null, 2]), null);
  assert.equal(median([]), null);
  assert.equal(audit(rows).pass, true);
  assert.equal(audit(rows.map((row, index) => ({ ...row, pf: [1.1, 1.5, 1.4][index] }))).pass, true);
  assert.equal(audit(rows.map((row, index) => ({ ...row, pf: [1.1, 1.2, 10][index] }))).pass, false);
});

test("missing, duplicate, failed, or risky seeds cannot pass", () => {
  for (const patch of [{ complete: false }, { pf: null }, { sharpe: Infinity }, { dd: .11 }, { trades: 29 }, { expectancy: 0 }]) {
    assert.equal(audit([{ ...rows[0], ...patch }, ...rows.slice(1)]).pass, false);
  }
  assert.equal(audit(rows.slice(1)).pass, false);
  assert.equal(audit([rows[0], rows[0], rows[2]]).pass, false);
  assert.equal(audit(rows, [7, 42, 42]).pass, false);
});

test("dashboard is self-contained with three accessible SVG visualizations", () => {
  assert.equal((html.match(/<svg /g) || []).length, 3);
  assert.equal((html.match(/role="img"/g) || []).length, 3);
  assert.doesNotMatch(html, /<script[^>]+src=|<link|@import|fetch\(|XMLHttpRequest|localStorage/);
  assert.match(html, /connect-src 'none'/);
  assert.match(html, /prefers-reduced-motion/);
  assert.match(html, /SYNTHETIC VISUALIZATION/);
});
