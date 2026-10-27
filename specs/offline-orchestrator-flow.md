# Offline Orchestrator Flow Specification

**Specification version:** 1.0.0  
**Milestone:** 1 - Offline Single-Strategy Validation  
**Status:** normative design; bounded configuration/hash checks implemented,
full validation contract and execution pipeline not implemented

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.
This specification refines [the phased architecture](../RL-Architecture.md)
and uses the field paths in the
[offline experiment template](../config/offline_experiment.template.json).
It does not freeze that template, select a dataset, or assert that an experiment
has been executed.

## 1. The 4-Tier State Machine

These four states describe **configuration and evidence readiness**, not
training progress or strategy performance.

| State | Exact meaning | Offline execution permitted |
|---|---|---|
| `INVALID` | At least one known blocking structure, type, logical, safety, or evidence error exists | No |
| `DRAFT_INCOMPLETE` | No known blocking error exists, but applicable required settings or explicit freeze confirmation are missing | No |
| `EVIDENCE_UNCHECKED` | Applicable settings are complete and consistent and `frozen` is true, but at least one required L3 check is pending | No |
| `READY_FOR_OFFLINE_EXECUTION` | L1, L2, and all applicable L3 checks pass for the same immutable input snapshot and `frozen` is true | Yes, offline only |

`READY_FOR_OFFLINE_EXECUTION` **strictly grants permission to run the offline
simulation**. It does **not** imply profitability, statistical significance,
successful OOS evaluation, live deployment readiness, or permission to place
orders. The readiness label MUST NOT be displayed as a trading-performance PASS.

### 1.1 Deterministic state resolution

On every evaluation, the validator MUST collect errors, unresolved fields,
warnings, and individual evidence results, then apply this precedence:

1. Any blocking error produces `INVALID`. In particular, `frozen: true` with
   known missing applicable required settings is an invalid freeze declaration.
2. Otherwise, missing applicable required settings or `frozen` not explicitly
   equal to boolean `true` produces `DRAFT_INCOMPLETE`.
3. Otherwise, any pending applicable L3 check produces `EVIDENCE_UNCHECKED`.
4. Otherwise, all required checks MUST be PASS to produce
   `READY_FOR_OFFLINE_EXECUTION`.

An explicitly supplied `frozen` value of the wrong type is an error, not an
unresolved confirmation. A draft MAY omit it or set it to `null` or `false`.
Pre-execution inspection MUST still report L3 outcomes when frozen confirmation
is missing and structurally safe inspection is possible; it MUST NOT infer an
evidence PASS from a flag.

Every evidence item MUST have its own `PASS`, `FAIL`, or `UNCHECKED` status,
with a stable check identifier, relevant field path, and reason. Checks prevented
by an earlier failure remain `UNCHECKED` with the blocking reason. Conditional
non-applicability MUST be justified by a schema-defined applicability rule;
it MUST NOT silently remove a required check.

### 1.2 Explicit transitions

The following transitions are reevaluations, never unconditional assignments:

| From | Event / condition | To |
|---|---|---|
| Any state | A blocking error is discovered, including L3 failure | `INVALID` |
| `INVALID` | Errors corrected; applicable settings or freeze confirmation still unresolved | `DRAFT_INCOMPLETE` |
| `INVALID` or `DRAFT_INCOMPLETE` | Complete, consistent frozen settings; L3 still pending | `EVIDENCE_UNCHECKED` |
| `EVIDENCE_UNCHECKED` | All required L3 checks pass for the frozen snapshot | `READY_FOR_OFFLINE_EXECUTION` |
| `INVALID` or `DRAFT_INCOMPLETE` | Correction/completion followed by a full successful L1-L3 evaluation | `READY_FOR_OFFLINE_EXECUTION` |
| `EVIDENCE_UNCHECKED` or `READY_FOR_OFFLINE_EXECUTION` | Freeze revoked or required settings removed without another blocking error | `DRAFT_INCOMPLETE` |
| `READY_FOR_OFFLINE_EXECUTION` | Data/config/contract identity changes or evidence becomes unavailable | Revoke readiness immediately; resolve to `INVALID`, `DRAFT_INCOMPLETE`, or `EVIDENCE_UNCHECKED` using section 1.1 |
| Any state | No relevant input or evidence changes | Same state |

Correcting an invalid input MUST NOT restore a cached readiness label.
Changed inputs require a new snapshot and fresh evidence. A run MUST verify
snapshot identity again before consuming data or training.

The independent run lifecycle is `NOT_STARTED -> RUNNING -> COMPLETE` or
`RUNNING -> FAILED`. A preflight refusal remains `NOT_STARTED`. A completed run
can have research decision FAIL; readiness itself never becomes an OOS decision.

## 2. L1-L3 Validation Gates

Both modes MUST validate supplied values; draft mode is not permission to accept
malformed values. Missing values MUST NOT be replaced with proposals or defaults.
The current template is a field inventory, not an executable JSON Schema.
A future implementation MUST publish a versioned strict schema and supported
strategy/accounting variants before accepting pre-execution configurations.

### 2.1 Layer 1 - Schema and structure

**Draft mode:**

- Missing applicable required settings or allowed unresolved `null` values
  produce unresolved-field entries.
- Wrong supplied types, malformed objects, and duplicate JSON keys are errors.
  Booleans MUST NOT be accepted as integers or numbers.
- Unknown fields MUST produce warnings and MUST NOT affect computation,
  applicability, or defaults. They MUST be removed or explicitly incorporated
  into a reviewed schema before pre-execution acceptance.

**Pre-execution mode:**

- Unknown fields at every schema-defined object level MUST be rejected.
  Keys in declared maps, such as feature or parameter maps, MUST follow their
  versioned map contract rather than an unrestricted escape hatch.
- Every applicable required field MUST exist and be non-null with the exact
  declared type. Numeric values MUST be finite; JSON `NaN` and infinity are
  forbidden. Enum-like settings MUST be supported, not merely non-empty strings.
- An unsupported template/schema version, unresolved draft specification status,
  or unsupported non-applicability representation MUST block execution.
- `proposals_not_confirmed` MAY remain as informational metadata only. Its
  contents MUST NOT satisfy formal configuration requirements.
- `safety.offline_only` MUST be true. Network access, broker execution,
  active-config writes, and automatic deployment permissions MUST all be false.

Costs of zero require explicit units and application rules. "Not applicable"
requires a supported, verifiable applicability representation; neither a
fabricated zero nor an arbitrary string is acceptable.

### 2.2 Layer 2 - Logical consistency

All timestamps MUST be unambiguous ISO-8601 instants consistent with
`dataset.timezone`; comparisons MUST use normalized instants, not string order.
All partitions use `[start, end)`:

```text
splits.train.start < splits.train.end
splits.validation.start < splits.validation.end
splits.test.start < splits.test.end
splits.train.end <= splits.validation.start
splits.validation.end <= splits.test.start
```

Equality between adjacent boundaries is permitted. Gaps are permitted only
under the declared calendar/partition policy. `validation` is for model selection;
`test` is the final OOS evaluation. They MUST NOT be conflated.

When fixed EMA crossover is the confirmed strategy, `fast_period` and
`slow_period` MUST be positive integers and `fast_period < slow_period`.
All other declared bounds MUST have finite supported endpoints with lower <=
upper; strict strategy constraints MUST hold for every admissible parameter
combination, or invalid combinations MUST be explicitly excluded.

`training.seeds` MUST contain at least three distinct, supported integer seeds,
with no boolean values. No seed list may be inferred from the proposal section.
Action mapping MUST match the explicitly confirmed policy interface and
instrument sizing. `0/1/2 -> -1/0/+1` is required only if explicitly adopted.

Costs, currency conversion, fill timing, leverage, and liquidation rules MUST
be internally consistent. Warm-up length MUST cover every declared indicator
and scaler input requirement. The scaler MUST fit training data only;
`features.refit_on_validation_or_test` MUST be false.

### 2.3 Layer 3 - Data evidence and hashing

L3 MUST enforce the **Data & Feature Contract** in section 2.4, which makes the
data isolation rules in sections 3-5 of the architecture explicit for this flow.
It MUST:

1. Resolve `dataset.path` to a physical, readable local regular file. No network
   download, remote path fallback, sample substitution, or broker fetch is allowed.
2. Compute SHA-256 over the exact raw file bytes and compare the lowercase
   64-character hexadecimal digest to `dataset.sha256` exactly. A mismatch is
   `INVALID`, not a warning. Hashing parsed/reformatted rows is not equivalent.
3. Parse and validate the same immutable bytes that were hashed. A local
   content-addressed snapshot MAY be used. If an immutable snapshot cannot be
   established, a concurrent modification MUST abort validation/execution.
4. Verify actual partition coverage, gap handling, indicator warm-up sufficiency,
   timestamp order, price basis, and schema-defined OHLCV requirements.
5. Record check results, actual row counts, first/last scored timestamps,
   snapshot identity, and the contract version/hash used.

An inaccessible file, hash mismatch, unsupported policy, insufficient history,
or violated contract is a FAIL. A check not performed is UNCHECKED.

### 2.4 Data & Feature Contract

This subsection is the referenced contract for Milestone 1; it is a specification,
not a claim that `data/collector.py` currently enforces every requirement.

- Source column mapping, timestamp meaning, timeframe, timezone, and market
  calendar MUST be explicit and supported. Timestamps MUST be unique and
  strictly increasing. Required numeric fields MUST be finite, volume
  non-negative, and OHLC ranges coherent under the instrument's price rules.
- `dataset.missing_bar_policy` MUST distinguish valid session/holiday closures
  from unexpected gaps. Undeclared imputation, silent row dropping, or filling
  with future observations is forbidden. Unsupported calendar/policy handling
  MUST block execution rather than assume a continuously traded market.
- Each scored partition MUST contain sufficient executable observations under
  the frozen evaluation contract. Earlier observations MAY supply warm-up to a
  later partition, but warm-up MUST NOT count as scored trades or returns.
- Features at decision time MUST depend only on information available then.
  Scalers MUST be fitted only to eligible training observations and applied
  unchanged to validation/test. Save ordered feature names, fitted statistics,
  training fit boundaries, scaler hash, and excluded warm-up rows.
- Orders MUST use the declared next executable price, never a future-known fill
  at the decision price. Final-bar entries with no subsequent fill are forbidden.
  All strategies MUST liquidate terminal positions with declared exit costs.
- L3 may inspect test-file integrity and coverage, but no test performance,
  reward, fitted statistic, or feature distribution may influence training,
  scaling, or model selection. Any later model-dependent check is a run-phase
  check, not fabricated preflight evidence.

## 3. Three-Seed Statistical Execution

### 3.1 Execution protocol

A valid Milestone 1 RL experiment MUST train and evaluate **every predeclared
seed**, with at least three distinct seeds, for example `7`, `42`, and `99`.
The examples are not defaults. A single-seed debugging run MUST NOT qualify as
the milestone experiment or receive an audit PASS.

Python, NumPy, PyTorch, environment, and action-space RNGs MUST be seeded.
CPU thread limits, dependency versions, deterministic settings, and training
budget MUST be recorded. Each seed uses identical splits, costs, features,
training budget, and a frozen validation-only checkpoint-selection rule.
Runs SHOULD execute sequentially for predictable CPU use.

Freeze every selected model before evaluating final test performance.
Evaluate all seed policies and confirmed baselines on identical OOS timestamps
using one accounting engine. No best-seed selection, failed-seed replacement,
selective omission, or OOS-driven retraining is permitted. Any missing/failed
seed makes the experiment incomplete and the audit decision FAIL.

### 3.2 Exact median-based OOS decision

For each metric, take its median across **all configured seeds**, never across
the best subset, individual trades pooled across seeds, or averaged policies.
Sort the per-seed values: for an odd count use the middle value; for an even
count use the arithmetic mean of the two middle values. Compute PF and Sharpe
medians independently; they need not come from the same seed.

The initial research profile uses these thresholds, recorded explicitly in
`evaluation.research_gate_thresholds` before freezing:

| Gate | Required rule |
|---|---|
| Aggregate profit factor | Median OOS PF >= 1.25 |
| Aggregate Sharpe | Median OOS Sharpe >= 1.0 using the frozen sampling/annualization definition |
| Downside protection | Every seed's maximum marked-equity drawdown <= 10% |
| Sample sufficiency | Every seed has at least 30 realized closed OOS trades |
| Seed stability | Every configured seed has strictly positive net OOS expectancy after all costs |

This profile MUST NOT be silently inferred from missing config values.
Medians are the aggregate performance gates; individual seed PF/Sharpe values
need not each pass those aggregate thresholds. The per-seed risk/sample/stability
guards prevent the median from concealing an unacceptable run.

Audit `pass` is true **if and only if** readiness was verified for the consumed
snapshots, the run completed, all required seed/baseline artifacts reconcile,
every RL gate metric is defined and finite, both median gates pass, and every
per-seed guard passes. Any other outcome is FAIL with explicit reasons.

PF MUST use the sum of positive net closed-trade P&L divided by the absolute
sum of negative net closed-trade P&L, with costs allocated under the frozen
accounting contract. Expectancy is mean net closed-trade P&L. Maximum drawdown
MUST use the marked-equity series and running peaks, not closed trades only.
Sharpe MUST use the frozen return frequency, risk-free convention, variance
estimator, and annualization rule consistently for every seed and baseline.

No-loss PF is unbounded, zero-variance Sharpe is undefined, and an empty trade
sample is insufficient. Such metrics MUST be represented as `null` with a
reason, MUST NOT be replaced with favorable constants, and MUST NOT be excluded
to compute a passing median. For RL seeds, the associated median is `null` and
the gate fails. A mathematically undefined baseline metric MUST be disclosed
with its reason and omitted only from that metric's comparison; it does not
invalidate an otherwise reconciled baseline or substitute for an RL gate value.

Medians across three seeds reduce reliance on a lucky initialization; they do
not establish statistical significance. Baseline deltas are descriptive unless
a suitable time-series inference method was predeclared. Baselines do not count
as RL seeds; Buy & Hold is not subject to the RL 30-trade requirement.

## 4. The Offline Audit Report - Milestone 1 Deliverable

### 4.1 Artifact structure and identity

Each isolated run directory MUST retain:

```text
config.snapshot.json
manifest.json
scaler_contract.json
feature_schema.json
seeds/<seed>/model + training diagnostics + fills/trades/equity/metrics
baselines/<name>/fills/trades/equity/metrics
audit.json
audit.md
```

These are artifact names, not commands or currently implemented outputs.
The manifest MUST list exact relative artifact paths and their SHA-256 hashes,
including both finalized audit files but excluding the manifest itself.
Audit artifact references MUST NOT include either audit's own hash or a
manifest hash, so generation has no cyclic digest dependency.

`config_sha256` is SHA-256 of the **exact UTF-8 bytes** of
`config.snapshot.json`, captured before validation/training and never rewritten.
All effective settings MUST be explicit in that snapshot; no operational
default or proposal may be injected afterward. Changes require a new snapshot,
hash, and validation. `dataset_sha256` is the verified raw input digest from L3.
Neither hash includes the audit document containing it, avoiding circular hashes.
The contract identity MUST include this specification's version and the SHA-256
of its exact retained document bytes; retain that document snapshot as an
additional manifest-listed artifact.

### 4.2 Required JSON fields

`audit.json` is the machine-readable source of truth. It MUST use strict JSON,
unique keys, finite numbers, UTC report timestamps, and the following structure:

| Field | Required content |
|---|---|
| `audit_version`, `experiment_id`, `run_id`, `generated_at` | Version and unique experiment/run identity |
| `readiness` | Final readiness state, mode, snapshot IDs, L1-L3 check results, warnings, unresolved fields, and errors |
| `run_status` | `NOT_STARTED`, `RUNNING`, `COMPLETE`, or `FAILED`; failure phase/reason when relevant |
| `config_sha256`, `dataset_sha256` | Exact verified configuration and dataset identities; `null` plus an explicit identity error if unavailable |
| `splits` | Exact frozen `train`, `validation`, and `test` start/end instants, timezone, and `[start,end)` convention |
| `partition_evidence` | Actual row counts, first/last scored timestamps, warm-up exclusions, and coverage results per split |
| `data_feature_contract` | Contract version/hash, scaler identity and fit boundaries, feature order, gap-policy outcome |
| `execution_contract` | Instrument units, starting capital, sizing, currency conversion, fill/cost rules, and terminal liquidation convention |
| `training` | Configured/completed seeds, budget, checkpoint-selection rule, selected model hashes, code fingerprint and environment versions |
| `per_seed` | Seed ID, PF/Sharpe/expectancy definitions and values, drawdown, trade count, net return, turnover, exposure, each cost component, final-liquidation costs, metric validity reasons, and evidence paths |
| `aggregate.median_oos_profit_factor` | Finite median PF over every configured seed, otherwise `null` |
| `aggregate.median_oos_sharpe` | Finite median Sharpe over every configured seed, otherwise `null` |
| `aggregate` (additional fields) | Aggregation method, seed count, per-metric validity/reasons, dispersion, and per-seed guard summary |
| `baselines` | Same-period net metrics, accounting evidence, and aligned differences versus each confirmed baseline |
| `decision.pass` | Strict JSON boolean; never `null`, a string, or omitted |
| `decision.label` | Exactly `PASS` when `pass` is true, otherwise `FAIL` |
| `decision.gates`, `decision.reasons` | Thresholds, observed values, gate outcomes, and all blocking/failure reasons |
| `decision.scope` | Exactly `OFFLINE_RESEARCH_ONLY` |
| `artifacts`, `limitations`, `recommendations` | Hashed evidence references, inference limitations, and evidence-linked next steps |

The exact validation split MUST be present even when it differs from the final
OOS test split; recording only percentages or a dataset-wide date range is
insufficient. Values unavailable after refusal/failure remain `null` with
reasons, never fabricated evidence. An initial `RUNNING` report MUST have
`pass: false` and reason `RUN_NOT_COMPLETE`; only finalization may set PASS.

### 4.3 Markdown, finalization, and failure semantics

`audit.md` MUST be rendered from the same structured result as `audit.json` and
include provenance hashes; exact splits; L1-L3 outcomes; per-seed and median
PF/Sharpe; costs and terminal liquidation; baseline comparisons; seed
dispersion; strict PASS/FAIL with reasons; limitations; and recommendations.
It MUST distinguish an execution failure from a completed research rejection.

Audit artifacts MUST be finalized using atomic file replacement. The manifest
MUST remain non-COMPLETE until all required artifacts exist, hashes verify, and
ledger cash/position/equity reconciliation passes within the frozen tolerance.
Preflight errors or runtime exceptions MUST leave a refusal/failure record with
`pass: false`; a failed run MUST NOT generate a success-shaped report. If an
output directory cannot be written, the CLI MUST report the artifact failure
and terminate nonzero rather than claim an audit was persisted.

At no point may PASS cause a broker request, MT5 launch, Telegram send,
`active_config.json` update, or automatic source/deployment change.
This specification introduces no execution functionality and records no
actual passed experiment.

### 4.4 Current implementation boundary

The Milestone 2 workstream's root-level `governance/validator.py` implements a
bounded configuration/hash validator only: required template sections, types,
half-open split ordering, EMA bounds, and physical-file SHA-256. Its readiness
enum is not evidence that the complete contract in this document is implemented.
In particular, OHLCV/calendar/warm-up checks, immutable run snapshots, training,
accounting, and audit finalization remain outside that module. Unknown keys in
its explicit schema are rejected even in draft inspection; conditional
non-applicability is not supported by its initial all-required profile.

Root-level `data/collector.py` separately implements strict quote integrity,
native half-even tick-quantized EMA, RTH gap checking against an explicit
caller-supplied schedule and nominal interval, and separate warm-up/active
frames. The schedule must already encode holidays and session closures;
no exchange calendar is inferred. These helpers do not expand standalone
`ConfigValidator` readiness; their PPO-coordinator integration is described below.

Root-level `engine_rl/envs/trading_env.py` adds a bounded Gymnasium accounting
profile: current-row bid/ask execution, fixed short/flat/long quantity,
cash-settled P&L, and immediate exit-cost-reserved equity. Slippage, commission,
and terminal lambda are explicit price-unit costs; effective fills include
commission exactly once. Forced liquidation uses current mid minus position
sign times half-spread plus slippage, commission, and lambda. Natural final
rows and internal step-limit truncation prohibit new exposure and reconcile
cash against normal realized P&L plus the terminal trade's full P&L with
absolute tolerance 1e-8. Explicit early truncation uses the last consumed
quote. It does not implement margin, swap, FX, training, or full audit artifacts;
external wrappers must not silently abandon open exposure. See README for
the current-row information-timing and truncation contracts.

Root `orchestrator.py` integrates those bounded components with an explicit
`FROZEN_PPO_RESEARCH` profile. It snapshots/hashes inputs, enforces the
caller-supplied calendar and causal splits, fits scaling on train only, and
spawns CPU-capped PPO training processes. Every seed's final checkpoint is
frozen and hashed before any OOS evaluation begins. Deterministic validation
metrics are informational; there is no validation-based checkpoint selection.
The audit retains model identities, health telemetry, fixed EMA/Buy & Hold
baselines, componentwise OOS medians, and per-seed risk/sample guards.
Research PASS is separate from configuration readiness or deployment.
Only `--deploy` permits the subsequent staged MQL5 build and alert gate.
Fallback counts/suspension persist, but recovery/risk changes are directives
only. Tagged fixed-parameter compilation does not export the learned PPO policy
or authorize terminal installation; originals remain unchanged. Neither
synthetic tests nor a clean build complete the real-data research audit required
above. See README for the supported configuration and CLI exit contracts.

Root `optimizer.py` wraps that pipeline in a resumable local SQLite Optuna study.
Sampled learning rate, entropy, batch size, and EMA periods are passed to real
seed workers; trial concurrency is one and numerical threads remain capped.
Median PF is a search objective, not an acceptance gate. Trials explicitly mark
the reused test partition as search-only: numeric metrics and threshold results
remain available, but research PASS and build-gate entry are blocked. The best
sampled parameter dictionary requires a separately frozen fresh-holdout
experiment; it is not an active config.
