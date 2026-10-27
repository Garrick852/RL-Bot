# RL-Bot: phased implementation architecture

**Current target: Milestone 1 (Offline Single-Strategy Validation).**
**Implementation status: prototype foundation; Milestone 1 is not complete.**
**Document revision: 4.1.0 (milestone plan, not a software release).**

The supplied v4.0.0 PDF describes the end-state vision. This document is the
repository's staged implementation whitepaper. A module, unit test, or generated
report is not evidence of a completed training pipeline, profitable strategy, or
statistically significant improvement. No real-data end-to-end experiment has
been recorded in this repository.

## 1. Milestones and scope

| Stage | Objective | Current status |
|---|---|---|
| Foundation | Independent ingestion, training, evaluation, and governance helpers | Prototype components exist; integration is incomplete |
| Milestone 1 | Reproducible, single-asset, single-strategy offline validation | Current design and implementation target |
| Milestone 2 workstream | Strict configuration and dataset-identity gatekeeper | Bounded validator implemented; does not establish Milestone 1 completion |
| Later research | Walk-forward studies, strategy plugins, distillation, TorchRL training | Deferred |
| Later integration | Verified MT5 optimization, multi-asset scheduling, platform exports | Deferred |
| Execution | Paper/live broker execution and deployment controls | Out of scope; requires separate review |

Milestone 1 must run after local dependencies are installed without downloading
market data, contacting a broker, invoking MT5, sending Telegram messages, or
changing `config/active_config.json`. Passing an offline experiment never
authorizes live deployment.

## 2. Current code versus required integration

| Existing component | Boundary / missing work |
|---|---|
| `data/collector.py` | Local CSV/features and scaler fitting exist; train-only transformation and pipeline provenance must be wired |
| `quant_optimization_bot/engine_rl/envs/trading_env.py` | Existing simplified exposure simulator; not the audited cash-settlement environment |
| Root `engine_rl/envs/trading_env.py` | Current bid/ask execution, reserved exit costs, penalized liquidation, and reconciliation; fixed-size cash-settled profile used by the PPO coordinator |
| `engine_rl/sb3_trainer.py` | One-model PPO wrapper with optional analysis; multi-seed orchestration and frozen model selection are missing |
| `engine_rl/result_analysis.py` | Descriptive rollout/sensitivity reporting; not an audited trade ledger or promotion gate |
| `quant_optimization_bot/governance/validator.py` | Metric checks and chronological split helper; data provenance and the full experiment protocol are caller responsibilities |
| Root `governance/validator.py` | Configuration, chronological splits, EMA bounds, and local hash checks; full Data & Feature Contract checks remain unimplemented |
| Root `data/collector.py` | Strict bid/ask/OHLCV, quantized EMA, caller-scheduled RTH gaps, and isolated warm-up/splits; used by PPO preflight, not the standalone config gate |
| Root `orchestrator.py` | Multi-seed CPU PPO with all-model freeze barrier, deterministic OOS, fixed baselines, strict research gates, and separately opt-in staged build/alerts |
| Root `data/parser_mt5.py` | Explicit-contract MT5 bar conversion, symbol resolution, timezone normalization, and quote validation; not a raw-tick parser or certification of executable quote timing |
| Root `engine_rl/sb3_trainer.py` | CPU-governed bounded PPO profile with per-update health telemetry; explicit rate/entropy/batch overrides reach every training-only worker |
| Root `optimizer.py` | Persistent serial Optuna/TPE search over multi-seed median PF; isolated trial audits and best-parameter exports, never untouched-holdout acceptance or deployment |
| Root `governance/fallback.py` | Pure three-stage fallback directives; no count persistence, automatic recovery, config writes, alert sending, or orchestration integration |
| Root `engine_mt5/patcher.py` | Atomic marker-restricted source edits and strict local compile-log/binary verification; separate staged-copy workflow required, actual MetaEditor execution unverified |
| Root `governance/telegram_notifier.py` | Explicit synchronous escaped Telegram alerts, environment-only credentials, fail-safe logging; not connected to the offline pipeline or verified by real transmissions |
| `orchestrator.py` | Generic process scheduler requiring a supplied runner; no offline end-to-end entry point |
| MT5 and broker adapters | Helpers or read-only seams; not part of Milestone 1 |

Existing per-trade Sharpe helpers are not automatically comparable to an
annualized, bar-return Sharpe. Existing analysis reports must not be relabeled
as Milestone 1 audit evidence.

## 3. Freeze one experiment before implementation runs

Choose and record one asset, one local OHLCV file, one timeframe, and one strategy
specification. EMA crossover is a proposed first baseline, not a confirmed choice.
Neither `US500.cash` nor `ES0905.csv` is assumed to be available or suitable.

The experiment specification must include:

- Dataset path, byte-level SHA-256, source description, symbol, timezone,
  timeframe, timestamp meaning, missing-bar policy, and chronological boundaries.
- Instrument accounting: currency, starting cash, point/contract value, sizing,
  minimum position increment, spread, commission, slippage, and margin/leverage.
  Do not apply stock accounting silently to an index CFD or futures contract.
- Fixed baseline rules, feature names/order, indicator lookbacks, warm-up,
  execution timing, action mapping, and reward definition.
- PPO profile, training budget, seed list, selection rule, split boundaries,
  risk gates, and baseline comparison protocol.

Learning exposure from indicator inputs is not the same as optimizing indicator
parameters. Milestone 1 must state exactly what PPO controls. Fixed indicator
periods must remain fixed unless an explicit parameter-action contract is added.
No unimplemented ICT rules or automatic rule distillation are implied.

## 4. Data isolation and causal execution

1. Validate the local data: finite prices/volume, coherent OHLC ranges, unique
   increasing timestamps, explicit timezone, and sufficient history.
2. Reserve chronological training, validation, and final OOS test partitions.
   Freeze their boundaries in the manifest before training. Never shuffle time.
3. Engineer trailing-only features. Prior bars may supply indicator warm-up to
   later partitions, but future bars and future labels cannot enter an earlier
   observation. Warm-up bars are not scored as test trades.
4. Fit the scaler on training observations only. Save its ordered feature
   contract, fitted statistics, training boundaries, and hash; transform both
   validation and test observations without refitting.
5. Specify decisions after bar close with fills no earlier than the next
   executable price. The final observable bar must not create a trade with no
   future execution opportunity.
6. Use validation only for the predeclared checkpoint/configuration selection
   rule. Freeze all policies and parameters before opening the test partition.

The final test is not a retry queue. If its results influence a strategy revision,
that partition becomes development data; obtain a new untouched holdout before
claiming another blind test.

## 5. Training, accounting, and baselines

The formal [offline orchestrator flow specification](specs/offline-orchestrator-flow.md)
defines readiness transitions, L1-L3 gates, the Data & Feature Contract,
three-seed execution, and the audit artifact contract. Its median-across-seeds
PF/Sharpe decision and per-seed risk/sample/stability guards govern the planned
Milestone 1 OOS decision; this does not implement those gates in existing helpers.

Train at least three independent CPU policies with distinct seeds, initially
`[7, 42, 99]`. Seed Python, NumPy, PyTorch, and environment/action-space RNGs.
Record dependency versions, CPU/thread settings, training budgets, and
determinism settings. Reproducibility is assessed within a recorded environment;
bitwise equality across library versions or hardware is not promised.

Evaluate every frozen seed policy on the same final OOS bars. Do not report only
the best seed. Run Buy & Hold and the selected fixed-rule strategy through the
same accounting engine, period, capital constraints, and transaction-cost model.
Include flat cash as a reference where meaningful.

The shared ledger must record decision/fill timestamps, signed quantity, price,
spread/slippage, commission, realized P&L, cash, exposure, and marked equity.
Account for reversals, partial reductions, flat exits, and terminal liquidation.
All strategies must pay exit costs at the final executable mark; an open trade
must not disappear or become a free final exit.

Report net return, gross profit/loss, profit factor, realized trade count, costs,
turnover, exposure, maximum marked-equity drawdown, and Sharpe on a declared
return sampling frequency. Annualization must use the instrument's stated
calendar; zero variance and no-loss profit factor must be explicitly labeled
undefined/unbounded rather than silently replaced or written as invalid JSON.
Report each seed and both baselines, plus aligned return/cost/drawdown deltas.

Three seeds alone do not establish statistical significance. Any confidence
interval or significance claim must name a predeclared method appropriate for
dependent time-series observations, its assumptions, resampling seed, and sample
size. Otherwise label the comparison descriptive and evidence insufficient.

## 6. Minimal `orchestrator.py` closed-loop design (not implemented)

Preserve existing matrix APIs. Add an explicitly offline entry point and a
sequential coordinator; Milestone 1 does not need a process pool or matrix.
The intended command contract is shown below and is **not runnable yet**:

```powershell
python -m quant_optimization_bot.orchestrator offline --config config\offline_experiment.json --output runs\experiment-001
```

`config\offline_experiment.template.json` now provides a draft specification
template; it is not consumed or validated by the current scheduler. Its
`proposals_not_confirmed` section is descriptive and must never supply implicit
runtime defaults. Unknown inputs remain `null`, `frozen` remains false, and
the asset and dataset are deliberately unselected. The proposed discrete
action indices `0/1/2` map to target positions `-1/0/+1`, not to ambiguous
hold/buy/sell commands.

Copy the template to `config\offline_experiment.json` only to define an
experiment; complete and review applicable fields before freezing it.
Explicitly document non-applicable fields and zero-cost assumptions.
Setting `frozen` to true alone does not prove validity or reproducibility.
The full research CLI and full-contract configuration validator are still planned.
Root `orchestrator.py` provides a bounded multi-seed PPO CLI with strict local
data preflight, frozen-model OOS, and research reports; it does not make the
illustrative full-contract research command above runnable. Optional `--deploy`
permits staged MQL5 builds and notifications, not policy export or live trading.
The Milestone 2 workstream adds a bounded root-level `governance/validator.py`
for configuration structure, split/EMA consistency, and local-file SHA-256
checks; see the README for its explicit limits. This is separate from the
existing package's OOS metric validator and does not implement this closed loop.
Reject missing fields, duplicate seeds, overlapping splits, unsupported accounting,
and existing output directories. No implicit broker connection or fallback data.

Planned execution order:

1. Validate/freeze configuration, hash input data, and create a run manifest.
2. Load local bars, establish splits, compute causal features, fit/save the
   training scaler, and transform all partitions.
3. Build shared simulation/accounting and fixed baselines.
4. Train seed policies sequentially, select on validation under the frozen rule,
   and save all model checkpoints and training diagnostics.
5. Freeze model/config hashes; evaluate each policy and baseline once on test.
6. Calculate comparable metrics, seed stability, gate outcomes, and recommendations.
7. Persist machine-readable evidence and a human-readable audit report; mark the
   run complete only after checking all referenced artifacts.

Failures must leave a manifest with the failed phase and actionable error,
produce a nonzero CLI exit, and never produce a success-shaped audit result.
Do not fabricate missing training/evaluation results.

### 6.1 Configuration validation planning baseline

This subsection records agreed design rules, not an implemented validator or
runtime schema. The experiment itself remains unfrozen and unspecified.

Validation has three layers: structure/types, cross-field consistency, and
local-data/freeze evidence. Draft inspection permits unresolved applicable
values and reports them; it does not permit supplied invalid types, malformed
structures, duplicate seeds, or contradictory values. Unknown fields produce
warnings in draft mode and must not influence computations or defaults;
pre-execution validation rejects unknown fields.

Use the template's actual paths:

| Concern | Paths and rules |
|---|---|
| EMA parameters | `strategy.ema.fast_period` and `strategy.ema.slow_period`: positive integers with fast < slow when EMA is selected |
| Seeds | `training.seeds`: at least three distinct integer seeds for the planned multi-seed experiment |
| Time partitions | `splits.train`, `splits.validation`, `splits.test`: start < end within each half-open interval, train.end <= validation.start, validation.end <= test.start |
| Data identity | `dataset.path` and `dataset.sha256`: readable local file with matching byte-level hash |
| Action mapping | `policy.discrete_action_index_to_target`: complete, unambiguous, supported mapping consistent with sizing and minimum position increments |
| Costs | `costs.*`: explicit values, units, application timing, direction, quantity, and applicable currency-conversion rules |
| Terminal liquidation | `execution.terminal_liquidation_price_rule` and `execution.terminal_liquidation_cost_rule`: distinct from swap rules |

Warm-up may use earlier history across partition boundaries, but never future
data or scored warm-up trades. Legitimate market-calendar gaps are not
automatically data corruption; enforce the declared missing-bar policy.

Unconfirmed choices remain in `proposals_not_confirmed`; corresponding formal
fields remain `null` until confirmed. The proposed `0/1/2` to `-1/0/+1` mapping
becomes a required exact mapping only if that proposal is explicitly adopted.
It is not a universal constraint and must never be copied into formal settings
as an implicit default.

Overall result priority and execution eligibility are separate from individual
evidence results:

| Result | Meaning |
|---|---|
| `INVALID` (highest priority) | A known structure/type/value error, contradiction, or failed required evidence check exists |
| `DRAFT_INCOMPLETE` | No known error, but applicable settings or freeze confirmation remain unresolved |
| `EVIDENCE_UNCHECKED` | Applicable settings are complete and consistent, but required evidence checks have not completed |
| `READY_FOR_OFFLINE_EXECUTION` | All applicable settings and evidence checks pass, no pre-execution blockers remain, and root `frozen` is explicitly true |

List errors, warnings, unresolved fields, and each evidence check's independent
result alongside the overall result. An unchecked file/hash must not conceal a
known config error. Never infer that an unchecked item passed.

`frozen: true` is a declaration, not proof. A freeze declaration with known
missing required settings or contradictions is invalid. A complete declaration
with unchecked evidence remains unverified. If settings and evidence pass but
`frozen` is not true, report pending freeze confirmation and deny execution.
These validation results are distinct from the run lifecycle statuses in
section 7.

Only applicable required fields must be non-null. Zero is an explicit numeric
assumption requiring the appropriate units and justification; it is not a
substitute for "not applicable." Non-applicability requires an explicit,
verifiable representation that the future schema must define, not a fabricated
number or an arbitrary string accepted without validation.

Planning cases:

| Case | Expected outcome |
|---|---|
| Unfrozen template with unresolved dataset, formal strategy/action fields, and costs | `DRAFT_INCOMPLETE`; evidence independently marked unchecked |
| `frozen: true` with missing dataset path or repeated seeds | `INVALID`; no execution |
| EMA fast >= slow, invalid types, or overlapping train/validation boundaries | `INVALID`, even during draft inspection |
| Complete consistent settings with required local-data evidence not yet checked | `EVIDENCE_UNCHECKED`; no execution |
| Settings/evidence pass but freeze confirmation is absent | Pending freeze confirmation (`DRAFT_INCOMPLETE`); no execution |
| All applicable pre-execution checks pass and `frozen: true` | Conditionally `READY_FOR_OFFLINE_EXECUTION`; no real example has yet demonstrated this |

Readiness means permission to run an offline experiment only. It establishes
neither strategy performance, statistical significance, nor deployment
authorization. No validator, CLI, or execution functionality is introduced by
this planning baseline.

## 7. Audit artifacts and decisions

Every run must retain a manifest with status (`RUNNING`, `COMPLETE`, or `FAILED`),
input/config/source revision hashes, dirty-source fingerprint if applicable,
environment/dependency versions, partition boundaries, seed IDs, costs, and
artifact hashes. Record phase transitions and errors without secrets.

Planned artifact set:

- Frozen experiment config, manifest, scaler contract, and feature schema.
- Model and training/validation diagnostics per seed.
- Fill/trade ledger and aligned marked-equity/return series per seed and baseline.
- Strict JSON metrics with explicit undefined-value annotations and seed summary.
- `audit.md`: provenance, costs, performance tables, baseline differences, seed
  stability, limitations, gate reasons, and evidence-linked recommendations.

Keep execution status separate from research outcomes. A completed run may
reject a strategy. Engineering completion does not require profitability or
outperforming Buy & Hold.

The initial research review requires median-across-seeds PF >= 1.25 and median
Sharpe >= 1.0, plus drawdown <= 10%, at least 30 trades, and positive expectancy
for every configured seed (at least three distinct seeds), as defined in the
formal flow specification. These are review thresholds, not statistical proof or live
authorization; declare metric definitions and frequency before applying them.
Do not apply a 30-trade gate to a single-entry Buy & Hold baseline as if it were
a failed training candidate. Failed research gates must retain all evidence and
recommend rejection or further development, not silently retrain on the test set.

## 8. Acceptance and implementation order

1. Freeze the asset/dataset, strategy/action contract, execution/cost model,
   splits, and baseline rules before the first real experiment.
2. Implement/test shared accounting, especially reversals, costs, bankruptcy
   handling, and terminal liquidation; reconcile ledger to equity exactly.
3. Implement train-only scaler transformation and causal split tests, including
   a test that changes in future bars cannot alter past observations.
4. Wire sequential three-seed SB3 training, frozen validation selection, and
   baseline evaluation into the planned offline coordinator.
5. Implement complete audit artifacts, failure reporting, and same-environment
   repeatability checks for ledgers/metrics within a declared tolerance.
6. Demonstrate one real, locally supplied historical-data run, with all seeds
   and baselines, no network/platform dependencies, and no active-config writes.

Synthetic tests are necessary for correctness but do not replace the final
real-data demonstration. Until that evidence exists, Milestone 1 remains open.
