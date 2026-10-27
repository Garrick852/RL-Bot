# RL-Bot

**Current target: Milestone 1 (Offline Single-Strategy Validation).**
This repository is a prototype foundation, not a complete autonomous trading
bot. The full multi-seed RL training/research pipeline is not implemented, and no
real-data end-to-end result has been recorded. See
[the phased architecture and offline coordinator plan](RL-Architecture.md)
for scope, evidence requirements, and acceptance criteria.

## Static planning dashboard

Open `dashboard.html` directly in a browser (no server or package installation
required). It follows the dark dashboard design but shows only manually authored
planning labels: `DRAFT_INCOMPLETE`, validation not run, and data evidence
unchecked. It does not load the config, inspect hardware, execute a validator,
train models, or deploy anything. The page has no JavaScript, external assets,
simulated event logs, or running/success animations. Native disclosure controls
expand unconfirmed proposals; the page does not auto-sync with repository state.

## Interactive offline flow demo

Open [specs/offline-orchestrator-flow.html](specs/offline-orchestrator-flow.html)
directly in a modern browser. This separate, single-file SVG/JavaScript demo
animates L1-L3 gates, three seed swimlanes, and an illustrative audit certificate.
Use **Execute Pipeline**, **Force L3 Hash Mismatch**, and **Reset** to explore
success, fail-closed validation, and input invalidation. Reduced-motion settings
are respected; no server, dependencies, or network access are needed.

All data and performance figures are explicitly synthetic. Web Crypto hashes
identify embedded demo bytes, not repository datasets. Readiness is granted
before training and is distinct from the illustrative research decision.
The [formal specification](specs/offline-orchestrator-flow.md) remains the
implementation contract; this visualization adds no real validator or RL runner.
Run its focused logic checks with `node --test tests\offline-flow.test.cjs`.

## Offline experiment specification template

Use `config\offline_experiment.template.json` to capture the unresolved
Milestone 1 decisions without modifying the multi-asset `master_matrix.json`.
It is a **draft specification, not a runnable configuration**: the offline CLI
and the complete Data & Feature Contract validator are not implemented.
A bounded configuration/hash gatekeeper is now available as described below.

Copy the template to `config\offline_experiment.json` when defining an experiment.
`null` means unknown, not zero, free trading, or a default. Supply explicit
cost values, units, and application formulas, including an explicit zero and
justification when a charge does not apply. Confirm all applicable fields;
document inapplicable fields rather than inventing values.

The `proposals_not_confirmed` section is informational only. EMA crossover,
Buy & Hold, seeds, and action mapping are not frozen decisions. For the proposed
Gymnasium discrete mapping, **action indices** `0/1/2` map to **target positions**
`-1/0/+1` (short/flat/long); a target of zero means flatten, not maintain position.
Instrument sizing still determines the actual units represented by each target.
The template's version tracks the document structure, not runtime schema support.
Setting `frozen` to true is not validation: data hashes, split boundaries,
strategy/cost rules, and the complete execution contract must be verified first.

## Milestone 2 workstream: configuration gatekeeper

`governance/validator.py` provides a standard-library-only `ConfigValidator`
and the four `ValidationState` values. This is distinct from the existing
`quant_optimization_bot.governance.validator` OOS-metrics API, which is unchanged.
Starting this workstream does not establish completion of a real Milestone 1
experiment.

```python
from governance.validator import ConfigValidator, ValidationState

# config is an explicitly supplied configuration mapping; no defaults are filled.
validator = ConfigValidator(config, base_directory=experiment_directory)
state = validator.validate()
if state is ValidationState.INVALID:
    print(validator.errors)
```

All six roots (`dataset`, `strategy`, `costs`, `execution`, `splits`, `frozen`)
must exist. Their nested fields follow the current template; missing/null
settings are unresolved only in an unfrozen draft. The initial profile requires
all nested fields, including EMA settings and explicit swap rules; it does not
yet implement conditional non-applicability. Unknown keys within those six
sections and unknown roots are rejected even for drafts. Other known template
sections are JSON/type-checked but not semantically certified.

Split timestamps require explicit UTC offsets and use half-open intervals;
adjacent boundaries may be equal and gaps are allowed. EMA periods must be
positive integers with fast < slow. Dataset hashes cover raw file bytes and
must match the declared lowercase SHA-256 exactly. Relative paths resolve
against `base_directory` (or the process working directory when omitted).
Every validation clears prior diagnostics/evidence. Failures are logged with
their reason and traceback and return `INVALID`.

`validate(check_evidence=False)` explicitly defers file inspection and cannot
return readiness. With evidence enabled, declared paths/hashes are checked
even for otherwise incomplete drafts. A supplied missing/unreadable file or
hash mismatch is invalid, not incomplete. An absent freeze confirmation still
denies readiness after successful hashing.

**Readiness here covers only this bounded module's checks.** It does not check
OHLCV contents, market calendars, warm-up sufficiency, policy/sizing semantics,
three-seed results, or cost-accounting implementations. File-change detection
during hashing is not an immutable data snapshot for a subsequent runner.
The full orchestrator must enforce the remaining specification and snapshot
identity before execution. No training, backtest, CLI, or broker action is added.
Run targeted tests with `python -m pytest tests\test_config_validator.py -q`.

## Milestone 2 workstream: historical quote collector

`data/collector.py` adds a separate pandas/NumPy `DataCollector`; the existing
`quant_optimization_bot.data.collector` prototype APIs remain unchanged.
Install the optional dependencies with `python -m pip install ".[data]"`.
No broker data collection, RL training, or backtesting is included.

```python
import pandas as pd
from data.collector import DataCollector

# expected_rth is an explicit timezone-aware DatetimeIndex covering the
# complete audit window, including holidays/early closes/session boundaries.
collector = DataCollector(
    expected_rth_timestamps=expected_rth,
    bar_interval=pd.Timedelta("1min"),
)
quotes = collector.load_csv(local_csv_path)
gap_report = collector.check_gap_policy(quotes, max_allowed_gaps=0)
warmup, active = collector.get_split(quotes, start_time, end_time, 3, 26)
context = pd.concat([warmup, active], ignore_index=True)
ema = collector.calculate_ema(context, period=26, tick_size=0.00001)
active_ema = ema.iloc[len(warmup):]  # Never score warm-up rows.
```

Required columns are `timestamp, open, high, low, close, volume, bid, ask`.
Missing/null/non-finite core quotes, nonnumeric prices, negative volume,
crossed bid/ask, incoherent OHLC, duplicate or unordered timestamps, and
timezone-naive timestamps raise `DataIntegrityError`. CSV loading normalizes
aware timestamps to UTC without filling, sorting, padding, or dropping rows.

EMA seeds from the quantized first close unless an explicit previous-bar
`initial_ema` is supplied. Every recursive update uses the previously quantized
EMA and exactly `round(value / tick_size) * tick_size`. This is Python binary
floating-point half-even rounding, not decimal arithmetic or a guarantee of
bitwise agreement with every C++ implementation.

Gap checks require both the nominal bar interval and a complete caller-supplied
RTH timestamp schedule. No instrument/calendar is inferred. Missing runs count
scheduled bars, include leading/trailing missing bars, and reset at declared
session closures (schedule deltas different from the nominal interval).
Off-hours observations cannot replace missing RTH quotes. Missing bars below
or equal to the tolerance remain visible in `GapReport`; exceeding it raises
`PollutedDatasetError`. Schedule omissions are treated as closures, so callers
must establish calendar provenance separately.

Splits are `[start_time, end_time)` with positive integer `k_multiplier` and
`slow_period`. Warm-up is exactly their product in preceding rows; insufficient
history or an empty active partition is rejected. Both returned frames are
independent copies. Quote validation alone does not apply the gap policy:
call `check_gap_policy` explicitly. The PPO coordinator below wires these
helpers together, but the standalone configuration validator remains bounded.
Run checks with `python -m pytest tests\test_data_collector.py -q`.

## Decoupled XGBoost alpha research

Install the optional CPU model dependencies with
`python -m pip install ".[xgb]"`. `data/feature_engineer.py` builds RSI, MACD,
ATR, Bollinger bandwidth, and rolling high/low distances from right-aligned
rolling windows and one-directional EWMs. Outputs use a UTC timestamp index.
The indicator columns use only row `t` and preceding rows; no centered windows
or negative shifts are used for features. `generate_labels(df, horizon)` alone
uses `close.shift(-horizon) / close - 1`, so the target at `t` describes the
forward return ending at `t+horizon`. `align_and_clean` jointly removes
indicator warm-up and label-tail rows.

The feature engineer marks output with a backward-only contract and a
fingerprint over index/feature values. Cleaning verifies and refreshes that
fingerprint; `XGBAlphaModel` rejects uncertified, modified, or unapproved
feature frames. This guardrail detects accidental feature edits/lookahead
columns, while causal correctness still depends on reviewed feature formulas
and timestamp/quote semantics.

```python
from data.feature_engineer import TabularFeatureEngineer
from models.xgb_alpha import XGBAlphaModel

engineer = TabularFeatureEngineer()
X = engineer.generate_features(training_bars)
y = engineer.generate_labels(training_bars, horizon=5)
X, y = engineer.align_and_clean(X, y)

alpha = XGBAlphaModel(
    horizon=5,
    max_moving_average_window=engineer.max_moving_average_window,
    n_splits=5,
)
fold_metrics = alpha.fit_walk_forward(X, y)
artifact = alpha.export()  # reports/xgb_alpha.json
```

`fit_walk_forward` must receive the training partition only. Its expanding
`TimeSeriesSplit` uses a gap of `horizon + max_moving_average_window`; each
validation fold logs MSE, directional accuracy, and Spearman rank correlation.
After fold scoring, the final regressor is refit on the supplied training
partition. JSON export is atomic and reloadable; no RL-policy wiring, OOS
acceptance gate, deployment, or live trading is performed. A separate untouched
holdout remains necessary for acceptance.
Run `python -m pytest tests\test_xgb_alpha.py -q`.

### Offline ONNX/MQL5 feature-parity bridge

Install `python -m pip install ".[xgb,onnx]"`. Train with
`XGBAlphaModel(..., standardize=True)`: each purged fold fits population
(ddof=0) Z-score vectors on its own training rows, and the final model uses
the full training partition. The booster then natively consumes the Z-scored
vector. Scaling is deliberately not folded into the ONNX graph, because
float32 de-scaling flipped tree splits by about 4e-4 in testing.

```python
from models.onnx_bridge import export_xgb_alpha_onnx

manifest = export_xgb_alpha_onnx(alpha, X, onnx_path="reports/xgb_alpha.onnx",
                                 feature_engineer=engineer)
```

The export writes three artifacts:

- `xgb_alpha.onnx`: float32 input named `features` with shape `[N, 8]`, and
  float32 output named `prediction` with shape `[N, 1]`. It uses opset 13,
  `ai.onnx.ml` 1, and IR version 8.
- `xgb_alpha_zscore.csv`: CRLF `feature,mean,std` values for MQL5.
- `xgb_alpha.manifest.json`: the ONNX SHA-256, feature order, Z-score vectors,
  feature windows, and verified XGBoost/ONNX maximum difference. Export fails
  above `1e-6`.

The MQL5 side has three files:

- `engine_mt5/include/ONNXAlpha.mqh`: `CONNXAlpha` handles
  `OnnxCreate` / `INVALID_HANDLE`, pins shapes to `[1,8]` and `[1,1]`, and
  calls `OnnxRun`.
- `engine_mt5/include/FeatureBuilder.mqh`: `CFeatureBuilder` ports the pandas
  formulas in `double`. It intentionally avoids `iRSI`, `iATR`, and similar
  built-ins because their seeding differs. EWMs start from the first
  observation, RSI/ATR use Wilder smoothing (`alpha = 1/N`), and Bollinger
  width uses population standard deviation. Its `ApplyZScore` methods reject
  non-positive standard deviations.
- `engine_mt5/scripts/validate_features.mq5`: copy the ONNX and Z-score CSV
  files into `MQL5\Files`, then run this script on the chart. It skips the
  forming bar and writes `mql5_bars.csv` with all closed bars, including
  warm-up. It also writes `mql5_features.csv` for the last 1000 bars:
  timestamp, 8 raw features, 8 Z-scores, and the prediction.

Copy the manifest, ONNX file, and both dumps into one folder. Then run:

```powershell
$env:RLBOT_MT5_PARITY_DIR = "C:\path\to\parity"
python -m pytest tests\test_feature_parity.py -q -rs
```

Python rebuilds features from the exact bars that MT5 used, so recursive
seeds match. It verifies the ONNX hash, requires every MQL5 timestamp to be
present, and asserts raw values, Z-scores, and predictions with
`numpy.allclose(rtol=0, atol=1e-4)`. MT5 server-clock times are treated as UTC
only for alignment.

Without that variable, the real gate is **skipped**, and a skip is not parity
evidence. The always-on tests prove only that a line-by-line Python
transcription of the MQL5 algorithm matches pandas within `1e-9`; they cannot
show that MetaEditor compiles the `.mqh` files or that MT5 produces identical
numbers. None of these files is wired into an EA or into any trading path.

### Alpha training CLI (`train_alpha.py`)

Train and export the alpha from the **same hash-locked dataset** that the RL
config uses. `alpha_label_end` must find the alpha's last feature row in that
file.

```powershell
python train_alpha.py data\US500_M15.csv --dataset-sha256 <sha> `
  --rl-train-start 2024-01-01T00:00:00+00:00 --horizon 4 --output-dir reports\alpha
```

Only rows strictly before `--rl-train-start` (and optionally at or after
`--alpha-start`) are used. Labels are generated inside that partition, so no
label bar reaches the RL split. Feature windows default to
`TabularFeatureEngineer` values; `--n-splits`, `--seed`, `--n-estimators`,
`--max-depth`, and `--learning-rate` are explicit. The command writes the ONNX
file, manifest, Z-score CSV, and `train_alpha.report.json` (purged
walk-forward OOS metrics plus provenance). It then reloads them through the
orchestrator's `load_alpha_model` and re-checks the leakage barrier. Existing
artifacts are never overwritten, and any failure removes partial output and
exits `1`. Copy the printed `features_config` into the experiment's `features`
section. Fold metrics are research evidence, not acceptance.

## Milestone 2 workstream: current-quote trading environment

`engine_rl/envs/trading_env.py` implements a separate Gymnasium `TradingEnv`.
Install its optional dependencies with `python -m pip install ".[env]"`.
The existing toy environment under `quant_optimization_bot` remains unchanged.

```python
from engine_rl.envs.trading_env import TradingEnv

env = TradingEnv(
    active_df,
    feature_columns=["scaled_signal"],  # Already fitted/scaled without OOS leakage.
    initial_cash=10000.0,
    quantity=1.0,
    point_value=1.0,
    slippage=0.01,
    commission=0.02,
    lambda_term=0.10,
    max_episode_steps=1000,
)
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(2)
```

These numbers are API examples, not frozen trading specifications. Actions
`0/1/2` explicitly target short/flat/long at fixed quantity. Every fill uses
the currently executed active row's bid/ask; the next row is returned only as
an observation, not used to compute the current reward or liquidity.
Features must be known before that row's executable quotes: lag indicators
derived from its later close as appropriate. No close-to-next-fill alignment
is inferred by the environment.

This initial profile uses a cash-settled derivative ledger with no notional
cash transfer, leverage/margin, swap, or FX model. **Slippage, commission, and
lambda_term are price units per traded unit; quantity × point_value converts
P&L into cash.** Effective entry fills include adverse slippage/commission;
normal exit marks also reserve spread, slippage, and commission immediately.
Commission is not deducted a second time. Cash contains realized settlements;
equity is cash plus conservative liquidation P&L.

Forced exits use `mid - sign(position) * (half_spread + slippage + commission
+ lambda_term)`, evaluated algebraically from current bid/ask. Final and
internally truncated steps permit no new entry/reversal. Unrealized P&L becomes
exactly zero after settlement. Reward is the cash-unit change in conservative
equity; lambda reduces terminal reward by `abs(quantity) * point_value *
lambda_term` versus a normal exit, but a favorable price move can still produce
a positive terminal reward.

Ordinary closed-trade P&L and the terminal trade's full net P&L are kept in
separate buckets. Before returning termination/truncation, cash delta must equal
their sum with absolute tolerance `1e-8` and zero relative tolerance, otherwise
`DoubleEntryReconciliationError` is raised. Failed final reconciliation blocks
further steps until reset.

Use the internal `max_episode_steps` limit, or call `env.truncate()` for an
explicit early exit at the last consumed row. A wrapper's external `TimeLimit`
or abandoning/resetting an episode cannot automatically settle open exposure;
reset begins a new independent episode and is not a completed audit.
No training algorithm or broker execution is included, and this environment is
not a broker adapter; the bounded PPO research coordinator below uses it.
Run checks with `python -m pytest tests\test_trading_env.py -q`.

## Offline PPO research coordinator and opt-in build gate

Root `orchestrator.py` now connects the strict validator, quote collector, and
cash-settled environment with **real multi-seed CPU PPO training and frozen-policy
OOS evaluation**. The old package matrix scheduler is unchanged. Install `".[sb3]"`
and run:

```powershell
python orchestrator.py path\to\frozen-ppo-config.json
# Optional isolated output destination:
python orchestrator.py path\to\frozen-ppo-config.json --report-directory reports\experiment-001
```

The existing unresolved template is intentionally refused. No asset, calendar,
cost, period, or seed is implicitly selected. Dataset paths resolve relative to
the configuration file. JSON duplicate keys and non-finite constants are
rejected. Preflight validates L1/L2 before reading quotes, loads a retained raw
byte snapshot, checks the original path with the validator's L3, and verifies
the loaded snapshot digest matches both. Gap, warm-up, feature, and supported
economic-profile checks complete before processes start.

### Supported explicit PPO configuration

Start from the template and fill every required field. This runner supports
only the following profile; other models are refused rather than silently
ignored or converted:

| Settings | Required values |
|---|---|
| Root metadata | `template_version: 1`, `specification_status: "FROZEN_PPO_RESEARCH"`, `frozen: true` |
| Dataset | `format: "csv"`, `timezone: "UTC"`, explicit pandas fixed `timeframe` such as `"1min"`, `price_basis: "bid_ask"`, `timestamp_meaning: "execution_quote_time"`, `missing_bar_policy: "explicit_rth_schedule"` |
| Column mapping | Identity mapping of `timestamp/open/high/low/close/volume/bid/ask` to those same names |
| Strategy features | `name: "fixed_ema_crossover"`, positive fast < slow periods, `ema.initialization_rule: "first_close_quantized"`; these EMAs drive only the fixed crossover baseline, never the PPO observation |
| Features | `observation_contract: "alpha_linear_policy_12x3_v1"`, `ordered_names` equal to the 12 `engine_rl.linear_policy.OBSERVATION_NAMES`, `alpha_manifest_path` + exact `alpha_manifest_sha256`, positive integer `warmup_bars`, `scaler_method: "alpha_manifest_zscore"`, `scaler_fit_partition: "alpha_training_partition"`, `refit_on_validation_or_test: false` |
| Policy | `controls: "target_position"`, `action_space_type: "discrete"`, explicit `{"0": -1, "1": 0, "2": 1}` mapping, `reward_definition: "liquidation_equity_delta"` |
| Training section | `algorithm: "PPO"`, `device: "cpu"`, `threads_per_worker: 1`, at least 3 distinct uint32 integer `seeds`, positive integer `total_timesteps_per_seed`, `checkpoint_selection_rule: "final_checkpoint"` |
| PPO hyperparameters | Six required keys: `n_steps: 2048`, `batch_size`, `learning_rate`, `clip_range: 0.2`, `gae_lambda: 0.95`, `gamma: 0.99`; optional `ent_coef` (legacy profiles retain zero). Learning rate in (0,1], entropy in [0,1], integer batch size one of 64/128/256/512; separate 128x128 policy/value networks stay fixed |
| Instrument | `asset_type: "cash_settled_derivative"`, `position_sizing_rule: "fixed_quantity"`, explicit cash/point value/tick/increment, same quote/account currency, `currency_conversion_rule: "identity"`, `margin_rule: "none"`, `leverage: 1`, `bankruptcy_policy: "report_fail"` |
| Spread | `value: 0`, `unit: "price"`, `application_rule: "embedded_bid_ask"`; actual spread comes from quotes, not an extra zero-spread assumption |
| Slippage/commission | Explicit nonnegative `value`, `unit: "price"`, `application_rule: "per_unit_per_side"` |
| Swap | Both values explicitly zero, `day_count_and_multiplier_rule: "explicit_zero"`; no nonzero swap implementation |
| Evaluation | Explicit positive `annualization_periods`, `return_sampling_frequency` equal to timeframe, `sharpe_definition: "bar_simple_returns_sample_std_zero_rf"`, `undefined_metric_policy: "null_and_fail"`, `deterministic_policy: true`, `baseline_rules: ["buy_and_hold","fixed_ema_crossover"]` |
| Safety | `offline_only: true`; all network/broker/active-config/deployment permission flags false |

`execution` must be exactly:

```json
{
  "decision_timing": "before_current_quote",
  "fill_timing": "current_row",
  "fill_price_rule": "bid_ask",
  "final_bar_entry_policy": "forbid",
  "terminal_liquidation_price_rule": "bid_ask_adverse_lambda",
  "terminal_liquidation_cost_rule": "per_unit_price_costs_once"
}
```

Supply a new root `orchestration` object with exactly these fields (example
numbers are not defaults; the abbreviated schedule must be replaced):

```json
{
  "expected_rth_timestamps": ["2024-01-02T14:30:00Z", "2024-01-02T14:31:00Z"],
  "max_allowed_gaps": 0,
  "k_multiplier": 3,
  "max_workers": 3,
  "quantity": 1,
  "lambda_term": 0.1
}
```

The complete schedule must cover the entire input window, including warm-up.
The runner does not infer sessions or accept a remote calendar.
`evaluation.research_gate_thresholds` requires exactly `min_profit_factor`,
`min_sharpe`, `max_drawdown`, `min_trades`, and `min_expectancy`. The first two
must be at least 1.25 and 1.0; maximum drawdown is a fraction in (0, 0.10];
minimum trades must be at least 30; minimum expectancy must be nonnegative
and every seed's expectancy must strictly exceed it. Stricter values are allowed.
Other template metadata is retained but is not a claim that its proposed
algorithms or additional model-selection rules were executed. Old smoke configs
are rejected, not silently upgraded.

### Execution and report semantics

Every partition receives isolated warm-up of `max(warmup_bars, k × slow)` bars
and is enriched by `engine_rl.alpha_pipeline.attach_alpha`: the hash-verified
alpha manifest's feature windows, its frozen training-partition Z-score (never
refitted on RL data), and the ONNX graph scored in float32 produce
`alpha_prediction` plus 8 Z-scored features (lagged one closed bar) and
`spread_fraction`. Every active row must receive a complete finite observation,
or the run is refused. `TradingEnvAdapter` locks environments to those 10
columns plus `position_side`/`unrealized_pnl_fraction` (12 inputs). The alpha
manifest records its training window and label horizon; the run is refused
unless the last alpha label bar is strictly before the RL train split, so
PPO never trains on in-sample alpha. Lagged tick-quantized EMAs are computed only
for the fixed baseline. All seed workers receive
train/validation frames only during training, never test frames or a mutable
source file. Each seed freezes its predeclared final checkpoint, records
validation metrics without tuning on them, and saves a SHA-256-identified ZIP.
Only after **all seeds finish training** are the frozen models loaded for
deterministic test evaluation. Models are hashed again after OOS.

Spawned `ProcessPoolExecutor` workers are capped by the requested worker limit,
seed count, and available CPU count. Numerical-library threads are limited to
one before worker imports. Each worker seeds PPO, runs all test rows, forces
terminal liquidation, and retains fills, trades, and equity. The requested
training budget may round up to a complete 2048-step rollout; actual steps and
PPO health telemetry are reported. Fixed EMA and Buy & Hold baselines use the
same test frames, execution costs, and terminal accounting.
Missing/failed/undefined seed metrics cannot be dropped from aggregation.

`reports/offline_audit_report.json` is atomically replaced, with an additional
per-run `reports/runs/<id>/audit.json` and config/data snapshots. The report
includes exact raw-byte `config_hash`/`dataset_hash`, split and gap evidence,
alpha provenance (manifest/ONNX SHA-256, training window, label end), per-seed evidence,
componentwise median PF/Sharpe/drawdown, thresholds, and explicit failure
reasons. Undefined metrics are `null` with reasons. **`threshold_pass` and `pass`
require every seed and all frozen research gates; `decision` is PASS or FAIL.**
Per-seed drawdown/trade-count/positive-expectancy safeguards still apply.

CLI exit 0 without `--deploy` means research execution completed and its report was persisted, not
that research passed. Preflight refusal, worker failure, or report-write failure
returns a nonzero exit. Pending worker tasks are cancelled on failure; already
running bounded episodes are drained by the process pool before shutdown.
Research PASS is not evidence of profitability, statistical significance,
policy equivalence in MT5, or authorization to trade live.

### Explicit `--deploy` boundary

Without `--deploy`, there are no compiler/patcher/notifier calls or fallback
journal changes, whether metrics pass or fail. With it, the research audit is
saved **before** the local build gate. Configure an optional root `deployment`
object with exactly these keys:

```json
{
  "source_path": "EA_Template.mq5",
  "source_sha256": "<exact SHA-256 of original source bytes>",
  "metaeditor_path": "C:\\Path\\To\\metaeditor64.exe",
  "parameters": {},
  "parameter_binding": "linear_ppo_policy_with_alpha_zscore_v1",
  "policy_seed": 42
}
```

`policy_seed` must be one of the predeclared training seeds (chosen before
results, not the best OOS seed).

**Holdout air-gap.** Before any lock, journal, patcher, or compiler work the
gate loads `reports\true_holdout_eval.json` (override with `--holdout-report`;
paths resolve against the repository root = cwd) and refuses with `INVALID`
unless: `contract` is `true_holdout_eval_v1`; `accepted`, `evaluated_once`,
`gap_ok`, and `reconciliation_ok` are all `true`; `median_oos >= theta_accept`;
the burn flag named in the audit exists and its dataset/candidate SHAs match the
audit; the audit's alpha manifest/ONNX SHAs equal the alpha being injected; and
`policy_seed` has exactly one `COMPLETE` row whose root-relative `model_path`
hashes to its recorded `model_sha256`. The gate then injects **that
holdout-accepted frozen model**, never the deploy run's freshly retrained model
(SB3 zips are not byte-reproducible, so a retrain could never match the audit).
It re-hashes it before and after loading, proves a linear actor, and extracts
`action_net.weight`/`bias` as `PPO_WEIGHTS[3][12]`/`PPO_BIAS[3]`. The outcome
records `policy_source: holdout_accepted_frozen_model`, the holdout report SHA,
median, threshold, and dataset/candidate SHAs. It re-verifies
the alpha manifest and ONNX hashes against the evaluated run, then injects
`InpAlphaMean0..7`/`InpAlphaStd0..7` and the seven feature-window inputs. Config
`parameters` may only add scalar anchored inputs; matrices, or any name the
gate generates, are refused, so policy and alpha values can only come from
verified artifacts. `InpModelFile` is not patched: the outcome records
`required_onnx_file` and `alpha_onnx_sha256`, which the operator must install.
The deploy run's own research `pass` is still required; its PPO hyperparameters
are not cross-checked against the candidate's trial config.

Paths resolve relative to the experiment config. Run
`python orchestrator.py path\to\frozen-ppo-config.json --deploy`.
The flag explicitly permits local build work and optional Telegram alerts
despite the default offline-only safety profile; it never permits broker
execution, active-config modification, terminal installation, or automatic
promotion. Install `".[telegram]"` for alerts and supply the documented Telegram
environment variables. Missing credentials or network errors do not alter decisions.

Failed research gates persist fallback counts per asset/source under
`reports\fallback`: Stage 1 requests reviewed recalibration on a fresh holdout;
Stage 2 requests retaining verified known-good settings and half reference
risk; Stage 3 latches `SUSPENDED`. These are directives, not automated parameter,
lot-size, or order changes. A passing candidate cannot reset a suspension.
Keep the same report directory to preserve governance history; recovery is an
external, explicitly authorized operation, not an automatic retry.
An exclusive lock rejects concurrent builds for that journal; stale locks
require inspection rather than automatic deletion.

Passing research gates patch and compile a **uniquely staged source beside
the original**, preserving relative includes. Only a clean zero-warning build
resets a non-suspended failure counter. Errors return build status `INVALID`,
delete the staged binary, and attempt an alert without crashing the pipeline.
Original source/binary remain unchanged. Successful status is
`STAGED_BUILD_SUCCEEDED` with `policy_exported: true`, **not live deployment**:
a clean build does not prove equivalent MT5 performance (the EA only logs
decisions). Compile/alert
results are separate from data readiness and research PASS; opt-in CLI runs
exit nonzero unless the staged build succeeds.
Run tests with `python -m pytest tests\test_orchestrator.py tests\test_deployment_gate.py -q`.

## Milestone 3 workstream: MT5 export bridge

`data/parser_mt5.py` provides `MT5DataParser` using the existing `".[data]"`
dependencies. It automatically resolves standard aliases or exact broker symbols
through local `config\broker_symbols.json`; unknown aliases are rejected.
Neither point size nor broker timezone is inferred from a symbol.

```python
from data.parser_mt5 import MT5DataParser

parser = MT5DataParser(
    symbol="US500",
    point_size=0.01,                 # Example only; confirm the broker's point size.
    source_timezone="America/New_York",  # Example only; confirm the export timezone.
    bid_ohlc_and_close_aligned_spread=True,
)
quotes = parser.parse_ohlcv(local_mt5_export)
```

The confirmation is a required source-contract attestation, not a parser
verification: **OHLC must be Bid and `<SPREAD>` must describe the spread at that
same close in broker points.** Standard MT5 bar exports do not universally
guarantee that spread alignment. Do not assert this contract for an unknown,
averaged, minimum, or otherwise non-close-aligned spread series. Broker point
size and instrument tick size are separate specifications.

Supported input is comma/tab/semicolon CSV with UTF-8 (optional BOM) or
BOM-marked UTF-16, containing `<DATE>`, `<TIME>`, `<OPEN>`, `<HIGH>`, `<LOW>`,
`<CLOSE>`, `<TICKVOL>`, and `<SPREAD>`. Optional `<VOL>` is ignored explicitly:
canonical volume always comes from `<TICKVOL>`. TDS files must first use this
same bar layout; raw tick files and other layouts are rejected, not aggregated.
Bid is close, and Ask is exactly `close + spread * point_size` using binary
floating-point arithmetic; no rounding, zero-spread substitution, or synthetic
quote repair occurs.

Broker-local timestamps are localized using the explicit IANA timezone and
converted to UTC; ambiguous/nonexistent DST times fail. Chronology, finite
quotes, OHLCV ranges, bid/ask ordering, and missing values are checked by
`DataCollector.validate_ohlcv_invariants()` before returning the frame.
File errors, malformed/duplicate headers, missing fields, and negative spread
raise `DataIntegrityError`.

The frame retains source and symbol-map hashes, quote-contract metadata, and
resolved symbol in `df.attrs`. **MT5 bar timestamps are not certified executable
quote times.** A close reconstructed at a bar-open timestamp must not be passed
to the current-row environment as if known at that earlier instant. Explicitly
establish bar availability/timestamp semantics before simulation. CSV output
does not preserve attrs: retain this provenance separately and hash the actual
canonical output file for a subsequent experiment. The parser is not
automatically wired into the PPO coordinator; it adds no downloads,
training, or broker execution.
Run tests with `python -m pytest tests\test_parser_mt5.py -q`.

## Milestone 3 workstream: CPU Track A trainer

`engine_rl/sb3_trainer.py` adds `SB3Trainer` for the root cash-settled environment.
Install `".[sb3]"`; this optional extra includes pandas/NumPy required by that
environment. The older package trainer remains unchanged; the root PPO
coordinator calls this trainer in isolated train-only seed workers.

```python
from engine_rl.sb3_trainer import SB3Trainer

# Instantiate before importing numerical libraries when launching a fresh worker.
trainer = SB3Trainer(seed=7, num_threads=1)
model = trainer.train(training_env, total_timesteps=4096)
health = trainer.telemetry
```

Only pass the explicitly isolated training environment; this helper does not
select partitions, fit scalers, evaluate OOS, save models, or grant research
acceptance. By default the CPU-only MlpPolicy uses n_steps=2048, batch_size=256,
learning_rate=3e-4, ent_coef=0, clip_range=0.2, gae_lambda=0.95, gamma=0.99, and separate
128×128 Tanh policy/value networks. Current SB3 expects
`net_arch={"pi": [128, 128], "vf": [128, 128]}`; this is equivalent to the
blueprint's deprecated list-wrapped dictionary. PPO rounds budgets up to whole
2048-step rollouts.
Explicit `hyperparameters=` can vary learning rate, entropy, and divisible
batch size within the bounded profile above. The orchestrator validates and
passes the same values to every seed; actual settings are retained in each
model's training evidence. Other PPO settings remain fixed. The network is
selected by name with `policy_architecture=` (config key
`training.policy_architecture`): `mlp_tanh_128x128` (default, above) or
`linear_actor` (`net_arch={"pi": [], "vf": [128, 128]}`), whose deterministic
action is exactly `argmax(W @ obs + b)` with a 3×obs_dim `W`. Unknown names are
refused.

The trainer sets OpenMP/MKL/OpenBLAS/NumExpr/Accelerate/BLIS environment budgets
before optional imports and calls `torch.set_num_threads()` at initialization
and training. Limits are process-wide and retained. Use a fresh worker for
reliable native-library initialization; already-loaded non-PyTorch BLAS runtimes
may ignore changed environment variables. Thread budgets cannot guarantee safe
temperatures or reserve CPU capacity for an independently launched MT5 sweep.

The custom callback logs `approx_kl`, `clip_fraction`, and `explained_variance`
once per observed optimizer update, including the final update. Missing or
non-finite metrics produce warnings and explicit `None`, never zeros. These
diagnostics support investigation; they are not an automatic collapse detector
or proof of performance. Configure Python INFO logging to see normal records.
Run checks with `python -m pytest tests\test_sb3_trainer.py -q`.

## VPS-scale Optuna search

Root `optimizer.py` wraps the same real multi-seed PPO pipeline, with no
operational mode. Install `".[optimizer]"`, then supply a **complete frozen
experiment**, including verified local data, costs, schedule, and splits:

```powershell
python optimizer.py config\offline_experiment.json --n-trials 20
# Resume the same study; this adds 10 trials, not a total-budget limit:
python optimizer.py config\offline_experiment.json --n-trials 10 --max-workers 1
```

The default input is `config\active_config.json`; the repository's existing
unconfigured/SUSPENDED placeholder is deliberately refused. Nothing invents
an asset, fills missing settings, or modifies that file. Dataset paths in each
deep-copied trial config are resolved against the original config location.

The persistent study uses `sqlite:///reports/optuna_vps_study.db`,
`load_if_exists=True`, maximization, and `TPESampler(seed=852)`. By default its
name is `offline_ppo_vps`; `--study-name` selects a separate study in the same
database. Studies bind to the exact base-config hash, search-space version,
and seed-worker budget; changing these requires a new name. A fixed sampler
seed reproduces fresh serial suggestions, not an exact RNG continuation after
restarting the process or bit-identical native-library training.

| Sample | Search domain |
|---|---|
| PPO learning rate | Log-uniform [0.00001, 0.001] |
| PPO entropy coefficient | Log-uniform [0.0001, 0.03] |
| PPO batch size | Categorical 64, 128, 256, 512 |

Search-space version 2 removed EMA sampling: EMAs now drive only the baseline,
and alpha windows are frozen in the hash-locked manifest (its path is resolved
to absolute for trial copies). Every trial runs all configured
seeds with the manifest Z-score and the normal all-model freeze barrier.
Objective value is the **median Profit Factor**, recomputed from every seed and
checked against the audit. Low finite performance remains useful to TPE.
Failed validation, crashed workers, missing/duplicate seeds, or undefined PF
prune the trial with a logged reason and retained trial metadata.

Trial configs, model ZIPs, telemetry, and audits are isolated beneath
`reports\optuna_trials`. Trials are serial (`n_jobs=1`), seed workers default to
one and never exceed the base config's worker cap, and each worker retains the
single-thread CPU contract. `--max-workers` only raises the requested seed-worker
cap, not trial concurrency. SQLite search is protected by an exclusive local
process lock; after a hard kill, inspect the old process and lock before
manually clearing a stale lock. Unfinished trials/artifacts remain evidence,
not successful results. Thread limits do not replace VPS temperature monitoring.

Each new eligible high-water mark atomically writes a **hash-bound, relocatable
holdout candidate** (`contract: holdout_candidate_v1`, `path_anchor:
repository_root`) to `config\best_candidate_config.json`: the sampled `params`,
`artifacts` (`trial_config`, `evidence` = the persisted `candidate_evidence.json`
audit, `alpha_manifest`, and one `models` entry per predeclared seed), each a
`{path, sha256}` pair whose path is a **POSIX path relative to the repository
root**, and `theta_accept`, which is the predeclared
`research_gate_thresholds.min_profit_factor` (never derived from search results).
Run the optimizer from the repository root; a trial whose artifacts fall outside
it is pruned. A candidate exported on a Linux VPS therefore resolves unchanged in
a Windows clone once `reports\optuna_trials\...` and the alpha artifacts are
copied alongside it. This is not a runnable active configuration. Resume repairs a missing export
from the committed best trial. Trials always call `run_pipeline(..., deploy=False)`;
no build work, notifications, or fallback-journal changes are requested.

**Repeatedly optimizing on the test partition makes it search data.** Trial
configs explicitly mark reused-holdout search, audits retain numeric metrics
and `threshold_pass` but cannot grant research `pass`, and the build gate rejects
search configs. Evaluate the candidate once on a genuinely fresh untouched
holdout with `scripts\holdout_eval.py` before considering acceptance.
Neither a high objective nor the exported candidate authorizes deployment.

### Single-shot true holdout (`scripts\holdout_eval.py`)

```powershell
python scripts\holdout_eval.py --data holdout.csv --sha256 <64-hex> `
  --config config\best_candidate_config.json `
  --holdout-start 2025-01-02T14:30:00+00:00 --holdout-end 2025-03-01T00:00:00+00:00 `
  --rth-schedule holdout_rth.json
```

Order of operations:

1. Hash `--data`; a mismatch raises `ValueError` and burns nothing.
2. Refuse with `HoldoutAlreadyUsedError` if `data\holdout_used.flag` or
   `reports\true_holdout_eval.json` already exists.
3. Preflight without reading holdout contents: candidate contract, then every
   root-relative artifact (paths with `\`, a leading `/`, `:`, or `..` are
   refused; absolute paths embedded in the hashed trial config/evidence are never
   dereferenced): trial config, evidence, each seed's frozen `linear_actor` model
   (SHA must equal the evidence row), and alpha manifest/ONNX
   SHAs; profile validation; `theta_accept` not weaker than `min_profit_factor`;
   `--holdout-start` not before the end of every search split.
4. Create the flag exclusively (`O_EXCL`, read-only). The holdout is now burned.
5. Load the CSV, apply the config gap policy against the `--rth-schedule`
   JSON list, use bars before `--holdout-start` only as alpha warm-up, run
   `attach_alpha`, and evaluate every predeclared seed's frozen policy.

`reports\true_holdout_eval.json` always records `median_oos` (median Profit
Factor across seeds, `null` if undefined), `reconciliation_ok`, `gap_ok`,
`evaluated_once: true`, and `accepted` (`median_oos >= theta_accept` and every
check true), plus dataset, schedule, candidate, and per-seed evidence. The burn
flag, candidate, and per-seed `model_path` are recorded root-relative. Failures
after the burn are recorded, not retried. Exit codes: `0` accepted, `1` rejected
or failed after the burn, `2` refused before the burn (or already burned).
Acceptance alone does not run `--deploy`, but `--deploy` cannot build without it.
`reports\` is gitignored, so commit the audit and the burn flag explicitly to make
the burn repository-wide:

```powershell
git add -f reports\true_holdout_eval.json data\holdout_used.flag
```

## Milestone 3 workstream: adaptive fallback directives

`governance/fallback.py` adds a pure `AdaptiveFallbackStateMachine`, separate
from the older package's callback-based fallback helper:

```python
from governance.fallback import AdaptiveFallbackStateMachine

# Count includes the just-recorded OOS rejection; evaluate does not increment it.
directive = AdaptiveFallbackStateMachine().evaluate(consecutive_failures=2)
```

| Consecutive failures | Directive |
|---|---|
| 0 | `NO_ACTION`; no activation or recovery permission |
| 1 | `RECALIBRATE_AND_RETRY`: reject candidate, request reviewed bounds expansion and the retry entropy profile `ppo_ent_coef: 0.03` |
| 2 | `GRACEFUL_DEGRADATION`: reject candidate, retain verified last-known-good `active_config.json`, request risk size at 50% of that reference |
| >=3 | `SUSPENDED`: reject candidate, block all new entries, request zero new-entry risk, and return a formatted CRITICAL alert payload |

The caller must verify the last-known-good configuration actually exists.
The repository's unconfigured `active_config.json` is not a known-good fallback.
The 50% sizing request is relative to the verified reference, not compounded
on each evaluation. Bound expansion still requires supported parameter-domain
review; no numeric bound factor is invented. The entropy value is a directive,
not an automatic modification of `SB3Trainer`.

Suspension must be latched by the coordinator until separately authorized
recovery; passing a smaller count is not authorization to resume. A false
`block_new_entries` in earlier stages does not override any existing halt or
offline-only restriction. Zero new-entry sizing is not an instruction to close
existing positions. A retry influenced by failed OOS results requires a new
untouched holdout, never tuning and reaccepting on the same test partition.

No files are read/written, parameters accepted, retries launched, positions
resized, or messages sent by this module. Failure-count persistence and
directive execution are deliberately outside its scope. The root orchestrator's
opt-in gate persists counts and suspension but only returns recovery directives.
Tests:
`python -m pytest tests\test_adaptive_fallback.py -q`.

## Milestone 4 workstream: local MQL5 source patching

`engine_mt5/patcher.py` provides the standard-library-only `MQL5Patcher`,
`SourcePatchError`, and `CompilationError`. The previous package's staged
patch/compile helper remains unchanged.

```python
from engine_mt5.patcher import MQL5Patcher

patcher = MQL5Patcher(timeout_seconds=30)
# Use a dedicated staged EA copy, not a live terminal's deployed source.
patcher.patch_source(staged_source, {"InpLeftBars": 12, "Risk": 0.25})
log_path = patcher.compile_ea(metaeditor_path, staged_source)
```

Each requested key must match exactly one real standalone declaration such as
`input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars`. Commented-out declarations,
block-comment contents, strings, missing tags, and duplicate tagged declarations
do not qualify. Supported literals are range-checked MQL integer types, finite
float/double, booleans, and identifier-only enum constants. Arbitrary expressions
and string/date/color injection are refused. A full envelope with a nonempty
`parameters` mapping is also accepted; supplying that envelope is not proof of
OOS approval. All edits are validated before atomic source replacement;
UTF-8/BOM UTF-16 encoding, endian order, and line endings are preserved.

Compilation calls the explicit local `metaeditor64.exe` using `/compile` and
a unique `/log` path with a strict timeout. The source's existing `.ex5` is
removed before compilation to prevent stale-binary acceptance. Acceptance
requires exit code zero, a nonempty new binary, unchanged source bytes, and one
final clean log summary. Both `0 error(s), 0 warning(s)` and native
`Result: 0 errors, 0 warnings, ... msec elapsed` summaries are supported, with
only optional elapsed-time/CPU metadata after counts. Any warning/error
diagnostic, nonzero count, extra summary, or missing/unreadable log fails closed.
Failure deletes the associated `.ex5`; inability to delete it is an explicit
unsafe-binary error, never a swallowed cleanup failure. Logs remain for review.

`verify_compile_log(log_path)` assumes ordinary `EA.log` / `EA.ex5` naming.
For any other log naming, pass `source_path=...` explicitly, as `compile_ea`
does. Source patching and compilation are separate operations: a compile failure
does not roll back an already patched source, so operate on staged copies.
Neither method loads a trading terminal, deploys binaries, contacts a network,
or grants strategy acceptance. Automated tests simulate MetaEditor; no actual
broker terminal compilation has been demonstrated.
Run checks with `python -m pytest tests\test_mql5_patcher.py -q`.

### Matrix anchors and the decision-only `EA_Template.mq5`

A list/tuple value patches a matrix declared as
`const double NAME[ROWS][COLS] = // OPTIMIZED_MATRIX:NAME` followed by a
comment-free `{ {..}, .. };` initializer. Dimensions may be literals or
identifiers resolved from exactly one `#define`. The value must match the
declared shape exactly; ragged rows, non-finite values, booleans, and strings
are refused. A one-dimensional `NAME[N]` declaration takes a flat list. Scalar,
matrix, and vector edits are applied together or not at all.

`engine_mt5/source/EA_Template.mq5` combines the ONNX alpha with a linear policy
head `PPO_WEIGHTS[ACTION_DIM=3][OBS_DIM=12]` plus `PPO_BIAS[3]` and argmax
(lowest index on ties, as `torch.argmax`). `engine_rl/linear_policy.py` is the
shared contract: actions follow `TradingEnv` (`0=SHORT, 1=FLAT, 2=LONG`), and
observations are `alpha_prediction`, the 8 Z-scored features (both lagged one
closed bar), `spread_fraction`, then `TradingEnv(state_features=...)` values
`position_side` and `unrealized_pnl_fraction`. `export_linear_policy(model)`
returns `PPO_WEIGHTS`/`PPO_BIAS` from a `linear_actor` PPO with a 12-dim
observation and refuses hidden layers. Tests prove the patched EA head selects
the same action as `model.predict(deterministic=True)`. The placeholder bias
`{0, 1, 0}` makes an unpatched head choose FLAT. The Z-score is supplied as 16
scalar inputs `InpAlphaMean0..7` / `InpAlphaStd0..7`, because MQL5 forbids array
inputs; use `models.onnx_bridge.mql5_zscore_inputs(manifest)`. The default std
of 0 makes an unpatched EA return `INIT_FAILED`, as does a missing ONNX model or
non-finite weights. `#property strict` (an MQL4 directive that would warn under
the zero-warning gate) is omitted.

Limitations: the EA only logs decisions and never sends orders. The orchestrator
now trains on this 12-dim contract and `--deploy` injects the holdout-accepted head.
Python EWM indicators seed over `warmup_bars` per partition while the EA seeds
over `InpWarmupBars`; use long warm-ups so seeding differences decay.
`TradingEnv` unrealized PnL includes exit slippage
and commission, while MT5 `POSITION_PROFIT` does not; SB3 float32 versus MQL5
double could differ at near-ties.
Deep MLP policies should be exported via ONNX instead of matrix patching.
Includes are relative (`../include/`). This EA has never been compiled with
MetaEditor. Tests: `python -m pytest tests\test_ea_template.py -q`.

## Milestone 4 workstream: explicit Telegram operational alerts

`governance/telegram_notifier.py` adds `TelegramAlertEngine` using `requests`.
Install `".[telegram]"` and supply `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
through your process environment or secret manager. Never put credentials in
configuration files, source, or logs. The class accepts no credential arguments
and reads the environment at send time.

```python
from governance.telegram_notifier import TelegramAlertEngine

alerts = TelegramAlertEngine()
# Explicitly sends a network message; only invoke in an authorized online context.
delivered = alerts.send_circuit_breaker("US500.cash", "Three consecutive OOS failures")
```

Dedicated methods are `send_deployment_success(asset, metrics, param_diff)`,
`send_fallback_alert(stage, reason, active_config)` (stage 1 or 2 only), and
`send_circuit_breaker(asset, reason)`. All return a delivery boolean.
Success metrics require finite `profit_factor` and `sharpe`; parameter changes
are serialized exactly, not truncated. A green success message describes
caller-attested L3/build completion, not proof of live deployment. Orange
fallback messages request recovery/degradation; red alerts declare `SUSPENDED`
and block-new-entry instructions. Notifications themselves execute none of
those actions.

Dynamic text is escaped for Markdown V2, including backslashes. Fallback
messages include only top-level `status`, `strategy_state`, and `lot_size`,
not the full active config. Callers must ensure reason strings, parameter
changes, and selected summary fields contain no confidential data.

Transmission is a single explicit POST to Telegram with `timeout=5`,
redirects disabled, and no retries. Requests' timeout is a connect/read-inactivity
limit, not a guaranteed total wall-clock deadline. Missing credentials,
dependency/transport errors, rejected HTTP/API results, invalid response JSON,
and formatting failures log an error and return false without crashing the
orchestrator. Exception bodies, response text, URLs containing tokens, chat IDs,
and message contents are deliberately not logged. Overlength messages are
refused rather than silently truncating exact changes.

The coordinator invokes this engine only with explicit `--deploy` opt-in;
ordinary offline runs never send notifications. Tests mock every HTTP call; no real Telegram send has
been performed. Run `python -m pytest tests\test_telegram_alerts.py -q`.

CPU-first quantitative optimization and reinforcement-learning scaffold based on
`RL-Architecture.md` (version 4.0.0). The current implementation provides local
CSV feature engineering and drift checks, out-of-sample validation, a three-stage
fallback state machine, safe MQL5 parameter patching/compile verification, and
bounded matrix scheduling. MT5 tester runs also produce a companion
`<report-name>.analysis.md` with parsed metrics, OOS/drift gate state, and ordered
recommendations. SB3 training can optionally evaluate the learned policy on a
separate environment and write an `.rl-analysis.md` report with closed-trade
metrics, action usage, PPO health, and per-indicator action sensitivity.

## Safety and operating mode

This repository does not place live orders. Broker adapters are configuration
readers and integration seams only; connecting a broker or enabling execution
requires a separate, explicit implementation and review. Candidate parameters
must pass the OOS validator before they can be persisted as active configuration.
MT5 source patching is opt-in and compiles a temporary copy before replacing the
original file. Telegram alerts are sent only when explicitly invoked and a bot
token is supplied.

Start with the built-in tests:

```powershell
python -m unittest discover -s tests -v
```

Validate an OOS report using the public API:

```python
from quant_optimization_bot.governance.validator import OOSMetrics, validate

result = validate(OOSMetrics(
    profit_factor=1.4,
    sharpe=1.1,
    max_drawdown_pct=8.0,
    trade_count=42,
))
print(result.passed, result.reasons)
```

Use `split_oos(rows, fraction=0.2)` to reserve the latest chronological rows
before evaluating a candidate; the validator intentionally accepts metrics and
does not infer whether they came from held-out data.
The report analyzer labels tester-only results `UNVERIFIED`; deployment is
eligible only when OOS validation and the KS drift gate are both supplied and
pass. Recommendations are decision support, not a substitute for independent
review or an automatic deployment authorization.
RL reports also treat evaluation as `UNVERIFIED` unless the caller explicitly
marks the evaluation environment as held-out OOS. Indicator sensitivity
describes policy dependence only; it is not causal importance. Multi-seed
stability and drift checks remain required before promotion.

Optional training integrations (Stable-Baselines3, TorchRL, Gymnasium,
MetaTrader5, ib_insync, or cTrader) are not installed or imported by the core.
Each integration reports a clear missing-dependency/configuration error until
its optional requirements and platform-specific setup are provided.
