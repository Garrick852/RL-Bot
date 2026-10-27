//+------------------------------------------------------------------+
//| EA_Template.mq5                                                  |
//| ONNX XGBoost alpha + linear policy head, DECISION-ONLY template. |
//|                                                                  |
//| Pipeline per newly CLOSED bar:                                   |
//|   CopyRates -> OHLCV double arrays -> CFeatureBuilder (8 raw)    |
//|   -> Z-score (training-partition mean/std inputs)                |
//|   -> CONNXAlpha.Predict -> alpha_prediction                      |
//|   -> obs[OBS_DIM] = {alpha, 8 z-features, spread, side, uPnL}    |
//|   -> logits = PPO_WEIGHTS * obs + PPO_BIAS -> argmax -> action   |
//|                                                                  |
//| ISOMORPHISM CONTRACT (engine_rl/linear_policy.py is the source   |
//| of truth; tests/test_ea_template.py enforces parity):            |
//|  - Actions 0/1/2 = SHORT/FLAT/LONG, the TradingEnv Discrete(3)   |
//|    target-position order.                                        |
//|  - obs order = linear_policy.OBSERVATION_NAMES.                  |
//|  - Weights/bias come from export_linear_policy() of a PPO trained|
//|    with policy_architecture="linear_actor" (no hidden actor      |
//|    layers), whose deterministic action is argmax(W*obs + b).     |
//|                                                                  |
//| SAFETY CONTRACT                                                  |
//|  - This template NEVER sends, modifies, or closes orders. The     |
//|    selected action is only logged/shown via Comment().           |
//|  - Placeholder PPO_WEIGHTS are zero and PPO_BIAS favours FLAT,   |
//|    so an unpatched head always resolves to ACTION_FLAT.          |
//|  - Missing ONNX model or unpatched Z-score std -> INIT_FAILED.   |
//+------------------------------------------------------------------+
#property copyright   "RL-Bot offline research"
#property version     "1.00"
#property description "Decision-only ONNX alpha + linear policy head template (no order execution)."
// No `#property strict`: it is an MQL4 directive (MQL5 is always strict) and
// would add a compiler warning that the zero-warning build gate rejects.
// Ship the ONNX model to Strategy Tester agents together with the EA.
#property tester_file "xgb_alpha.onnx"

#include "../include/FeatureBuilder.mqh"
#include "../include/ONNXAlpha.mqh"

//+------------------------------------------------------------------+
//| Policy dimensions                                                |
//+------------------------------------------------------------------+
#define OBS_DIM 12
#define ACTION_DIM 3

//--- Observation layout = engine_rl.linear_policy.OBSERVATION_NAMES.
#define OBS_ALPHA             0   // alpha_prediction of the last closed bar
#define OBS_ZSCORE_BEGIN      1   // 1..8: z_<FEATURE_NAMES> of the last closed bar
#define OBS_SPREAD_FRACTION   9   // spread_fraction = (ask - bid) / bid at decision time
#define OBS_POSITION_SIDE     10  // position_side: -1 short, 0 flat, +1 long
#define OBS_UNREALIZED_PNL    11  // unrealized_pnl_fraction = floating profit / equity

//--- Action semantics = TradingEnv Discrete(3) targets. Decision-only.
#define ACTION_SHORT 0
#define ACTION_FLAT  1
#define ACTION_LONG  2

//+------------------------------------------------------------------+
//| Linear policy head: logits[i] = sum_j W[i][j]*obs[j] + b[i].     |
//| Replaced only by engine_mt5/patcher.py (OPTIMIZED_MATRIX anchors)|
//| with exact ACTION_DIM x OBS_DIM / ACTION_DIM finite literals.    |
//| Keep these initializers comment-free: the patcher refuses others.|
//+------------------------------------------------------------------+
const double PPO_WEIGHTS[ACTION_DIM][OBS_DIM] = // OPTIMIZED_MATRIX:PPO_WEIGHTS
  {
     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  };

//--- Placeholder bias makes the unpatched head choose FLAT (index 1).
const double PPO_BIAS[ACTION_DIM] = // OPTIMIZED_MATRIX:PPO_BIAS
  {0.0, 1.0, 0.0};

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "ONNX alpha model"
input string InpModelFile         = "xgb_alpha.onnx"; // ONNX file in MQL5\Files (or Common\Files)
input bool   InpModelCommonFolder = false;            // true: load from the terminal Common\Files folder
input int    InpWarmupBars        = 500;              // closed bars per evaluation (EWM seed decay)
input long   InpMagic             = 852;              // identifies this strategy's positions

input group "Feature windows (must equal xgb_alpha.manifest.json feature_engineer)"
input int    InpRsiWindow  = 14; // OPTIMIZED:InpRsiWindow
input int    InpMacdFast   = 12; // OPTIMIZED:InpMacdFast
input int    InpMacdSlow   = 26; // OPTIMIZED:InpMacdSlow
input int    InpMacdSignal = 9; // OPTIMIZED:InpMacdSignal
input int    InpAtrWindow  = 14; // OPTIMIZED:InpAtrWindow
input int    InpBBWindow   = 20; // OPTIMIZED:InpBBWindow
input int    InpPAWindow   = 20; // OPTIMIZED:InpPAWindow

//--- MQL5 forbids array-typed inputs, so the Z-score vectors are one
//--- scalar per feature (FEATURE_NAMES order) and are copied into arrays
//--- in OnInit. Values come from the export manifest via
//--- models.onnx_bridge.mql5_zscore_inputs(). Std defaults of 0.0 are
//--- deliberately invalid so an unpatched EA fails INIT (fail-closed).
input group "Alpha Z-score mean (training partition)"
input double InpAlphaMean0 = 0.0; // OPTIMIZED:InpAlphaMean0
input double InpAlphaMean1 = 0.0; // OPTIMIZED:InpAlphaMean1
input double InpAlphaMean2 = 0.0; // OPTIMIZED:InpAlphaMean2
input double InpAlphaMean3 = 0.0; // OPTIMIZED:InpAlphaMean3
input double InpAlphaMean4 = 0.0; // OPTIMIZED:InpAlphaMean4
input double InpAlphaMean5 = 0.0; // OPTIMIZED:InpAlphaMean5
input double InpAlphaMean6 = 0.0; // OPTIMIZED:InpAlphaMean6
input double InpAlphaMean7 = 0.0; // OPTIMIZED:InpAlphaMean7

input group "Alpha Z-score std (training partition, ddof=0, must be > 0)"
input double InpAlphaStd0 = 0.0; // OPTIMIZED:InpAlphaStd0
input double InpAlphaStd1 = 0.0; // OPTIMIZED:InpAlphaStd1
input double InpAlphaStd2 = 0.0; // OPTIMIZED:InpAlphaStd2
input double InpAlphaStd3 = 0.0; // OPTIMIZED:InpAlphaStd3
input double InpAlphaStd4 = 0.0; // OPTIMIZED:InpAlphaStd4
input double InpAlphaStd5 = 0.0; // OPTIMIZED:InpAlphaStd5
input double InpAlphaStd6 = 0.0; // OPTIMIZED:InpAlphaStd6
input double InpAlphaStd7 = 0.0; // OPTIMIZED:InpAlphaStd7

//+------------------------------------------------------------------+
//| Global state                                                     |
//+------------------------------------------------------------------+
CONNXAlpha      g_alpha;
CFeatureBuilder g_builder;
double          g_alpha_mean[ALPHA_FEATURE_COUNT];
double          g_alpha_std[ALPHA_FEATURE_COUNT];
datetime        g_last_closed_bar = 0;
int             g_last_action = ACTION_FLAT;

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
string ActionName(const int action)
  {
   switch(action)
     {
      case ACTION_SHORT: return "SHORT";
      case ACTION_FLAT:  return "FLAT";
      case ACTION_LONG:  return "LONG";
     }
   return "UNKNOWN";
  }

//--- Copy scalar Z-score inputs into arrays and validate them.
bool LoadZScoreInputs(void)
  {
   g_alpha_mean[0] = InpAlphaMean0; g_alpha_std[0] = InpAlphaStd0;
   g_alpha_mean[1] = InpAlphaMean1; g_alpha_std[1] = InpAlphaStd1;
   g_alpha_mean[2] = InpAlphaMean2; g_alpha_std[2] = InpAlphaStd2;
   g_alpha_mean[3] = InpAlphaMean3; g_alpha_std[3] = InpAlphaStd3;
   g_alpha_mean[4] = InpAlphaMean4; g_alpha_std[4] = InpAlphaStd4;
   g_alpha_mean[5] = InpAlphaMean5; g_alpha_std[5] = InpAlphaStd5;
   g_alpha_mean[6] = InpAlphaMean6; g_alpha_std[6] = InpAlphaStd6;
   g_alpha_mean[7] = InpAlphaMean7; g_alpha_std[7] = InpAlphaStd7;
   for(int i = 0; i < ALPHA_FEATURE_COUNT; i++)
     {
      if(!MathIsValidNumber(g_alpha_mean[i]) || !MathIsValidNumber(g_alpha_std[i]) || g_alpha_std[i] <= 0.0)
        {
         PrintFormat("EA_Template: invalid Z-score for %s (mean=%g std=%g); patch from the manifest",
                     AlphaFeatureName(i), g_alpha_mean[i], g_alpha_std[i]);
         return false;
        }
     }
   return true;
  }

//--- Weights and bias must be finite. All-zero weights are allowed (bias-only) but flagged.
bool ValidateWeights(void)
  {
   bool all_zero = true;
   for(int i = 0; i < ACTION_DIM; i++)
     {
      if(!MathIsValidNumber(PPO_BIAS[i]))
        {
         PrintFormat("EA_Template: PPO_BIAS[%d] is not finite", i);
         return false;
        }
      for(int j = 0; j < OBS_DIM; j++)
        {
         if(!MathIsValidNumber(PPO_WEIGHTS[i][j]))
           {
            PrintFormat("EA_Template: PPO_WEIGHTS[%d][%d] is not finite", i, j);
            return false;
           }
         if(PPO_WEIGHTS[i][j] != 0.0)
            all_zero = false;
        }
     }
   if(all_zero)
      Print("EA_Template: PPO_WEIGHTS are the all-zero placeholder; decisions depend on PPO_BIAS only");
   return true;
  }

//--- Extract the most recent `count` CLOSED bars (shift 1 excludes the forming
//--- bar) as chronological (oldest-first) OHLCV double arrays.
bool ExtractOHLCV(const int count, datetime &times[], double &open[], double &high[],
                  double &low[], double &close[], double &volume[])
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, _Period, 1, count, rates);
   if(copied != count)
      return false;
   if(ArrayResize(times, count) != count || ArrayResize(open, count) != count ||
      ArrayResize(high, count) != count || ArrayResize(low, count) != count ||
      ArrayResize(close, count) != count || ArrayResize(volume, count) != count)
      return false;
   for(int i = 0; i < count; i++)
     {
      if(i > 0 && rates[i].time <= rates[i - 1].time)
         return false;
      times[i]  = rates[i].time;
      open[i]   = rates[i].open;
      high[i]   = rates[i].high;
      low[i]    = rates[i].low;
      close[i]  = rates[i].close;
      volume[i] = (double)rates[i].tick_volume; // tick volume; not consumed by the 8 alpha features
     }
   return true;
  }

//--- Account state for this symbol and magic number, mirroring TradingEnv
//--- state_features: position_side = sign(net position); unrealized_pnl_fraction
//--- = floating profit / equity (swap excluded: TradingEnv models no swap).
void FillEnvironmentState(double &obs[])
  {
   double signed_volume = 0.0;
   double floating_pnl = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      double lots = PositionGetDouble(POSITION_VOLUME);
      bool is_long = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      signed_volume += is_long ? lots : -lots;
      floating_pnl += PositionGetDouble(POSITION_PROFIT);
     }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   obs[OBS_POSITION_SIDE] = (signed_volume > 0.0) ? 1.0 : ((signed_volume < 0.0) ? -1.0 : 0.0);
   obs[OBS_UNREALIZED_PNL] = (equity > 0.0) ? floating_pnl / equity : 0.0;
  }

//--- spread_fraction at decision time; false if no valid quote is available.
bool FillSpreadFraction(double &obs[])
  {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick) || tick.bid <= 0.0 || tick.ask < tick.bid)
      return false;
   obs[OBS_SPREAD_FRACTION] = (tick.ask - tick.bid) / tick.bid;
   return true;
  }

//--- logits[i] = sum_j PPO_WEIGHTS[i][j] * obs[j] + PPO_BIAS[i]  (no activation).
bool ComputeLogits(const double &obs[], double &logits[])
  {
   for(int i = 0; i < ACTION_DIM; i++)
     {
      double sum = PPO_BIAS[i];
      for(int j = 0; j < OBS_DIM; j++)
         sum += PPO_WEIGHTS[i][j] * obs[j];
      if(!MathIsValidNumber(sum))
         return false;
      logits[i] = sum;
     }
   return true;
  }

//--- Deterministic argmax; ties resolve to the lowest index, like torch.argmax.
int Argmax(const double &logits[])
  {
   int best = 0;
   for(int i = 1; i < ACTION_DIM; i++)
      if(logits[i] > logits[best])
         best = i;
   return best;
  }

//--- Full closed-bar evaluation. Returns false on any data/model failure.
bool EvaluateClosedBar(double &alpha_prediction, int &action, double &logits[])
  {
   datetime times[];
   double open[], high[], low[], close[], volume[];
   if(!ExtractOHLCV(InpWarmupBars, times, open, high, low, close, volume))
     {
      Print("EA_Template: insufficient or non-chronological closed-bar history");
      return false;
     }

   //--- 1) Raw causal features for every bar; use only the newest closed bar.
   double features[];
   bool valid[];
   if(!g_builder.Build(high, low, close, InpWarmupBars, features, valid))
     {
      Print("EA_Template: CFeatureBuilder rejected the bar history");
      return false;
     }
   int last = InpWarmupBars - 1;
   if(!valid[last])
     {
      Print("EA_Template: newest closed bar has no fully warmed-up feature row");
      return false;
     }

   //--- 2) Z-score in double, then cast once to float32 for the ONNX runtime.
   double scaled[];
   if(!g_builder.ApplyZScore(features, last * ALPHA_FEATURE_COUNT, g_alpha_mean, g_alpha_std, scaled))
     {
      Print("EA_Template: Z-score standardization failed");
      return false;
     }
   float onnx_input[ALPHA_FEATURE_COUNT];
   for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
      onnx_input[f] = (float)scaled[f];

   //--- 3) ONNX alpha prediction.
   float prediction = 0.0f;
   if(!g_alpha.Predict(onnx_input, prediction))
     {
      Print("EA_Template: ONNX alpha prediction failed");
      return false;
     }
   alpha_prediction = (double)prediction;

   //--- 4) Observation vector in linear_policy.OBSERVATION_NAMES order. Market
   //--- values are rounded through float32 exactly like TradingEnv's storage.
   double obs[OBS_DIM];
   ArrayInitialize(obs, 0.0);
   obs[OBS_ALPHA] = alpha_prediction;
   for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
      obs[OBS_ZSCORE_BEGIN + f] = (double)onnx_input[f];
   if(!FillSpreadFraction(obs))
     {
      Print("EA_Template: no valid bid/ask quote for spread_fraction");
      return false;
     }
   obs[OBS_SPREAD_FRACTION] = (double)(float)obs[OBS_SPREAD_FRACTION];
   FillEnvironmentState(obs);

   //--- 5) Linear policy head + argmax.
   if(!ComputeLogits(obs, logits))
     {
      Print("EA_Template: non-finite policy logits");
      return false;
     }
   action = Argmax(logits);
   return true;
  }

//+------------------------------------------------------------------+
//| Expert initialization: every dependency must load (fail-closed). |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!g_builder.Configure(InpRsiWindow, InpMacdFast, InpMacdSlow, InpMacdSignal,
                           InpAtrWindow, InpBBWindow, InpPAWindow))
     {
      Print("EA_Template: invalid feature windows");
      return INIT_PARAMETERS_INCORRECT;
     }
   if(InpWarmupBars <= g_builder.FirstValidIndex())
     {
      PrintFormat("EA_Template: InpWarmupBars must exceed %d", g_builder.FirstValidIndex());
      return INIT_PARAMETERS_INCORRECT;
     }
   if(!LoadZScoreInputs() || !ValidateWeights())
      return INIT_FAILED;
   //--- ONNX model is mandatory: no model, no EA.
   if(!g_alpha.Init(InpModelFile, ALPHA_FEATURE_COUNT, InpModelCommonFolder))
     {
      PrintFormat("EA_Template: ONNX alpha model '%s' failed to load; refusing to start", InpModelFile);
      return INIT_FAILED;
     }
   g_last_closed_bar = 0;
   g_last_action = ACTION_FLAT;
   PrintFormat("EA_Template: initialized (decision-only, no order execution) %s %s",
               _Symbol, EnumToString(_Period));
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   g_alpha.Release();
   Comment("");
  }

//+------------------------------------------------------------------+
//| Evaluate once per newly closed bar. Decision-only.               |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime closed_bar = iTime(_Symbol, _Period, 1);
   if(closed_bar == 0 || closed_bar == g_last_closed_bar)
      return;

   double logits[ACTION_DIM];
   ArrayInitialize(logits, 0.0);
   double alpha_prediction = 0.0;
   int action = ACTION_FLAT;
   if(!EvaluateClosedBar(alpha_prediction, action, logits))
     {
      //--- Fail-closed: any failure degrades to FLAT (no exposure) for this bar; retried on the next closed bar.
      g_last_closed_bar = closed_bar;
      g_last_action = ACTION_FLAT;
      Comment("RL-Bot alpha: evaluation failed -> FLAT (no orders are ever sent)");
      return;
     }
   g_last_closed_bar = closed_bar;
   g_last_action = action;
   string message = StringFormat(
      "RL-Bot %s bar=%s alpha=%.8g logits=[%.6g, %.6g, %.6g] action=%s (decision-only)",
      _Symbol, TimeToString(closed_bar, TIME_DATE | TIME_MINUTES), alpha_prediction,
      logits[0], logits[1], logits[2], ActionName(action));
   Print(message);
   Comment(message);
  }
//+------------------------------------------------------------------+
