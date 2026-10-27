//+------------------------------------------------------------------+
//| FeatureBuilder.mqh                                               |
//| Offline parity port of data/feature_engineer.py                  |
//| (TabularFeatureEngineer.generate_features). NOT wired to any EA. |
//|                                                                  |
//| Numerical contract (pandas semantics, all math in double):       |
//|  - EWM(adjust=False): seeded with the FIRST non-NaN observation  |
//|    (not an SMA); update w=((1-a)*w + a*x)/((1-a)+a); a row is    |
//|    valid once nobs >= min_periods.                               |
//|  - RSI : Wilder a=1/rsi_window on gains/losses of close.diff();  |
//|          first change is at bar 1, valid from bar rsi_window.    |
//|          avg_loss==0 -> 100 (gain>0) or 50 (gain==0).            |
//|  - MACD: EMA a=2/(N+1) seeded at close[0]; valid from slow-1.    |
//|          signal = EMA(macd) seeded at first valid MACD.          |
//|  - ATR : Wilder a=1/atr_window over TR; TR[0]=high-low.          |
//|  - BBW : 4*pop_std(close,N)/SMA(close,N)  (ddof=0).              |
//|  - Distances: (close-maxHigh_N)/maxHigh_N, (close-minLow_N)/...  |
//| Built-in iRSI/iMACD/iATR/iBands are deliberately NOT used: their |
//| seeding (SMA seeds, sample std, applied-price rules) differs.    |
//+------------------------------------------------------------------+
#ifndef RLBOT_FEATURE_BUILDER_MQH
#define RLBOT_FEATURE_BUILDER_MQH

#define ALPHA_FEATURE_COUNT 8

enum ENUM_ALPHA_FEATURE
  {
   ALPHA_RSI = 0,
   ALPHA_MACD = 1,
   ALPHA_MACD_SIGNAL = 2,
   ALPHA_MACD_HISTOGRAM = 3,
   ALPHA_ATR = 4,
   ALPHA_BOLLINGER_WIDTH = 5,
   ALPHA_DIST_ROLLING_HIGH = 6,
   ALPHA_DIST_ROLLING_LOW = 7
  };

string AlphaFeatureName(const int index)
  {
   static const string names[ALPHA_FEATURE_COUNT] =
     {
      "rsi", "macd", "macd_signal", "macd_histogram", "atr",
      "bollinger_width", "distance_from_rolling_high", "distance_from_rolling_low"
     };
   if(index < 0 || index >= ALPHA_FEATURE_COUNT)
      return "";
   return names[index];
  }

//--- pandas ewm(adjust=False, ignore_na=False) for series with only leading NaNs
struct SPandasEwm
  {
   double            alpha;
   int               min_periods;
   int               nobs;
   double            value;

   void              Init(const double a, const int min_obs) { alpha = a; min_periods = min_obs; nobs = 0; value = 0.0; }
   bool              Valid(void) const { return nobs >= min_periods; }
   void              Update(const double x)
     {
      if(nobs == 0)
         value = x;
      else
        {
         double old_wt = 1.0 - alpha;
         if(value != x)
            value = (old_wt * value + alpha * x) / (old_wt + alpha);
        }
      nobs++;
     }
  };

class CFeatureBuilder
  {
private:
   int               m_rsi_window;
   int               m_macd_fast;
   int               m_macd_slow;
   int               m_macd_signal;
   int               m_atr_window;
   int               m_bb_window;
   int               m_pa_window;

public:
                     CFeatureBuilder(void)
     {
      m_rsi_window = 14; m_macd_fast = 12; m_macd_slow = 26; m_macd_signal = 9;
      m_atr_window = 14; m_bb_window = 20; m_pa_window = 20;
     }

   bool              Configure(const int rsi_window, const int macd_fast, const int macd_slow,
                               const int macd_signal, const int atr_window,
                               const int bb_window, const int pa_window)
     {
      if(rsi_window < 1 || macd_fast < 1 || macd_slow <= macd_fast || macd_signal < 1 ||
         atr_window < 1 || bb_window < 1 || pa_window < 1)
         return false;
      m_rsi_window = rsi_window; m_macd_fast = macd_fast; m_macd_slow = macd_slow;
      m_macd_signal = macd_signal; m_atr_window = atr_window;
      m_bb_window = bb_window; m_pa_window = pa_window;
      return true;
     }

   //--- first bar index (oldest = 0) at which every feature is defined
   int               FirstValidIndex(void) const
     {
      int first = m_rsi_window;
      first = MathMax(first, (m_macd_slow - 1) + (m_macd_signal - 1));
      first = MathMax(first, m_atr_window - 1);
      first = MathMax(first, m_bb_window - 1);
      first = MathMax(first, m_pa_window - 1);
      return first;
     }

   //--- Build raw features for chronological (oldest-first) closed bars.
   //--- features is resized to total*ALPHA_FEATURE_COUNT, row-major.
   bool              Build(const double &high[], const double &low[], const double &close[],
                           const int total, double &features[], bool &valid[])
     {
      if(total <= 0 || ArraySize(high) < total || ArraySize(low) < total || ArraySize(close) < total)
         return false;
      if(ArrayResize(features, total * ALPHA_FEATURE_COUNT) < 0 || ArrayResize(valid, total) < 0)
         return false;
      ArrayInitialize(features, EMPTY_VALUE);
      ArrayInitialize(valid, false);

      SPandasEwm gain_ewm, loss_ewm, fast_ewm, slow_ewm, signal_ewm, atr_ewm;
      gain_ewm.Init(1.0 / m_rsi_window, m_rsi_window);
      loss_ewm.Init(1.0 / m_rsi_window, m_rsi_window);
      fast_ewm.Init(2.0 / (m_macd_fast + 1.0), m_macd_fast);
      slow_ewm.Init(2.0 / (m_macd_slow + 1.0), m_macd_slow);
      signal_ewm.Init(2.0 / (m_macd_signal + 1.0), m_macd_signal);
      atr_ewm.Init(1.0 / m_atr_window, m_atr_window);
      int first_valid = FirstValidIndex();

      for(int i = 0; i < total; i++)
        {
         double h = high[i];
         double l = low[i];
         double c = close[i];
         if(!MathIsValidNumber(h) || !MathIsValidNumber(l) || !MathIsValidNumber(c) ||
            c <= 0.0 || l <= 0.0 || h < c || l > c)
            return false;

         //--- RSI (change undefined at bar 0)
         if(i > 0)
           {
            double change = c - close[i - 1];
            gain_ewm.Update(change > 0.0 ? change : 0.0);
            loss_ewm.Update(change < 0.0 ? -change : 0.0);
           }
         //--- MACD: NaN until both EMAs reach min_periods
         fast_ewm.Update(c);
         slow_ewm.Update(c);
         bool macd_ok = fast_ewm.Valid() && slow_ewm.Valid();
         double macd = macd_ok ? fast_ewm.value - slow_ewm.value : 0.0;
         if(macd_ok)
            signal_ewm.Update(macd);
         //--- ATR: pandas max(axis=1) skips the NaN previous close at bar 0
         double tr = h - l;
         if(i > 0)
           {
            double pc = close[i - 1];
            tr = MathMax(tr, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
           }
         atr_ewm.Update(tr);

         if(i < first_valid || !gain_ewm.Valid() || !signal_ewm.Valid() || !atr_ewm.Valid())
            continue;

         double rsi;
         double ag = gain_ewm.value;
         double al = loss_ewm.value;
         if(al == 0.0)
            rsi = (ag > 0.0) ? 100.0 : 50.0;
         else
            rsi = 100.0 - 100.0 / (1.0 + ag / al);

         //--- Bollinger width over [i-N+1, i]
         double sum = 0.0;
         for(int k = i - m_bb_window + 1; k <= i; k++)
            sum += close[k];
         double mean = sum / m_bb_window;
         double ss = 0.0;
         for(int k = i - m_bb_window + 1; k <= i; k++)
            ss += (close[k] - mean) * (close[k] - mean);
         if(mean == 0.0)
            return false;
         double bbw = 4.0 * MathSqrt(ss / m_bb_window) / mean;

         //--- Rolling extremes over [i-N+1, i]
         double hh = high[i];
         double ll = low[i];
         for(int k = i - m_pa_window + 1; k <= i; k++)
           {
            hh = MathMax(hh, high[k]);
            ll = MathMin(ll, low[k]);
           }

         int row = i * ALPHA_FEATURE_COUNT;
         features[row + ALPHA_RSI] = rsi;
         features[row + ALPHA_MACD] = macd;
         features[row + ALPHA_MACD_SIGNAL] = signal_ewm.value;
         features[row + ALPHA_MACD_HISTOGRAM] = macd - signal_ewm.value;
         features[row + ALPHA_ATR] = atr_ewm.value;
         features[row + ALPHA_BOLLINGER_WIDTH] = bbw;
         features[row + ALPHA_DIST_ROLLING_HIGH] = (c - hh) / hh;
         features[row + ALPHA_DIST_ROLLING_LOW] = (c - ll) / ll;
         valid[i] = true;
        }
      return true;
     }

   //--- Requested float32 in-place overload: (x - mean) / std.
   bool              ApplyZScore(float &features[], const float &mean[], const float &std[])
     {
      if(ArraySize(features) != ALPHA_FEATURE_COUNT || ArraySize(mean) != ALPHA_FEATURE_COUNT ||
         ArraySize(std) != ALPHA_FEATURE_COUNT)
         return false;
      for(int i = 0; i < ALPHA_FEATURE_COUNT; i++)
        {
         if(!MathIsValidNumber(std[i]) || std[i] <= 0.0f || !MathIsValidNumber(mean[i]) ||
            !MathIsValidNumber(features[i]))
            return false;
        }
      for(int i = 0; i < ALPHA_FEATURE_COUNT; i++)
         features[i] = (features[i] - mean[i]) / std[i];
      return true;
     }

   //--- Double-precision overload used by the parity dumper (Python computes in float64).
   bool              ApplyZScore(const double &raw[], const int offset, const double &mean[],
                                 const double &std[], double &scaled[])
     {
      if(offset < 0 || ArraySize(raw) < offset + ALPHA_FEATURE_COUNT ||
         ArraySize(mean) != ALPHA_FEATURE_COUNT || ArraySize(std) != ALPHA_FEATURE_COUNT)
         return false;
      if(ArrayResize(scaled, ALPHA_FEATURE_COUNT) != ALPHA_FEATURE_COUNT)
         return false;
      for(int i = 0; i < ALPHA_FEATURE_COUNT; i++)
        {
         double x = raw[offset + i];
         if(!MathIsValidNumber(std[i]) || std[i] <= 0.0 || !MathIsValidNumber(mean[i]) ||
            !MathIsValidNumber(x))
            return false;
         scaled[i] = (x - mean[i]) / std[i];
        }
      return true;
     }
  };

#endif // RLBOT_FEATURE_BUILDER_MQH
//+------------------------------------------------------------------+
