//+------------------------------------------------------------------+
//| validate_features.mq5                                            |
//| OFFLINE parity dumper. Writes to MQL5\Files:                     |
//|   mql5_bars.csv     - every closed bar used (warm-up included)   |
//|   mql5_features.csv - last InpExportBars rows: raw, z, prediction|
//| Then run: RLBOT_MT5_PARITY_DIR=<dir> pytest tests/test_feature_parity.py
//| Not an EA: places no orders and is not wired to live trading.    |
//+------------------------------------------------------------------+
#property script_show_inputs

#include "../include/FeatureBuilder.mqh"
#include "../include/ONNXAlpha.mqh"

input string InpModelFile    = "xgb_alpha.onnx";        // ONNX model in MQL5\Files
input string InpZScoreFile   = "xgb_alpha_zscore.csv";  // feature,mean,std from Python export
input string InpBarsFile     = "mql5_bars.csv";
input string InpFeaturesFile = "mql5_features.csv";
input int    InpExportBars   = 1000;
input int    InpWarmupBars   = 500;                     // extra history before the export window
input int    InpRsiWindow    = 14;                      // must equal manifest feature_engineer
input int    InpMacdFast     = 12;
input int    InpMacdSlow     = 26;
input int    InpMacdSignal   = 9;
input int    InpAtrWindow    = 14;
input int    InpBBWindow     = 20;
input int    InpPAWindow     = 20;

string Num(const double value) { return StringFormat("%.17g", value); }

bool LoadZScore(const string file_name, double &mean[], double &std[])
  {
   ArrayResize(mean, ALPHA_FEATURE_COUNT);
   ArrayResize(std, ALPHA_FEATURE_COUNT);
   int handle = FileOpen(file_name, FILE_READ | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("validate_features: cannot open %s, error=%d", file_name, GetLastError());
      return false;
     }
   string first = FileReadString(handle);
   StringTrimLeft(first);
   StringTrimRight(first);
   bool ok = (first == "feature,mean,std");
   for(int i = 0; ok && i < ALPHA_FEATURE_COUNT; i++)
     {
      string parts[];
      string line = FileReadString(handle);
      StringTrimLeft(line);
      StringTrimRight(line);
      if(StringSplit(line, ',', parts) != 3 || parts[0] != AlphaFeatureName(i))
        {
         PrintFormat("validate_features: z-score row %d mismatch: '%s'", i, line);
         ok = false;
         break;
        }
      mean[i] = StringToDouble(parts[1]);
      std[i] = StringToDouble(parts[2]);
      if(!MathIsValidNumber(mean[i]) || !MathIsValidNumber(std[i]) || std[i] <= 0.0)
         ok = false;
     }
   FileClose(handle);
   return ok;
  }

void OnStart()
  {
   CFeatureBuilder builder;
   if(!builder.Configure(InpRsiWindow, InpMacdFast, InpMacdSlow, InpMacdSignal,
                         InpAtrWindow, InpBBWindow, InpPAWindow))
     { Print("validate_features: invalid feature windows"); return; }
   if(InpExportBars <= 0 || InpWarmupBars < builder.FirstValidIndex())
     {
      PrintFormat("validate_features: need InpExportBars>0 and InpWarmupBars>=%d", builder.FirstValidIndex());
      return;
     }

   int total = InpExportBars + InpWarmupBars;
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   //--- start=1 skips the still-forming bar; result is oldest-first
   int copied = CopyRates(_Symbol, _Period, 1, total, rates);
   if(copied != total)
     {
      PrintFormat("validate_features: CopyRates returned %d of %d bars, error=%d", copied, total, GetLastError());
      return;
     }
   double high[], low[], close[];
   ArrayResize(high, total); ArrayResize(low, total); ArrayResize(close, total);
   for(int i = 0; i < total; i++)
     {
      if(i > 0 && rates[i].time <= rates[i - 1].time)
        { Print("validate_features: bars are not strictly increasing"); return; }
      high[i] = rates[i].high; low[i] = rates[i].low; close[i] = rates[i].close;
     }

   double features[];
   bool valid[];
   if(!builder.Build(high, low, close, total, features, valid))
     { Print("validate_features: feature build rejected the bar history"); return; }

   double mean[], std[];
   if(!LoadZScore(InpZScoreFile, mean, std))
     { Print("validate_features: z-score file invalid"); return; }

   CONNXAlpha alpha;
   if(!alpha.Init(InpModelFile, ALPHA_FEATURE_COUNT))
      return;

   int bars_handle = FileOpen(InpBarsFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(bars_handle == INVALID_HANDLE)
     { PrintFormat("validate_features: cannot write %s", InpBarsFile); return; }
   FileWriteString(bars_handle, "timestamp,open,high,low,close\n");
   for(int i = 0; i < total; i++)
      FileWriteString(bars_handle, TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS) + "," +
                      Num(rates[i].open) + "," + Num(rates[i].high) + "," +
                      Num(rates[i].low) + "," + Num(rates[i].close) + "\n");
   FileClose(bars_handle);

   int out = FileOpen(InpFeaturesFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(out == INVALID_HANDLE)
     { PrintFormat("validate_features: cannot write %s", InpFeaturesFile); return; }
   string header = "timestamp";
   for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
      header += ",raw_" + AlphaFeatureName(f);
   for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
      header += ",z_" + AlphaFeatureName(f);
   FileWriteString(out, header + ",prediction\n");

   double scaled[];
   float input_vector[];
   ArrayResize(input_vector, ALPHA_FEATURE_COUNT);
   int written = 0;
   for(int i = InpWarmupBars; i < total; i++)
     {
      int offset = i * ALPHA_FEATURE_COUNT;
      if(!valid[i] || !builder.ApplyZScore(features, offset, mean, std, scaled))
        {
         PrintFormat("validate_features: invalid feature row at %s", TimeToString(rates[i].time));
         FileClose(out);
         FileDelete(InpFeaturesFile);
         return;
        }
      for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
         input_vector[f] = (float)scaled[f];
      float prediction;
      if(!alpha.Predict(input_vector, prediction))
        {
         FileClose(out);
         FileDelete(InpFeaturesFile);
         return;
        }
      string line = TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS);
      for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
         line += "," + Num(features[offset + f]);
      for(int f = 0; f < ALPHA_FEATURE_COUNT; f++)
         line += "," + Num(scaled[f]);
      FileWriteString(out, line + "," + Num((double)prediction) + "\n");
      written++;
     }
   FileClose(out);
   PrintFormat("validate_features: wrote %d feature rows and %d bars for %s %s",
               written, total, _Symbol, EnumToString(_Period));
  }
//+------------------------------------------------------------------+
