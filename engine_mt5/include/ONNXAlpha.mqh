//+------------------------------------------------------------------+
//| ONNXAlpha.mqh                                                    |
//| Offline ONNX wrapper for the exported XGBoost alpha model.       |
//| Input : float32 [1, feature_count] Z-scored features.            |
//| Output: float32 [1, 1] forward-return prediction.                |
//| NOT wired to any EA; used only by scripts/validate_features.mq5. |
//+------------------------------------------------------------------+
#ifndef RLBOT_ONNX_ALPHA_MQH
#define RLBOT_ONNX_ALPHA_MQH

class CONNXAlpha
  {
private:
   long              m_handle;
   int               m_feature_count;

public:
                     CONNXAlpha(void) : m_handle(INVALID_HANDLE), m_feature_count(0) {}
                    ~CONNXAlpha(void) { Release(); }

   bool              IsReady(void) const { return m_handle != INVALID_HANDLE && m_feature_count > 0; }

   void              Release(void)
     {
      if(m_handle != INVALID_HANDLE)
         OnnxRelease(m_handle);
      m_handle = INVALID_HANDLE;
      m_feature_count = 0;
     }

   //--- Load <MQL5\Files\file_name> (or the common Files folder) and pin shapes.
   bool              Init(const string file_name, const int feature_count, const bool common_folder = false)
     {
      Release();
      if(feature_count <= 0)
        {
         PrintFormat("CONNXAlpha: invalid feature_count=%d", feature_count);
         return false;
        }
      uint flags = ONNX_DEFAULT;
      if(common_folder)
         flags |= ONNX_COMMON_FOLDER;
      ResetLastError();
      long handle = OnnxCreate(file_name, flags);
      if(handle == INVALID_HANDLE)
        {
         PrintFormat("CONNXAlpha: OnnxCreate('%s') failed, error=%d", file_name, GetLastError());
         return false;
        }
      if(OnnxGetInputCount(handle) != 1 || OnnxGetOutputCount(handle) != 1)
        {
         PrintFormat("CONNXAlpha: expected 1 input/1 output, got %I64d/%I64d",
                     OnnxGetInputCount(handle), OnnxGetOutputCount(handle));
         OnnxRelease(handle);
         return false;
        }
      ulong input_shape[2];
      input_shape[0] = 1;
      input_shape[1] = (ulong)feature_count;
      ResetLastError();
      if(!OnnxSetInputShape(handle, 0, input_shape))
        {
         PrintFormat("CONNXAlpha: OnnxSetInputShape failed, error=%d", GetLastError());
         OnnxRelease(handle);
         return false;
        }
      ulong output_shape[2];
      output_shape[0] = 1;
      output_shape[1] = 1;
      ResetLastError();
      if(!OnnxSetOutputShape(handle, 0, output_shape))
        {
         PrintFormat("CONNXAlpha: OnnxSetOutputShape failed, error=%d", GetLastError());
         OnnxRelease(handle);
         return false;
        }
      m_handle = handle;
      m_feature_count = feature_count;
      return true;
     }

   //--- features: exactly feature_count Z-scored float32 values in FEATURE_NAMES order.
   bool              Predict(const float &features[], float &prediction)
     {
      prediction = 0.0f;
      if(!IsReady() || ArraySize(features) != m_feature_count)
         return false;
      matrixf input_data(1, m_feature_count);
      for(int i = 0; i < m_feature_count; i++)
        {
         if(!MathIsValidNumber(features[i]))
            return false;
         input_data[0][i] = features[i];
        }
      matrixf output_data(1, 1);
      ResetLastError();
      if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, input_data, output_data))
        {
         PrintFormat("CONNXAlpha: OnnxRun failed, error=%d", GetLastError());
         return false;
        }
      prediction = output_data[0][0];
      return MathIsValidNumber(prediction);
     }
  };

#endif // RLBOT_ONNX_ALPHA_MQH
//+------------------------------------------------------------------+
