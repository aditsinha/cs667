
#ifndef WRAPPER_H_
#define WRAPPER_H_

typedef struct {
  void* data;
} configuration_t;

typedef struct {
  void* data;
} party_t;

typedef struct {
  void* params;
  void* config;
} model_t;

#ifdef __cplusplus
extern "C" {
#endif

  configuration_t* GetConfiguration(char* filename);
  party_t* GetParty(configuration_t* config, char* data_filename, int is_training);
  int* ComputeGradient(party_t* party, model_t* model);
  int* ComputeNoisyGradient(party_t* party, model_t* model);
  void UpdateModel(model_t* model, int step_num, int* gradient);
  model_t* InitialModel(configuration_t* config);
  int GetIterationCount(configuration_t* config);
  int GetBatchSize(configuration_t* config);
  
  double EvaluateModel(party_t* party, model_t* model);
  
  int GetDataFeatureCount(party_t* party);
  int GetDataRowCount(party_t* party);
  int GetQuantizeBitsPrecision(configuration_t* config);
  double GetNoiseStdDev(configuration_t* config);
  double GetRegularizedRegressionNoise(configuration_t* config);
  double GetRegularization(configuration_t* config);
  
  void QuantizePartyData(party_t* party, long** features, long* labels, int precision);
  model_t* UnquantizeModel(configuration_t* config, int* quantized, int precision);
  model_t* UnquantizeLongModel(configuration_t* config, long* quantized, int precision);
  
  double GetLearningRate(configuration_t* config, int iteration);

#ifdef __cplusplus
}
#endif

#endif
