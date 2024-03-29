import sys
sys.path.insert(0,'./tsad/tsad')
from utils.evaluating.evaluating import evaluating
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0,'./SKAB/algorithms')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
tf.debugging.experimental.disable_dump_debug_info()
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
from Conv_AE import Conv_AE
import numpy as np
from sklearn.preprocessing import StandardScaler
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
import torch
import pandas as pd
N_STEPS = 60
Q = 0.999 # quantile for upper control limit (UCL) selection
NUMBER_OF_SAMPLES_TO_GENERATE = 10 ## Muiltiple by 60 to get the actual number of samples
NON_ANOMALY = 2 ## Muiltiple by 60 to get the actual number of samples
ANOMALY = 1 ## Muiltiple by 60 to get the actual number of samples
ROUNDS = 2

def get_synth_data_from_syntheizers(NON_ANOMALY_SAMPLES,ANOMALY_SAMPLES,ROUNDS,cols,model_0,model_1):
    anomaly_df = pd.DataFrame()

    for round in range(ROUNDS):
        _,non_anomaly_data = model_0.generate_numpy(NON_ANOMALY_SAMPLES)
        _,anomaly_data = model_1.generate_numpy(ANOMALY_SAMPLES)
        non_anomaly_data = non_anomaly_data.reshape(NON_ANOMALY_SAMPLES*60,8)
        anomaly_data = anomaly_data.reshape(ANOMALY_SAMPLES*60,8)
        non_anomaly_data_df = pd.DataFrame(non_anomaly_data,columns=cols)
        anomaly_data_df = pd.DataFrame(anomaly_data,columns=cols)

        anomaly_df = pd.concat([anomaly_df,pd.concat([non_anomaly_data_df,anomaly_data_df],axis=0)],axis=0)

    return anomaly_df.reset_index().drop('index',axis=1)

def create_sequences(values, time_steps=N_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def main():
  path_to_data = './data/SKAB'
  all_files=[]
  for root, dirs, files in os.walk(path_to_data):
      for file in files:
          if file.endswith(".csv"):
               all_files.append(os.path.join(root, file))

  list_of_df = [pd.read_csv(file,
                            sep=';',
                            index_col='datetime',
                            parse_dates=True) for file in all_files if 'anomaly-free' not in file]
  # anomaly-free df loading
  anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0],
                              sep=';',
                              index_col='datetime',
                              parse_dates=True)


  model_0 = DGAN(DGANConfig(
    max_sequence_len=60,
    sample_len=12,
    batch_size=min(1000, 1000),
    apply_feature_scaling=True,
    apply_example_scaling=False,
    use_attribute_discriminator=False,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=15000,
))
  model_1 = DGAN(DGANConfig(
    max_sequence_len=60,
    sample_len=12,
    batch_size=min(1000, 1000),
    apply_feature_scaling=True,
    apply_example_scaling=False,
    use_attribute_discriminator=False,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=15000,
))
    
  model_0 = model_0.load("models/dual/model_0_15000_valve1.pth",map_location=torch.device('cpu'))
  model_1 = model_0.load("models/dual/model_1_15000_valve1.pth",map_location=torch.device('cpu'))


  model = Conv_AE()
  
  predicted_outlier, predicted_cp = [], []
  for df in list_of_df:
      X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
  
      synth_data = get_synth_data_from_syntheizers(NON_ANOMALY,ANOMALY,ROUNDS,X_train.columns,model_0,model_1)

      StSc = StandardScaler()
      StSc.fit(synth_data)
      synth_data = StSc.transform(synth_data)
      # scaler init and fitting
      StSc = StandardScaler()
      StSc.fit(X_train)
      X_train = StSc.transform(X_train)
  
      # X_train = pd.concat([X_train,synth_data],axis=0)
      X_train = np.concatenate([X_train,synth_data])
      # convert into input/output
      X = create_sequences(X_train, N_STEPS)
  
      # model fitting
      model.fit(X)
  
      # results predicting
      residuals = pd.Series(np.sum(np.mean(np.abs(X - model.predict(X)), axis=1), axis=1))
      UCL = residuals.quantile(Q) * 4/3
  
      # results predicting
      X = create_sequences(StSc.transform(df.drop(['anomaly','changepoint'], axis=1)), N_STEPS)
      cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X - model.predict(X)), axis=1), axis=1))
  
      # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
      anomalous_data = cnn_residuals > UCL
      anomalous_data_indices = []
      for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
          if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
              anomalous_data_indices.append(data_idx)
  
      prediction = pd.Series(data=0, index=df.index)
      prediction.iloc[anomalous_data_indices] = 1
  
      # predicted outliers saving
      predicted_outlier.append(prediction)
  
      # predicted CPs saving
      prediction_cp = abs(prediction.diff())
      prediction_cp[0] = prediction[0]
      predicted_cp.append(prediction_cp)

  true_outlier = [df.anomaly for df in list_of_df]
  binary = evaluating(
    true_outlier,
    predicted_outlier,
    metric='binary'
  )

if __name__ == '__main__':
  main()
