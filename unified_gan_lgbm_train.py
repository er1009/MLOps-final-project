import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

def main():
  #loading the dataset
  all_files=[]
  for dirname,_,filenames in os.walk('/data/skab'):
      for filename in filenames:
          if filename.endswith('csv'):
              all_files.append(f'{dirname}/{filename}')
  all_files.sort()
  
  valve1_dat={file.split('/')[-1]:pd.read_csv(file,sep=';',index_col='datetime',parse_dates=True)
                for file in all_files if 'valve1' in file}
  
  #concatenate data(order in time series by sort_index)
  valve1_data=pd.concat(list(valve1_dat.values()),axis=0).sort_index()
  display(valve1_data.head(5))
  
  #train_pre(valve1_data is dataframe)
  train_pre=valve1_data
  
  #train_pre ⇒ train:valid_pre=7:3
  train_pre_size=len(train_pre)
  train_size=int(train_pre_size*0.7)
  train=train_pre[0:train_size]
  
  
  train_synth_0 = train[train["anomaly"]==0].drop(['anomaly','changepoint'],axis=1)
  train_synth_1 = train[train["anomaly"]==1].drop(['anomaly','changepoint'],axis=1)
  
  x_train_pre=train.drop('anomaly',axis=1)
  x_train=x_train_pre.drop('changepoint',axis=1)
  y_train=train['anomaly'].values
  
  #valid_pre ⇒ valid:test=2:1
  valid_pre_size=train_pre_size-train_size
  valid_size=int(valid_pre_size*0.66)
  valid=train_pre[train_size:train_size+valid_size]
  
  
  x_valid_pre=valid.drop('anomaly',axis=1)
  x_valid=x_valid_pre.drop('changepoint',axis=1)
  y_valid=valid['anomaly'].values
  
  test=train_pre[train_size+valid_size:]
  x_test_pre=test.drop('anomaly',axis=1)
  x_test=x_test_pre.drop('changepoint',axis=1)
  y_test=test['anomaly'].values
  
