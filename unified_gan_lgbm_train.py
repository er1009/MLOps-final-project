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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import random
import torch

def lgb_train_predict(x_train,y_train,x_valid,y_valid,x_test,y_test,params, \
                      test_flag=False):

    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_valid=lgb.Dataset(x_valid,y_valid)
    lgb_test=lgb.Dataset(x_test,y_test)

    model_lgb=lgb.train(params=params,train_set=lgb_train, \
                        valid_sets=[lgb_train,lgb_valid], \
                        )

    if test_flag:
        test_pred=np.zeros((len(y_test),1))
        test_pred[:,0]=np.where(model_lgb.predict(x_test)>=0.5,1,0)
        test_acc=accuracy_score(y_test.reshape(-1,1),test_pred)
        test_f1score=f1_score(y_test.reshape(-1,1),test_pred)
        test_cm=confusion_matrix(y_test.reshape(-1,1),test_pred)

        return test_acc,test_f1score,test_cm,test_pred,model_lgb

    else:
        train_pred=np.zeros((len(y_train),1))
        train_pred[:,0]=np.where(model_lgb.predict(x_train)>=0.5,1,0)
        train_acc=accuracy_score(y_train.reshape(-1,1),train_pred)

        valid_pred=np.zeros((len(y_valid),1))
        valid_pred[:,0]=np.where(model_lgb.predict(x_valid)>=0.5,1,0)
        valid_acc=accuracy_score(y_valid.reshape(-1,1),valid_pred)

        return train_acc,valid_acc


def create_dataset(dataset,look_back):

    data_X=np.zeros((len(dataset)-look_back+1,3))
    j=0

    for i in range(look_back-1,len(dataset)):

        data_pre=dataset[i-look_back+1:i+1,0]

        data_pre_mean=np.mean(data_pre,axis=0)
        data_pre_min=np.min(data_pre,axis=0)
        data_pre_max=np.max(data_pre,axis=0)

        data_X[j,:]=np.array([data_pre_mean,data_pre_min,data_pre_max])
        j+=1

    return np.array(data_X).reshape(-1,3)


def smooth_curve(x):
    #x=1 dimension array
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def main():
  #loading the dataset
  all_files=[]
  for dirname,_,filenames in os.walk('./data/SKAB'):
      for filename in filenames:
          if filename.endswith('csv'):
              all_files.append(f'{dirname}/{filename}')
  all_files.sort()
  
  valve1_dat={file.split('/')[-1]:pd.read_csv(file,sep=';',index_col='datetime',parse_dates=True)
                for file in all_files if 'valve1' in file}
  
  #concatenate data(order in time series by sort_index)
  valve1_data=pd.concat(list(valve1_dat.values()),axis=0).sort_index()
  # display(valve1_data.head(5))
  
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

  features = x_train.to_numpy()
  # Obsevations every second, so 60 * 1 second = 1 minute
  n = features.shape[0] // 60
  features = features[:(n*60),:].reshape(-1, 60, features.shape[1])
  # Shape is now (# examples, # time points, # features)
  # print(features.shape)

  dgan_model = DGAN(DGANConfig(
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

  dgan_model = dgan_model.load('models/unified/model_15000_valve1.pth',map_location=torch.device('cpu'))
  NUMBER_OF_SAMPLES_TO_GENERATE = 100 
  _, synthetic_features = dgan_model.generate_numpy(NUMBER_OF_SAMPLES_TO_GENERATE)
  synthetic_data = pd.DataFrame(synthetic_features.reshape(NUMBER_OF_SAMPLES_TO_GENERATE*60,features.shape[2]),columns=x_train.columns)
  kmeans = KMeans(n_clusters=2,random_state=42).fit(synthetic_data)
  y_pred = kmeans.labels_
  synthetic_data['anomaly'] = y_pred

  train_pre=valve1_data
  #train_pre ⇒ train:valid_pre=7:3
  train_pre_size=len(train_pre)
  train_size=int(train_pre_size*0.7)
  train=train_pre[0:train_size]
  synthetic_data["changepoint"] = 0
  train = pd.concat([train,synthetic_data],axis=0)
  x_train_pre=train.drop('anomaly',axis=1)
  x_train=x_train_pre.drop('changepoint',axis=1)
  y_train=train['anomaly'].values
  #valid_pre ⇒ valid:test=2:1
  valid_pre_size=train_pre_size-train_size
  valid_size=int(valid_pre_size*0.66)
  valid=train_pre[train_size:train_size+valid_size]
  valid_synth_0 = valid[valid["anomaly"]==0]
  valid_synth_1 = valid[valid["anomaly"]==1]
  x_valid_pre=valid.drop('anomaly',axis=1)
  x_valid=x_valid_pre.drop('changepoint',axis=1)
  y_valid=valid['anomaly'].values
  test=train_pre[train_size+valid_size:]
  x_test_pre=test.drop('anomaly',axis=1)
  x_test=x_test_pre.drop('changepoint',axis=1)
  y_test=test['anomaly'].values

  x_train_win=np.zeros_like(x_train.values)
  x_valid_win=np.zeros_like(x_valid.values)
  x_test_win=np.zeros_like(x_test.values)
  
  data_dim=8
  for i in range(0,data_dim):
      x_train_win[:,i]=smooth_curve(x_train.values[:,i].flatten())
      x_valid_win[:,i]=smooth_curve(x_valid.values[:,i].flatten())
      x_test_win[:,i]=smooth_curve(x_test.values[:,i].flatten())

  sc = StandardScaler()
  # Calculate the transform matrix and it is applied to valid and test data
  x_train_std = sc.fit_transform(x_train_win)
  x_valid_std = sc.transform(x_valid_win)
  x_test_std = sc.transform(x_test_win)

  look_back=10
  #Dimension of input data
  data_dim=8
  for i in range(0,data_dim):
      if i==0:
          #train data
          x_train_win=create_dataset(x_train_std[:,i].reshape(-1,1),look_back)
          #valid data
          x_valid_win=create_dataset(x_valid_std[:,i].reshape(-1,1),look_back)
          #test data
          x_test_win=create_dataset(x_test_std[:,i].reshape(-1,1),look_back)
      else:
          #train data
          x_train_win=np.concatenate([x_train_win,create_dataset( \
                                    x_train_std[:,i].reshape(-1,1),look_back)],axis=-1)
          #valid data
          x_valid_win=np.concatenate([x_valid_win,create_dataset( \
                                    x_valid_std[:,i].reshape(-1,1),look_back)],axis=-1)
          #test data
          x_test_win=np.concatenate([x_test_win,create_dataset( \
                                    x_test_std[:,i].reshape(-1,1),look_back)],axis=-1)
  
  #change the shape of data
  train_x_win=x_train_win.reshape(-1,3*data_dim)
  train_y=y_train[look_back-1:]
  
  valid_x_win=x_valid_win.reshape(-1,3*data_dim)
  valid_y=y_valid[look_back-1:]
  
  test_x_win=x_test_win.reshape(-1,3*data_dim)
  test_y=y_test[look_back-1:]
  
  #change data type of _x_win from ndarray into dataframe to calculate the importance of characteristic.
  features=['A1_mean','A1_min','A1_max', \
            'A2_mean','A2_min','A2_max', \
            'Cur_mean','Cur_min','Cur_max', \
            'Pre_mean','Pre_min','Pre_max', \
            'Temp_mean','Temp_min','Temp_max', \
            'Ther_mean','Ther_min','Ther_max', \
            'Vol_mean','Vol_min','Vol_max', \
            'Flow_mean','Flow_min','Flow_max']
  
  train_x=pd.DataFrame(train_x_win,columns=features)
  valid_x=pd.DataFrame(valid_x_win,columns=features)
  test_x=pd.DataFrame(test_x_win,columns=features)

  optimization_trial = 100

  results_val_acc = {}
  results_train_acc= {}
  
  for _ in range(optimization_trial):
      # =====the searching area of hyper parameter =====
      lr = 10 ** np.random.uniform(-3, 0)
      min_data_in_leaf=np.random.choice(range(1,21),1)[0]
      max_depth=np.random.choice(range(3,31),1)[0]
      num_leaves=np.random.choice(range(20,41),1)[0]
      # ================================================
  
      #Hyper parameter
      lgb_params={'objective':'binary',
                  'metric':'binary_error',
                  'force_row_wise':True,
                  'seed':0,
                  'learning_rate':lr,
                  'min_data_in_leaf':min_data_in_leaf,
                  'max_depth':max_depth,
                  'num_leaves':num_leaves,
                  'verbosity':0,
                  'early_stopping_round':20
                 }
  
  
      train_acc,valid_acc=lgb_train_predict(train_x,train_y,valid_x,valid_y,test_x,test_y,params=lgb_params,test_flag=False)
      print('optimization'+str(len(results_val_acc)+1))
      print("train acc:" + str(train_acc) + "valid acc:" + str(valid_acc) + " | lr:" + str(lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + \
            ",max_depth:" + str(max_depth) + ",num_leaves:" + str(num_leaves))
      key = " lr:" + str(lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + ", max_depth:" + str(max_depth) + ",num_leaves:" + str(num_leaves)
      results_val_acc[key] = valid_acc
      results_train_acc[key] = train_acc

  lgb_params={'objective':'binary',
            'metric':'binary_error',
            'force_row_wise':True,
            'seed':0,
            'learning_rate':0.0424127,
            'min_data_in_leaf':15,
            'max_depth':24,
            'num_leaves':29
           }

  test_acc,test_f1score,test_cm,test_pred,model_lgb=lgb_train_predict(train_x,train_y,valid_x,valid_y,test_x,test_y,params=lgb_params,test_flag=True)
  model_lgb.save_model("trained_models/model_lgbm_unifed.txt")
  print('test_acc:' + str(test_acc))
  print('test_f1score:' + str(test_f1score))
  print('test_confusionMatrix')
  display(test_cm)

if __name__ == '__main__':
  main()
  
