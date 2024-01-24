# -*- coding: utf-8 -*-
"""cifar10-noise.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VLuRTs1MD93jMnpTXrI74Um5PXjEeWDj
"""

import tensorflow as tf
import torch
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gdown

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, recall_score, f1_score
import random

from sklearn.preprocessing import LabelBinarizer
import time
import pandas as pd

!gdown 1VUbFJiiqWmoBknmhyuY965wFZZ5uI6ax

!unzip cifar10_noise_label

import torch
noise_file = torch.load('./CIFAR-10_human.pt')
clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random1_label = noise_file['random1_label']
random2_label = noise_file['random2_label']
random3_label = noise_file['random3_label']

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test)=cifar10.load_data()

print('Shape of x_train is {}'.format(x_train.shape))
print('Shape of x_test is {}'.format(x_test.shape))
print('Shape of y_train is {}'.format(y_train.shape))
print('Shape of y_test is {}'.format(y_test.shape))

from tensorflow.keras.utils import to_categorical

# Normalizing
x_train=x_train/255
x_test=x_test/255

#One hot encoding
# y_train_cat=to_categorical(y_train,10)
# y_test_cat=to_categorical(y_test,10)


# clean_label_cat = to_categorical(clean_label,10)
# worst_label_cat = to_categorical(worst_label,10)
# aggre_label_cat = to_categorical(aggre_label,10)
# random1_label_cat = to_categorical(random1_label,10)
# random2_label_cat = to_categorical(random2_label,10)
# random3_label_cat = to_categorical(random3_label,10)

cnt_noise = 0
for i in range(0, len(y_train)):
    if random2_label[i] != y_train[i]: cnt_noise += 1
print("train size: ", y_train.shape[0])
print("noise size: ", cnt_noise)

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
  lb = LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return roc_auc_score(y_test, y_pred, average=average)

#evaluate the performance of CNN
def cnn_cifar10_flip(y_train, x_train, x_test, y_test):

  model6 = Sequential()
  model6.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
  model6.add(BatchNormalization())
  model6.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model6.add(BatchNormalization())
  model6.add(MaxPool2D((2, 2)))
  model6.add(Dropout(0.2))
  model6.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model6.add(BatchNormalization())
  model6.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model6.add(BatchNormalization())
  model6.add(MaxPool2D((2, 2)))
  model6.add(Dropout(0.3))
  model6.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model6.add(BatchNormalization())
  model6.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model6.add(BatchNormalization())
  model6.add(MaxPool2D((2, 2)))
  model6.add(Dropout(0.4))
  model6.add(Flatten())
  model6.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model6.add(BatchNormalization())
  model6.add(Dropout(0.5))
  model6.add(Dense(10, activation='softmax'))

  model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Image Data Generator , we are shifting image accross width and height also we are flipping the image horizantally.
  datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,rotation_range=20)
  it_train = datagen.flow(x_train,to_categorical(y_train, 10))
  steps = int(x_train.shape[0] / 64)
  history6=model6.fit_generator(it_train,epochs=200,steps_per_epoch=steps,validation_data=(x_test,to_categorical(y_test, 10)))
  predict_res = model6.predict(x_test)
  y_pred = np.argmax(predict_res,axis=1)

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='micro')
  recall = recall_score(y_test, y_pred, average='micro')
  f1 = f1_score(y_test, y_pred, average='micro')
  auc = multiclass_roc_auc_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  return accuracy, precision, recall, f1, auc, cm, model6


# choose the noise set here
y_train_noisy = random2_label.copy()  # worst, aggr, random1, random2, random3
cur_noise = "random2"   # worst, aggr, random1, random2, random3


y_train_true = clean_label.copy()
exps = 10 # run ten times for each simulation
exp_df = pd.DataFrame()
other_test_res_exp_df = pd.DataFrame()
final_res_cm = dict()
# human_flip_err_dict = dict()
good_user = True
percent_noise_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for order in range(0, exps):
  # shuffle the dataset for each experiment
  train_idx_shuffled = np.arange(len(y_train_noisy))
  np.random.shuffle(train_idx_shuffled)

  y_train_shuffled = y_train_noisy[train_idx_shuffled].copy()
  y_train_true_shuffled = y_train_true[train_idx_shuffled].copy()
  x_train_shuffled = x_train[train_idx_shuffled].copy()


  flipped_indexes = list()
  for i in range(0, y_train_shuffled.shape[0]):
      if y_train_shuffled[i] != y_train_true_shuffled[i]: flipped_indexes.append(i)
  print("===================total amount of noise: ", len(flipped_indexes))

  # Human inspect and correct instance labels based on probability dict 'human_classify_probs'
  record_dict_acc = dict()
  record_dict_recall = dict()
  record_dict_precision = dict()
  record_dict_f1 = dict()
  record_dict_auc = dict()
  record_dict_time_process = dict()
  # cnn_record_dict_cm = dict()



  # generate probability dict of user correction if each order is performed by different users
  # human_flip_err_dict[order] = human_classify_probs


  for percent_noise in percent_noise_list:
    y_train_flip_shuffled = y_train_shuffled.copy()


    num_relabel = round(percent_noise * len(flipped_indexes))


    acc_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()
    auc_list = list()
    process_time_list = list()

    if percent_noise == 0: # No need for flipping labels
      start_time = time.time()

    # i = 0, means model-only, used machine learning model directly training with the original dataset
    if percent_noise != 0:

      flip_back_indexes = random.sample(flipped_indexes, num_relabel)  # the noisy label which will be flipped this round



      start_time = time.time()


      if good_user:  # GOOD USER 100% flip labels correctly

        start_time = time.time()
        for j in flip_back_indexes:
          cur_label = y_train_flip_shuffled[j]
          true_label = y_train_true_shuffled[j]
          if cur_label == true_label:
            print("contaminate fail index", j)
          y_train_flip_shuffled[j] = true_label


    # machine learning models learn the data
    print("==================current exp order: ", order)
    print("=====================current percent noise cleaning: ", percent_noise)
    acc, precision, recall, f1, auc, cm, cnn = cnn_cifar10_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'fashionMNIST') and (model == 'cnn'): acc, precision, recall, f1, auc, cm, cnn = cnn_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test, img_shape, batch_size=128, epochs=50)
    # if (dataset == 'fashionMNIST') and (model == 'svc'): acc, precision, recall, f1, auc, cm, svc = svc_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'fashionMNIST') and (model == 'lr'): acc, precision, recall, f1, auc, cm, lr = lr_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'fashionMNIST') and (model == 'dt'): acc, precision, recall, f1, auc, cm, dt = dt_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'fashionMNIST') and (model == 'rf'): acc, precision, recall, f1, auc, cm, rf = rf_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)


    # if (dataset == 'agnews_10pct') and (model == 'sgd'): acc, precision, recall, f1, auc, cm, sgd = sgd_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'agnews_10pct') and (model == 'mn'): acc, precision, recall, f1, auc, cm, mn = mn_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
    # if (dataset == 'agnews_10pct') and (model == 'lstm'): acc, precision, recall, f1, auc, cm, lstm = lstm_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test, batch_size=256, epochs=20)

    end_time = time.time()

    #### performance metricslr
    acc_list.append(acc)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_list.append(f1)
    auc_list.append(auc)

    #### time
    process_time_list.append(end_time - start_time)
    # res_error_probs = res_err_probs(cm)
    # for k1, v1 in res_error_probs.items():
    #   for k2, v2 in res_error_probs[k1].items():
    #     new_key = str(k1) + '-' + str(k2) + '(%)'
    #     if new_key not in cnn_record_dict_cm.keys():
    #       cnn_record_dict_cm[new_key] = list()
    #     cnn_record_dict_cm[new_key].append(v2)


    record_dict_acc[percent_noise] = acc_list
    record_dict_recall[percent_noise] = recall_list
    record_dict_precision[percent_noise] = precision_list
    record_dict_f1[percent_noise] = f1_list
    record_dict_auc[percent_noise] = auc_list

    record_dict_time_process[percent_noise] = process_time_list   # time should be accumulative, so edit time in the dataframe later

    final_res_cm = cm   # using final model output to calculate the cost of final error
    # break

  record_df_acc = pd.DataFrame(record_dict_acc.values(), index=record_dict_acc.keys())
  record_df_recall = pd.DataFrame(record_dict_recall.values(), index=record_dict_recall.keys())
  record_df_precision = pd.DataFrame(record_dict_precision.values(), index=record_dict_precision.keys())
  record_df_f1 = pd.DataFrame(record_dict_f1.values(), index=record_dict_f1.keys())
  record_df_auc = pd.DataFrame(record_dict_auc.values(), index=record_dict_auc.keys())
  record_df_process_time = pd.DataFrame(record_dict_time_process.values(), index=record_dict_time_process.keys())

  # a collection of list to record evaluation results
  accuracy = list()
  interact_num = list()
  num_iter = list()
  recall = list()
  precision = list()
  f1 = list()
  auc = list()
  process_time = list()

  for i in range(0, record_df_acc.shape[0]):
    for j in record_df_acc.iloc[i].index:
      # store all evaluation metrics
      interact_num.append(record_df_acc.iloc[i].name)
      num_iter.append(j)
      accuracy.append(record_df_acc.iloc[i][j])
      precision.append(record_df_precision.iloc[i][j])
      recall.append(record_df_recall.iloc[i][j])
      f1.append(record_df_f1.iloc[i][j])
      auc.append(record_df_auc.iloc[i][j])

      process_time.append(record_df_process_time.iloc[i][j])


  new_df = pd.DataFrame()
  new_df['interact_num'] = interact_num
  new_df['iter_num'] = num_iter
  new_df['acc'] = accuracy
  ####
  new_df['precision'] = precision
  new_df['recall'] = recall
  new_df['f1'] = f1
  new_df['auc'] = auc
  new_df['process_time'] = process_time   # process time is training execution time
  new_df['process_time_cum'] = new_df['process_time'].cumsum()    # cumulating processing time
  # for k, v in cnn_record_dict_cm.items():
  #   new_df[k] = v

  new_df['order'] = order
  new_df.to_csv('cifar10_' + cur_noise + '_noise_order' + order + '_beta' + percent_noise + '.csv')
  exp_df = pd.concat([exp_df, new_df], ignore_index=True)

# user_dict = human_flip_err_dict
# human_flip_err_df = pd.DataFrame.from_dict({(i,j,k): user_dict[i][j][k]
#                         for i in user_dict.keys()
#                         for j in user_dict[i].keys()
#                         for k in user_dict[i][j].keys()},
#                     orient='index')

exp_df.to_csv("cifar10_" + cur_noise + "_noise" + ".csv")

