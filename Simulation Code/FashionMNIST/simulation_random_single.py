from keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from subprocess import check_output
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, recall_score, f1_score
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import time
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import collections
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.metrics import  confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text  import TfidfVectorizer
import string as s
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
  lb = LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return roc_auc_score(y_test, y_pred, average=average)


def correct_prob(y_classes, good_user):
  human_classify_probs = dict()
  flip_correct_p = 1
  # if it is the binary classification task, human will 100% flip the label correctly.
  if len(y_classes) == 2: good_user = True
  if good_user: flip_correct_p = 1
  else: flip_correct_p = 0.5

  # generate different probabilities for human change i -> j
  for i in y_classes:
    human_classify_probs[i] = dict()     # y_classes[i] is actual class
    for j in y_classes:
      if i == j: continue
      else:
        human_classify_probs[i][j] = dict()
        # human_classify_probs[i][j][i] = random.randint(95, 100) / 100   # assuming user more likely to flip the label correctly(good), over 80% possibly to correct the label
        human_classify_probs[i][j][i] = flip_correct_p
        remaining_prob = 1 - human_classify_probs[i][j][i]
        num_other_classes = len(y_classes) - 1
      for k in y_classes:
        if k == i or k == j: continue
        else:
          # human_classify_probs[i][j][k] = round(human_classify_probs_ijk[0] * remaining_prob, 3)  # y_classes[j] is flipped class
          human_classify_probs[i][j][k] = remaining_prob/ (num_other_classes - 1)
          # del human_classify_probs_ijk[0]

  return human_classify_probs

# ρ_ij, classification results probabilities
def res_err_probs(cm):
  res = dict()
  sum_i = np.sum(cm)
  for i in range(0, cm.shape[0]):
    true = i
    res[i] = dict()
    for j in range(0, cm.shape[1]):
      res[i][j] = round(cm[i][j] / sum_i * 100, 2)    # percentage of ground truth i is classied to j
  return res


#evaluate the performance of CNN
def cnn_fashion_flip(y_train, X_train, X_test, y_test, img_shape, batch_size, epochs):
    cnn = Sequential([
        Conv2D(filters=16,kernel_size=3,activation='relu',input_shape = img_shape),
        MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
        Dropout(0.2),
        Flatten(), # flatten out the layers
        Dense(32,activation='relu'),
        Dense(10,activation = 'softmax')
    ])

    cnn.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])


    history = cnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    predict_res = cnn.predict(X_test)
    y_pred = np.argmax(predict_res,axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    auc = multiclass_roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, auc, cm, cnn

#evaluate the performance of logistic regression
def lr_fashion_flip(y_train, x_train, x_test, y_test):
    lr = LogisticRegression(C=1.0, multi_class='ovr', penalty='l1', verbose=True, solver='liblinear').fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    auc = multiclass_roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, auc, cm, lr

#evaluate the performance of linear SVC
def svc_fashion_flip(y_train, x_train, x_test, y_test):

    lsvc = LinearSVC(C=1.0, loss='hinge', multi_class='ovr', penalty='l2',verbose=True).fit(x_train, y_train)
    y_pred = lsvc.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    auc = multiclass_roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, auc, cm, lsvc

#evaluate the performance of random forest
def rf_fashion_flip(y_train, x_train, x_test, y_test):

    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=100).fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    auc = multiclass_roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, auc, cm, rf

# evaluate the performance of decision tree
def dt_fashion_flip(y_train, x_train, x_test, y_test):

    dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, splitter='best').fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    auc = multiclass_roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, auc, cm, dt

def tokenization(text):
    lst=text.split()
    return lst


def lowercasing(lst):
    new_lst=[]
    for  i in  lst:
        i=i.lower()
        new_lst.append(i) 
    return new_lst


def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for j in s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst



def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]

    for i in  lst:
        for j in  s.digits:
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in  nodig_lst:
        if  i!='':
            new_lst.append(i)
    return new_lst




def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst





def lemmatzation(lst):
    lemmatizer=nltk.stem.WordNetLemmatizer()
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst


def text_process_pipeline(X_train, X_test):
  # Change the array to a series for using the apply function
  X_train = pd.Series(X_train)
  X_test = pd.Series(X_test)

  X_train=X_train.apply(tokenization)
  X_test=X_test.apply(tokenization)
  X_train=X_train.apply(lowercasing)
  X_test=X_test.apply(lowercasing)
  X_train=X_train.apply(remove_punctuations) 
  X_test=X_test.apply(remove_punctuations)
  X_train=X_train.apply(remove_numbers)
  X_test=X_test.apply(remove_numbers)
  X_train=X_train.apply(remove_stopwords)
  X_test=X_test.apply(remove_stopwords)  
  X_train=X_train.apply(lemmatzation)
  X_test=X_test.apply(lemmatzation)


  X_train=X_train.apply(lambda x: ''.join(i+' ' for i in x))
  X_test=X_test.apply(lambda x: ''.join(i+' '  for i in x))


  tfidf=TfidfVectorizer(max_features=10000,min_df=6)
  train_1=tfidf.fit_transform(X_train)
  test_1=tfidf.transform(X_test)
  # print("No. of features extracted")
  # print(len(tfidf.get_feature_names()))
  # print(tfidf.get_feature_names()[:20])

  X_train_arr=train_1.toarray()
  X_test_arr=test_1.toarray()
  return X_train_arr, X_test_arr





#evaluate the performance of lstm for agnews_10pct
def lstm_agnews_10pct_flip(y_train, X_train, X_test, y_test, batch_size, epochs):
  #Max Length of sentences in Train Dataset
  maxLen = max([len(item.split()) for item in np.array(X_train)])

  # Tokenize and Pad data
  vocab_size = 10000 # arbitrarily chosen
  embed_size = 32 # arbitrarily chosen
  # Create and Fit tokenizer
  tok = Tokenizer(num_words=vocab_size)
  tok.fit_on_texts(X_train)
  # Tokenize data
  X_train = tok.texts_to_sequences(X_train)
  X_test = tok.texts_to_sequences(X_test)
  # Pad data
  X_train = pad_sequences(X_train, maxlen=maxLen)
  X_test = pad_sequences(X_test, maxlen=maxLen)

  lstm = Sequential(
    [
     Embedding(vocab_size, embed_size, input_length=maxLen),
     Bidirectional(LSTM(128, return_sequences=True)),
     Bidirectional(LSTM(64, return_sequences=True)),
     GlobalMaxPooling1D(),
    #  Dense(1024),
    #  Dropout(0.25),
    #  Dense(512),
    #  Dropout(0.25),
     Dense(256),
     Dropout(0.25),
     Dense(128),
     Dropout(0.25),
     Dense(64),
     Dropout(0.25),
     Dense(4),
     Activation('softmax')
    ]
  )
  callbacks = [
    EarlyStopping(     #EarlyStopping is used to stop at the epoch where val_accuracy does not improve significantly
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=6,
        verbose=1
    ),
    # ModelCheckpoint(
    #     filepath='weights.h5',
    #     monitor='val_accuracy', 
    #     mode='max', 
    #     save_best_only=True,
    #     save_weights_only=True,
    #     verbose=1
    # )
  ]

  #Compile and Fit Model
  lstm.compile(loss='sparse_categorical_crossentropy', #Sparse Categorical Crossentropy Loss because data is not one-hot encoded
                optimizer='adam', 
                metrics=['accuracy']) 

  lstm.fit(X_train, 
            y_train, 
            # batch_size=256, 
            batch_size=batch_size, 
            validation_data=(X_test, y_test), 
            # epochs=20, 
            epochs=epochs, 
            callbacks=callbacks)

  predict_res = lstm.predict(X_test)
  y_pred = np.argmax(predict_res,axis=1)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='micro')
  recall = recall_score(y_test, y_pred, average='micro')
  f1 = f1_score(y_test, y_pred, average='micro')
  auc = multiclass_roc_auc_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  return accuracy, precision, recall, f1, auc, cm, lstm



#evaluate the performance of multinomial naive bayes for agnews_10pct
def mn_agnews_10pct_flip(y_train, x_train, x_test, y_test):
  x_train_arr, x_test_arr = text_process_pipeline(x_train, x_test)
  mn = MultinomialNB().fit(x_train_arr,y_train)
  y_pred = mn.predict(x_test_arr)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='micro')
  recall = recall_score(y_test, y_pred, average='micro')
  f1 = f1_score(y_test, y_pred, average='micro')
  auc = multiclass_roc_auc_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  return accuracy, precision, recall, f1, auc, cm, mn


#evaluate the performance of Stochastic Gradient Descent Classifier for agnews_10pct
def sgd_agnews_10pct_flip(y_train, x_train, x_test, y_test):
  x_train_arr, x_test_arr = text_process_pipeline(x_train, x_test)
  sgd = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.001).fit(x_train_arr,y_train)
  y_pred = sgd.predict(x_test_arr)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='micro')
  recall = recall_score(y_test, y_pred, average='micro')
  f1 = f1_score(y_test, y_pred, average='micro')
  auc = multiclass_roc_auc_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  return accuracy, precision, recall, f1, auc, cm, sgd




# the simulation function
def simulate(dataset, x_train, y_train, x_test, y_test, y_train_true, noise_ratio, model, classes, good_user, percent_noise_list):

    cnt_noise = 0
    for i in range(0, len(y_train)):
        if y_train[i] != y_train_true[i]: cnt_noise += 1
    print("train size: ", y_train.shape[0])
    print("noise size: ", cnt_noise)

    if (dataset == 'fashionMNIST') and (model == 'cnn'):
        batch_size = 128
        epochs = 50
        # num_classes = len(y_classes)
        img_shape = (28, 28, 1)

        # accuracy, precision, recall, f1, auc, cm, cnn = cnn_fashion_flip(y_train_true, x_train, x_test, y_test, img_shape, batch_size, epochs)

        # accuracy, precision, recall, f1, auc, cm, cnn = cnn_fashion_flip(y_train, x_train, x_test, y_test, img_shape, batch_size, epochs)


    if (dataset == 'fashionMNIST') and (model == 'svc'):
        x_train_size = x_train.shape[0]
        x_train = x_train.reshape(x_train_size, -1)
        x_test_size = x_test.shape[0]
        x_test = x_test.reshape(x_test_size, -1)
        # try the baseline model with clean dataset

        # accuracy, precision, recall, f1, auc, cm, lsvc = svc_fashion_flip(y_train_true, x_train, x_test, y_test)
 
        # accuracy, precision, recall, f1, auc, cm, lsvc = svc_fashion_flip(y_train, x_train, x_test, y_test)


    if (dataset == 'fashionMNIST') and (model == 'lr'):
        x_train_size = x_train.shape[0]
        x_train = x_train.reshape(x_train_size, -1)
        x_test_size = x_test.shape[0]
        x_test = x_test.reshape(x_test_size, -1)
        # try the baseline model with clean dataset

        # accuracy, precision, recall, f1, auc, cm, lr = lr_fashion_flip(y_train_true, x_train, x_test, y_test)

        # accuracy, precision, recall, f1, auc, cm, lr = lr_fashion_flip(y_train, x_train, x_test, y_test)


    if (dataset == 'fashionMNIST') and (model == 'dt'):
        x_train_size = x_train.shape[0]
        x_train = x_train.reshape(x_train_size, -1)
        x_test_size = x_test.shape[0]
        x_test = x_test.reshape(x_test_size, -1)
        # try the baseline model with clean dataset

        # accuracy, precision, recall, f1, auc, cm, dt = dt_fashion_flip(y_train_true, x_train, x_test, y_test)

        # accuracy, precision, recall, f1, auc, cm, dt = dt_fashion_flip(y_train, x_train, x_test, y_test)


    if (dataset == 'fashionMNIST') and (model == 'rf'):
        x_train_size = x_train.shape[0]
        x_train = x_train.reshape(x_train_size, -1)
        x_test_size = x_test.shape[0]
        x_test = x_test.reshape(x_test_size, -1)
        # try the baseline model with clean dataset

        # accuracy, precision, recall, f1, auc, cm, rf = rf_fashion_flip(y_train_true, x_train, x_test, y_test)

        # accuracy, precision, recall, f1, auc, cm, rf = rf_fashion_flip(y_train, x_train, x_test, y_test)

    human_classify_probs = correct_prob(classes, good_user)    


    exps = 10 # run ten times for each simulation
    exp_df = pd.DataFrame()
    other_test_res_exp_df = pd.DataFrame()
    final_res_cm = dict()
    human_flip_err_dict = dict()

    for order in range(0, exps):
      # shuffle the dataset for each experiment
      train_idx_shuffled = np.arange(len(y_train))  
      np.random.shuffle(train_idx_shuffled)

      y_train_shuffled = y_train[train_idx_shuffled].copy()
      y_train_true_shuffled = y_train_true[train_idx_shuffled].copy()
      x_train_shuffled = x_train[train_idx_shuffled].copy()


      flipped_indexes = list()
      for i in range(0, y_train_shuffled.shape[0]):
          if y_train_shuffled[i] != y_train_true_shuffled[i]: flipped_indexes.append(i)
     

      # Human inspect and correct instance labels based on probability dict 'human_classify_probs'
      record_dict_acc = dict()
      record_dict_recall = dict()
      record_dict_precision = dict()
      record_dict_f1 = dict()
      record_dict_auc = dict()
      record_dict_time_process = dict()
      cnn_record_dict_cm = dict()

  

      # generate probability dict of user correction if each order is performed by different users
      human_flip_err_dict[order] = human_classify_probs

      
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

          # if not good_user:

          #   # categorize index to different classes
          #   flip_idx_partition_dict = dict()
          #   for j in flip_back_indexes:
          #     cur_label = y_train_flip_shuffled[j]
          #     true_label = y_train_true_shuffled[j]
          #     if cur_label == true_label:
          #       print("contaminate fail index", j)

          #     if true_label not in flip_idx_partition_dict.keys():
          #       flip_idx_partition_dict[true_label] = dict()
          #     if cur_label not in flip_idx_partition_dict[true_label].keys():
          #       flip_idx_partition_dict[true_label][cur_label] = list()
          #     flip_idx_partition_dict[true_label][cur_label].append(j)



            
          #   # split selected indexes to different flipping target classes
          #   flip_to_target_idx_dict = dict()

          #   for true_label, v1 in flip_idx_partition_dict.items():
          #     flip_to_target_idx_dict[true_label] = dict()
          #     for cur_label, idx_list in v1.items():
          #       flip_to_target_idx_dict[true_label][cur_label] = dict()
          #       random.shuffle(idx_list)                  # mimic the random selection
          #       p = 0
          #       for target, prob in human_classify_probs[true_label][cur_label].items():
          #         num = round(prob * len(idx_list))
          #         flip_to_target_idx_dict[true_label][cur_label][target] = idx_list[p : p + num]
          #         p = p + num



          #   start_time = time.time()


          #   for k, v in flip_to_target_idx_dict.items():
          #     for cur, target_dict in flip_to_target_idx_dict[k].items():
          #       for target, idx_list in target_dict.items():
          #         # print(target)
          #         # print(idx_list)
          #         for idx in idx_list:
          #           y_train_flip_shuffled[idx] = target   # flip the label in the original training set to the target class based on probability dict, might be correct might be wrong.


        
        # machine learning models learn the data
        if (dataset == 'fashionMNIST') and (model == 'cnn'): acc, precision, recall, f1, auc, cm, cnn = cnn_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test, img_shape, batch_size=128, epochs=50)
        if (dataset == 'fashionMNIST') and (model == 'svc'): acc, precision, recall, f1, auc, cm, svc = svc_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
        if (dataset == 'fashionMNIST') and (model == 'lr'): acc, precision, recall, f1, auc, cm, lr = lr_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
        if (dataset == 'fashionMNIST') and (model == 'dt'): acc, precision, recall, f1, auc, cm, dt = dt_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
        if (dataset == 'fashionMNIST') and (model == 'rf'): acc, precision, recall, f1, auc, cm, rf = rf_fashion_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)


        if (dataset == 'agnews_10pct') and (model == 'sgd'): acc, precision, recall, f1, auc, cm, sgd = sgd_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
        if (dataset == 'agnews_10pct') and (model == 'mn'): acc, precision, recall, f1, auc, cm, mn = mn_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test)
        if (dataset == 'agnews_10pct') and (model == 'lstm'): acc, precision, recall, f1, auc, cm, lstm = lstm_agnews_10pct_flip(y_train_flip_shuffled, x_train_shuffled, x_test, y_test, batch_size=256, epochs=20)

        end_time = time.time()

        #### performance metricslr
        acc_list.append(acc)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        auc_list.append(auc)

        #### time
        process_time_list.append(end_time - start_time)
        res_error_probs = res_err_probs(cm)
        for k1, v1 in res_error_probs.items():
          for k2, v2 in res_error_probs[k1].items():
            new_key = str(k1) + '-' + str(k2) + '(%)'
            if new_key not in cnn_record_dict_cm.keys():
              cnn_record_dict_cm[new_key] = list()
            cnn_record_dict_cm[new_key].append(v2)


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
      for k, v in cnn_record_dict_cm.items():
        new_df[k] = v

      new_df['order'] = order   
      exp_df = pd.concat([exp_df, new_df], ignore_index=True)
                    
    user_dict = human_flip_err_dict
    human_flip_err_df = pd.DataFrame.from_dict({(i,j,k): user_dict[i][j][k] 
                            for i in user_dict.keys() 
                            for j in user_dict[i].keys()
                            for k in user_dict[i][j].keys()},
                        orient='index')

    return exp_df, human_flip_err_df