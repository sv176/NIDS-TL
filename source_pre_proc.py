import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

from large_neural_network_models import *
import large_neural_network_models as lnn

seed = 7
np.random.seed(seed)

desired_width= 200
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', desired_width)
myadam= tf.keras.optimizers.Adam(learning_rate=0.001)



#Transfer Learning Phase:
#This phase will train the specific attack class along with the normal features.
#An identical set up will also be used, but without using the pre-trainied model. the same architecture of the neural network will be used with varying target data sizes, resulting in an accuracy trend that will be shown with the change in the size of data.
#Each experiment is repeated 5 times and averaged (as recommended by the author, Dr Ankush Singla) Accuracies and other performance metrics will be tested for each type of attack




Labels = ["dur",
         "proto",
         "service",
         "state",
         "spkts",
         "dpkts",
         "sbytes",
         "dbytes",
         "rate",
         "sttl",
         "dttl",
         "sload",
         "dload",
         "sloss",
         "dloss",
         "sinpkt",
         "dinpkt",
         "sjit",
         "djit",
         "swin",
         "stcpb",
         "dtcpb",
         "dwin",
         "tcprtt",
         "synack",
         "ackdat",
         "smean",
         "dmean",
         "trans_depth",
         "response_body_len",
         "ct_srv_src",
         "ct_state_ttl",
         "ct_dst_ltm",
         "ct_src_dport_ltm",
         "ct_dst_sport_ltm",
         "ct_dst_src_ltm",
         "is_ftp_login",
         "ct_ftp_cmd",
         "ct_flw_http_mthd",
         "ct_src_ltm",
         "ct_srv_dst",
         "is_sm_ips_ports",
         "attack_cat",
         "Label"]

#read training and testing data
train_file= pd.read_table("UNSW_NB15_training-set2.csv", sep=",", names= Labels)
test_file= pd.read_table("UNSW_NB15_testing-set.csv", sep=",", names=Labels)

input= "DoS"

#separate data into train and test file, with 0.75:0.25 ratio
except_reconnaissance_train= train_file[train_file.attack_cat !=input]
except_reconnaissance_train.reset_index(drop=True, inplace=True)
except_reconnaissance_test= test_file[test_file.attack_cat !=input]
except_reconnaissance_test.reset_index(drop=True, inplace=True)

uniqueAttacks =list(except_reconnaissance_train["attack_cat"].unique())
print("Entries:", uniqueAttacks)



###############Pre-processing###############
X_train = except_reconnaissance_train.drop(['attack_cat', 'Label'], axis =1)
print("Train class created...")
X_test= except_reconnaissance_test.drop(columns=['attack_cat','Label'],axis=1)
print('Test class created...')

#Collecting categorical columns from train and test
categoric_columns_train= X_train.select_dtypes(include= ['object','bool']).copy()
categoric_columns_train.reset_index(drop=True, inplace=True)
print("Categorical columns (train) created...")
categoric_columns_test= X_test.select_dtypes(include= ['object','bool']).copy()
categoric_columns_test.reset_index(drop=True, inplace=True)
print("Categorical columns (test) created...")

#Collecting numerical columns from train and test
numeric_columns_train= X_train.select_dtypes(include= ['int64','float64'])
numeric_columns_train.reset_index(drop=True, inplace=True)
print("Numerical columns (train) created...")
numeric_columns_test= X_test.select_dtypes(include= ['int64','float64'])
numeric_columns_test.reset_index(drop=True, inplace=True)
print("Numerical columns (test) created...")

#Normalising numeric columns from train and test
scale= StandardScaler()
scaled_numeric_train= pd.DataFrame(scale.fit_transform(numeric_columns_train),columns= numeric_columns_train.columns)
scaled_numeric_train.reset_index(drop=True, inplace=True)
print("Numeric columns (train) scaled")
scaled_numeric_test= pd.DataFrame(scale.fit_transform(numeric_columns_test),columns= numeric_columns_test.columns)
scaled_numeric_test.reset_index(drop=True, inplace=True)
print("Numeric columns (test) scaled")

#Encoding categorical columns from train and test
encoder= LabelEncoder()
    ##
encoded_categoric_train= categoric_columns_train.apply(encoder.fit_transform)
encoded_categoric_train.reset_index(drop=True, inplace=True)
print('Categorical columns (train) encoded')
encoded_categoric_test= categoric_columns_test.apply(encoder.fit_transform)
encoded_categoric_test.reset_index(drop=True, inplace=True)
print('Categorical columns (test) encoded')

cat_train= (except_reconnaissance_train.select_dtypes(include= ['int64','float64']).copy()).apply(encoder.fit_transform)
cat_train.reset_index(drop=True, inplace=True)
cat_test= (except_reconnaissance_test.select_dtypes(include= ['int64','float64']).copy()).apply(encoder.fit_transform)
cat_test.reset_index(drop=True, inplace=True)
cat_Y_train= cat_train[['Label']].copy()
cat_Y_train.reset_index(drop=True, inplace=True)
cat_Y_test= cat_test[['Label']].copy()
cat_Y_test.reset_index(drop=True, inplace=True)


#list out columns of the dataset
class_columns= list(pd.concat([encoded_categoric_train,numeric_columns_train, cat_Y_train],axis=1).columns)

#prepare training set for neural network model
X_trainingx = np.concatenate((encoded_categoric_train.values, scaled_numeric_train), axis=1)
Xtrain= X_trainingx
c,r= cat_Y_train.values.shape
y_train= cat_Y_train.values.reshape(c, )
y_train= y_train[:,np.newaxis]
XY_training= np.concatenate((Xtrain,y_train), axis=1)
XY_training_dataframe= pd.DataFrame(XY_training, columns= class_columns)

#prepare testing set for neural network model
X_testingx= np.concatenate((encoded_categoric_test.values,scaled_numeric_test), axis=1)
Xtest= X_testingx
d,s= cat_Y_test.values.shape
y_test= cat_Y_test.values.reshape(d, )
y_test= y_test[:,np.newaxis]
XY_testing= np.concatenate((Xtest, y_test), axis= 1)
XY_testing_dataframe= pd.DataFrame(XY_testing, columns= class_columns)

#Training and testing set for neural network model
X_training= XY_training_dataframe.drop(['Label'],axis=1)
Y_train= (XY_training_dataframe[['Label']].copy()).values.reshape(c, )
X_testing= XY_testing_dataframe.drop(['Label'],axis=1)
Y_test= (XY_testing_dataframe[['Label']].copy()).values.reshape(d, )

pt_model_values=[]
pre_train=pt_model(X_training,Y_train,"cm_pt_dos.pdf",X_testing,Y_test)
pt_model_values.append(pre_train)
