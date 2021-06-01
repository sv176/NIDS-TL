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
import pickle
from source_pre_proc import *
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


reconnaissance_only= train_file[train_file['attack_cat'] == input]
normal_only= train_file[train_file['attack_cat']== 'Normal']

reconnaissance_only_test= test_file[test_file['attack_cat']==input]
normal_only_test= test_file[test_file['attack_cat']=='Normal']


################Pre-processing phase################
#separate data into features and target
X_reconnaissance = reconnaissance_only.drop(['attack_cat', 'Label'], axis =1)
X_reconnaissance.reset_index(drop=True, inplace=True)
Y_reconnaissance = pd.DataFrame(reconnaissance_only['Label'].copy())
Y_reconnaissance.reset_index(drop=True, inplace=True)

X_reconnaissance_test= reconnaissance_only_test.drop(['attack_cat','Label'], axis=1)
X_reconnaissance_test.reset_index(drop=True, inplace=True)
Y_reconnaissance_test= pd.DataFrame(reconnaissance_only_test['Label'].copy())
Y_reconnaissance_test.reset_index(drop=True, inplace=True)

X_normal = normal_only.drop(['attack_cat', 'Label'], axis =1)
X_normal.reset_index(drop=True, inplace=True)
Y_normal = pd.DataFrame(normal_only['Label'].copy())
Y_normal.reset_index(drop=True, inplace=True)

X_normal_test = normal_only_test.drop(['attack_cat', 'Label'], axis =1)
X_normal_test.reset_index(drop=True, inplace=True)
Y_normal_test = pd.DataFrame(normal_only_test['Label'].copy())
Y_normal_test.reset_index(drop=True, inplace=True)


print('Data separated into features and target column')

#extract categoric columns
categoric_columns_reconnaissance= X_reconnaissance.select_dtypes(include= ['object','bool']).copy()
categoric_columns_reconnaissance_test= X_reconnaissance_test.select_dtypes(include= ['object','bool']).copy()
categoric_columns_normal= X_normal.select_dtypes(include= ['object', 'bool']).copy()
categoric_columns_normal_test= X_normal_test.select_dtypes(include= ['object', 'bool']).copy()
print("Categorical columns created")
#extract numeric columns
numeric_columns_reconnaissance= X_reconnaissance.select_dtypes(include= ['int64','float64'])
numeric_columns_reconnaissance_test= X_reconnaissance_test.select_dtypes(include= ['int64','float64'])
numeric_columns_normal= X_normal.select_dtypes(include= ['int64','float64'])
numeric_columns_normal_test= X_normal_test.select_dtypes(include= ['int64','float64'])
print("Numerical columns created")

#encode categoric columns
encoder= LabelEncoder()
encoded_categoric_reconnaissance= pd.DataFrame(categoric_columns_reconnaissance.apply(encoder.fit_transform))
encoded_categoric_reconnaissance.reset_index(drop=True, inplace=True)
encoded_categoric_reconnaissance_test= pd.DataFrame(categoric_columns_reconnaissance_test.apply(encoder.fit_transform))
encoded_categoric_reconnaissance_test.reset_index(drop=True, inplace=True)
encoded_categoric_normal= pd.DataFrame(categoric_columns_normal.apply(encoder.fit_transform))
encoded_categoric_normal.reset_index(drop=True, inplace=True)
encoded_categoric_normal_test= pd.DataFrame(categoric_columns_normal_test.apply(encoder.fit_transform))
encoded_categoric_normal_test.reset_index(drop=True, inplace=True)
print('Categorical columns encoded')
print('Will encode the target data at the end the when the data is combined and separated')
#print('Steps from now: 1) Separate relevant data into parts listed. 2) Combine data for each branch 3) Encode target data 4) Perform feature selection 5) Train and Test model.)

#scale numeric columns
scale= StandardScaler()
scaled_numeric_reconnaissance= pd.DataFrame(scale.fit_transform(numeric_columns_reconnaissance.values),columns= numeric_columns_reconnaissance.columns)
scaled_numeric_reconnaissance.reset_index(drop=True, inplace=True)
scaled_numeric_reconnaissance_test= pd.DataFrame(scale.fit_transform(numeric_columns_reconnaissance_test.values),columns= numeric_columns_reconnaissance_test.columns)
scaled_numeric_reconnaissance_test.reset_index(drop=True, inplace=True)
scaled_numeric_normal= pd.DataFrame(scale.fit_transform(numeric_columns_normal.values),columns= numeric_columns_normal.columns)
scaled_numeric_normal.reset_index(drop=True, inplace=True)
scaled_numeric_normal_test= pd.DataFrame(scale.fit_transform(numeric_columns_normal_test.values),columns= numeric_columns_normal_test.columns)
scaled_numeric_normal_test.reset_index(drop=True, inplace=True)

#split data into half (x6)
XY_reconnaissance= pd.concat((encoded_categoric_reconnaissance, scaled_numeric_reconnaissance, Y_reconnaissance), axis=1)
XY_reconnaissance_test= pd.concat((encoded_categoric_reconnaissance_test, scaled_numeric_reconnaissance_test, Y_reconnaissance_test), axis=1)
XY_normal= pd.concat((encoded_categoric_normal, scaled_numeric_normal, Y_normal), axis=1)
XY_normal= XY_normal.head(len(XY_reconnaissance))
XY_normal_test= pd.concat((encoded_categoric_normal_test, scaled_numeric_normal_test, Y_normal_test), axis=1)
XY_normal_test= XY_normal_test.head(len(XY_reconnaissance_test))
class_columns= pd.concat([encoded_categoric_reconnaissance,numeric_columns_reconnaissance],axis=1).columns
class_columns_test= pd.concat([encoded_categoric_reconnaissance_test,numeric_columns_reconnaissance_test],axis=1).columns
print("Total number of reconnaissance (attack) data:",len(XY_reconnaissance))
print("Total number of reconnaissance (attack) test data:",len(XY_reconnaissance_test))
print("Total number of normal (benign) data:",len(XY_normal))
print("Total number of normal (benign) test data:",len(XY_normal_test))


#data for set 1
reconnaissance_1= XY_reconnaissance.sample(frac=1)
reconnaissance_1.reset_index(drop=True, inplace=True)
normal_1= XY_normal.sample(frac=1)
normal_1.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 1:",len(reconnaissance_1))
print("Total normal (benign) data in sample 1:",len(normal_1))
#data for set 2
reconnaissance_2= reconnaissance_1.sample(frac=0.5)
reconnaissance_2.reset_index(drop=True, inplace=True)
normal_2= normal_1.sample(frac=0.5)
normal_2.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 2:",len(reconnaissance_2))
print("Total normal (benign) data in sample 2:",len(normal_2))
#data for set 3
reconnaissance_3= reconnaissance_2.sample(frac=0.5)
reconnaissance_3.reset_index(drop=True, inplace=True)
normal_3= normal_2.sample(frac=0.5)
normal_3.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 3:",len(reconnaissance_3))
print("Total normal (benign) data in sample 3:",len(normal_3))
#data for set 4
reconnaissance_4= reconnaissance_3.sample(frac=0.5)
reconnaissance_4.reset_index(drop=True, inplace=True)
normal_4= normal_3.sample(frac=0.5)
normal_4.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 4:",len(reconnaissance_4))
print("Total normal (benign) data in sample 4:",len(normal_4))
#data for set 5
reconnaissance_5= reconnaissance_4.sample(frac=0.5)
reconnaissance_5.reset_index(drop=True, inplace=True)
normal_5= normal_4.sample(frac=0.5)
normal_5.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 5:",len(reconnaissance_5))
print("Total normal (benign) data in sample 5:",len(normal_5))
#data for set 6
reconnaissance_6= reconnaissance_5.sample(frac=0.5)
reconnaissance_6.reset_index(drop=True, inplace=True)
normal_6= normal_5.sample(frac=0.5)
normal_6.reset_index(drop=True, inplace=True)

print("Total reconnaissance (attack) data in sample 6:",len(reconnaissance_6))
print("Total normal (benign) data in sample 6:",len(normal_6))


#testing data
X_reconnaissance_test= XY_reconnaissance_test.drop(['Label'],axis=1)
Y_reconnaissance_test= XY_reconnaissance_test['Label'].copy()
X_reconnaissance_test.reset_index(drop=True, inplace=True)
Y_reconnaissance_test.reset_index(drop=True, inplace=True)

X_normal_test= XY_normal_test.drop(['Label'],axis=1)
Y_normal_test= XY_normal_test['Label'].copy()
X_normal_test.reset_index(drop=True, inplace=True)
Y_normal_test.reset_index(drop=True, inplace=True)

X_testing= pd.concat([X_reconnaissance_test, X_normal_test], axis=0)
Y_testing= pd.concat([Y_reconnaissance_test, Y_normal_test], axis=0)
XY_testing= pd.concat([X_testing,Y_testing], axis=1)
X_test_1= XY_testing.drop(['Label'], axis=1)
Y_test_1= XY_testing['Label'].copy()




#Sample 1 (70000 samples shared between train and test)
X_train_1_reconnaissance=reconnaissance_1.copy()
X_train_1_normal= normal_1.copy()



#Combined Sample
X_train_1= pd.concat([X_train_1_reconnaissance,X_train_1_normal],axis=0)
X_train_1.reset_index(drop=True, inplace=True)
X_train_1=shuffle(X_train_1)


#testing data
X_reconnaissance_test= XY_reconnaissance_test.drop(['Label'],axis=1)
Y_reconnaissance_test= XY_reconnaissance_test['Label'].copy()
X_reconnaissance_test.reset_index(drop=True, inplace=True)
Y_reconnaissance_test.reset_index(drop=True, inplace=True)

X_normal_test= XY_normal_test.drop(['Label'],axis=1)
Y_normal_test= XY_normal_test['Label'].copy()
X_normal_test.reset_index(drop=True, inplace=True)
Y_normal_test.reset_index(drop=True, inplace=True)

#Sample 2 (35000 samples shared between train and test)
X_train_2_reconnaissance= reconnaissance_2.copy()
X_train_2_normal= normal_2.copy()
#Combined Sample
X_train_2= pd.concat([X_train_2_reconnaissance,X_train_2_normal],axis=0)
X_train_2.reset_index(drop=True, inplace=True)

X_train_2=shuffle(X_train_2)




#Sample 3 (17500 samples shared between train and test)
X_train_3_reconnaissance= reconnaissance_3.copy()
X_train_3_normal= normal_3.copy()
#Combined Sample
X_train_3= pd.concat([X_train_3_reconnaissance,X_train_3_normal],axis=0)
X_train_3.reset_index(drop=True, inplace=True)

X_train_3=shuffle(X_train_3)



#Sample 4 (8750 samples shared between train and test)
X_train_4_reconnaissance= reconnaissance_4.copy()
X_train_4_normal= normal_4.copy()
#Combined Sample
X_train_4= pd.concat([X_train_4_reconnaissance,X_train_4_normal],axis=0)
X_train_4.reset_index(drop=True, inplace=True)

X_train_4=shuffle(X_train_4)


#Sample 5 (4375 samples shared between train and test)
X_train_5_reconnaissance= reconnaissance_5.copy()
X_train_5_normal= normal_5.copy()
#Combined Sample
X_train_5= pd.concat([X_train_5_reconnaissance,X_train_5_normal],axis=0)
X_train_5.reset_index(drop=True, inplace=True)

X_train_5=shuffle(X_train_5)


#Sample 6 (2180 samples shared between train and test)
X_train_6_reconnaissance = reconnaissance_6.copy()
X_train_6_normal = normal_6.copy()
#Combined Sample
X_train_6= pd.concat([X_train_6_reconnaissance,X_train_6_normal],axis=0)
X_train_6.reset_index(drop=True, inplace=True)

X_train_6=shuffle(X_train_6)

X1_train= X_train_1.drop(['Label'],axis=1)
Y1_train= X_train_1['Label'].copy()
X1_train.reset_index(drop=True, inplace=True)
Y1_train.reset_index(drop=True, inplace=True)

X2_train= X_train_2.drop(['Label'],axis=1)
Y2_train= X_train_2['Label'].copy()
X2_train.reset_index(drop=True, inplace=True)
Y2_train.reset_index(drop=True, inplace=True)

X3_train= X_train_3.drop(['Label'],axis=1)
Y3_train= X_train_3['Label'].copy()
X3_train.reset_index(drop=True, inplace=True)
Y3_train.reset_index(drop=True, inplace=True)

X4_train= X_train_4.drop(['Label'],axis=1)
Y4_train= X_train_4['Label'].copy()
X4_train.reset_index(drop=True, inplace=True)
Y4_train.reset_index(drop=True, inplace=True)

X5_train= X_train_5.drop(['Label'],axis=1)
Y5_train= X_train_5['Label'].copy()
X5_train.reset_index(drop=True, inplace=True)
Y5_train.reset_index(drop=True, inplace=True)

X6_train= X_train_6.drop(['Label'],axis=1)
Y6_train= X_train_6['Label'].copy()
X6_train.reset_index(drop=True, inplace=True)
Y6_train.reset_index(drop=True, inplace=True)


print(len(X1_train))
print(len(X2_train))
print(len(X3_train))
Xy_train = [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train]
Yx_train = [Y1_train, Y2_train, Y3_train, Y4_train, Y5_train, Y6_train]
X_tested= [X_test_1,X_test_1,X_test_1,X_test_1,X_test_1,X_test_1]
Y_tested= [Y_test_1,Y_test_1,Y_test_1,Y_test_1,Y_test_1,Y_test_1]


loss_curve_tlc={}
loss_curve_bc= {}
for x in range(1,7,1):
    loss_curve_tlc["loss_tlc_dos"+str(x)+".pdf"]=x
for x in range(1,7,1):
    loss_curve_bc["loss_bc_dos"+str(x)+".pdf"]=x


confusion_matrix_tlc= {}
confusion_matrix_bc= {}
for x in range(1,7,1):
    confusion_matrix_tlc["cm_tl_dos"+str(x)+".pdf"]=x
for x in range(1,7,1):
    confusion_matrix_bc["cm_bc_dos"+str(x)+".pdf"]=x


f1_tlc={}
f1_bc= {}
for x in range(1,7,1):
    f1_tlc["f1_tl_dos"+str(x)]=x
for x in range(1,7,1):
    f1_bc["f1_bc_dos"+str(x)]=x


combined_tlc= zip(Xy_train,Yx_train, confusion_matrix_tlc.keys(),f1_tlc.keys())
combined_bc= zip(Xy_train,Yx_train, confusion_matrix_bc.keys(),f1_bc.keys())

f1_dos_base=[]
f1_dos_trans=[]



for a, b, c, d in combined_tlc:
    tl_model=lnn.model_tl(a, b, c, d, X_test_1, Y_test_1)
    f1_dos_trans.append(tl_model)
for a, b, c, d in combined_bc:
    bc_model=lnn.unseen_model(a, b, c, d, X_test_1, Y_test_1)
    f1_dos_base.append(bc_model)

print(f1_dos_trans)
print(f1_dos_base)
np.save("f1_dos_base",f1_dos_base)
np.save("f1_dos_tlc",f1_dos_trans)
