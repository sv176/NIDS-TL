import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import imblearn
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import itertools
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import itertools
import seaborn as sns
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
seed=42

desired_width= 200
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', desired_width)

#label the columns of data
labels = ["dur",
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
         "label"]



#read training and testing data
train_file = pd.read_table("UNSW_NB15_training-set.csv", sep=",", names= labels)
test_file = pd.read_table("UNSW_NB15_testing-set.csv", sep=",", names= labels)
# Get a series of unique values in column 'Age' of the dataframe
uniqueAttacks =list(train_file["attack_cat"].unique())
print("Entries:", uniqueAttacks)
attacks = ['Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
normal = ['Normal']
lab= [1,0]

#print(train_file.info())
#print(train_file.isnull().sum())
#print(train_file.shape)

#check the level of the distrubution of the category and label data
X_train = train_file.drop(['attack_cat', 'label'], axis =1)
Y_distributed_train= train_file['attack_cat'].values
Y_binary = train_file['label'].values
Y_binary_train= Y_binary
print("Train class created...")

Y_distributed_test= test_file['attack_cat'].values
Y_binary_test= test_file['label'].values
X_test= test_file.drop(axis=1,columns=['attack_cat','label'])
print('Test class created...')



distribution = []
for u in uniqueAttacks:
    dist=(Y_distributed_train == u).sum()
    distribution.append(dist)

attackOrNo=[]
for x in lab:
    lab_dist= (Y_binary==x).sum()
    attackOrNo.append(lab_dist)
print(attackOrNo)

#graphical representation of attacks for a distributed classifier
y_posLab=np.arange(len(lab))
plt.bar(y_posLab, attackOrNo, align= 'center', alpha=0.5)
plt.xticks(y_posLab, lab)
plt.xlabel('Labels')
plt.title('Attack vs Normal')
#plt.show()

#graphical representation of attacks for a binary classifier
y_pos=np.arange(len(uniqueAttacks))
plt.bar(y_pos, distribution, align= 'center', alpha=0.5)
plt.xticks(y_pos, uniqueAttacks)
plt.xlabel('Class')
plt.title('Attack Class Distribution')
#plt.show()
print(X_train.shape)
#segregate numeric from categoric columns
categoric_columns_train= X_train.select_dtypes(include= ['object','bool']).copy()
print("Categorical columns (train) created...")
print(categoric_columns_train)
numeric_columns_train= X_train.select_dtypes(include= ['int64','float64'])
print("Numerical columns (train) created...")
numeric_columns_test= X_test.select_dtypes(include= ['int64','float64'])
print("Numerical columns (test) created...")
scale= StandardScaler()
scaled_numeric_train= pd.DataFrame(scale.fit_transform(numeric_columns_train),columns= numeric_columns_train.columns)
scaled_numeric_test= pd.DataFrame(scale.fit_transform(numeric_columns_test),columns= numeric_columns_test.columns)
print("Numeric columns (train and test) scaled")

categoric_columns_test= X_test.select_dtypes(include= ['object','bool']).copy()
print("Categorical columns (test) created...")
print(categoric_columns_test)

encoder= LabelEncoder()
##
cat_train= (train_file.select_dtypes(include= ['int64']).copy()).apply(encoder.fit_transform)
cat_test= (test_file.select_dtypes(include= ['int64']).copy()).apply(encoder.fit_transform)
cat_Y_train= cat_train[['label']].copy()
cat_Y_test= cat_test[['label']].copy()
print("Categories to add attack_cat into the code, will help when using SMOTE!")
##

encoded_categoric_train= categoric_columns_train.apply(encoder.fit_transform)
encoded_test= categoric_columns_test.apply(encoder.fit_transform)
print('Categorical columns (train and test) encoded')

ref_class_col= pd.concat([encoded_categoric_train,numeric_columns_train],axis=1).columns
ref_class = np.concatenate((encoded_categoric_train.values, scaled_numeric_train), axis=1)
X= ref_class
print(ref_class)
c,r= cat_Y_train.values.shape
y_train= cat_Y_train.values.reshape(c, )

smote = SMOTE()
X_res, y_res= smote.fit_sample(X, y_train)
#X_res,y_res = X , y_train
print('Initial set {}'.format(Counter(y_train)))
print('Updated set {}'.format(Counter(y_res)))

print("Data resampled...")

#from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier();
# fit random forest classifier on the training set
classifier.fit(X_res, y_res);
print("Classifier applied to data for feature selection...")
# extract important features
score = np.round(classifier.feature_importances_,3)
importance = pd.DataFrame({'feature':ref_class_col,'importance':score})
importance = importance.sort_values('importance',ascending=False).set_index('feature')
print("Importance score in ascending order...")
# plot importances
plt.rcParams['figure.figsize'] = (11, 4)
importance.plot.bar();
print("Importance score in ascending order...")
#manually extract the best 12 features and go forward with the process
plt.show()
new_column= list(ref_class_col)
new_column.append('label')
selected_features= (importance.index[0:11]).tolist()
#(selected_features.append('label'))
new_y_res= y_res[:,np.newaxis]
res_arr= np.concatenate((X_res,new_y_res), axis=1)
res_df= pd.DataFrame(res_arr, columns= new_column)
ref_test= pd.concat([scaled_numeric_test,encoded_test,test_file['label']],axis=1)
encoder2= OneHotEncoder()
class_dict= defaultdict(list)

attack= [('Attack',1)]
normal= [('Normal',0)]

res_train_set= res_df[(res_df['label'] == 0) | (res_df['label'] == 1)]
class_dict['Attack'+'_'+'Normal'].append(res_train_set)
ref_test_set= ref_test[(ref_test['label'] == 0) | (ref_test['label'] == 1)]
class_dict['Attack'+'_'+'Normal'].append(ref_test_set)

X_spec_train= class_dict['Attack_Normal'][0]
X_spec_test= class_dict['Attack_Normal'][1]


grp_class='Attack_Normal'

selected=selected_features.copy()
selected.append('label')

X_training = X_spec_train[selected_features]
print("Features ready for training")
X_testing = ref_test[selected_features]
encoder2= OneHotEncoder()
print("Features ready for testing")

XY_training= X_spec_train[selected]
XY_testing = X_spec_test[selected]
Y_training= X_spec_train[['label']].copy()
c, r = Y_training.values.shape
Y_train= Y_training.values.reshape(c, )
print("Target axis set up for training")

Y_testing= X_spec_test[['label']].copy()
print(Y_testing.values.shape)
a, b= Y_testing.values.shape
Y_test= Y_testing.values.reshape(a, )
print("Target axis set up for testing")
model=Sequential()

model.add(Dense(8, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_training, Y_train, epochs=10, batch_size=1,
          verbose=2, validation_split=0.1)
score = model.evaluate(X_testing, Y_test)
print('Accuracy: %0.2f%%' % (score[1] * 100))

#Train RandomForestClassifier Model
RF_Classifier = (RandomForestClassifier(criterion='entropy', n_jobs=-1, random_state=0)).fit(X_training, Y_train)

# Train SVM Model
SVC_Classifier = (SVC(random_state=0)).fit(X_training, Y_train)

# Train Gaussian Naive Baye Model
BNB_Classifier = (BernoulliNB()).fit(X_training, Y_train)

# Train Decision Tree Model
DTC_Classifier = (tree.DecisionTreeClassifier(criterion='entropy', random_state=0)).fit(X_training, Y_train)


models = []
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('RandomForest Classifier', RF_Classifier))
models.append(('SVM Classifier', SVC_Classifier))

#run the test dataset
for i, v in models:
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_training))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_training))
    classification = metrics.classification_report(Y_train, v.predict(X_training))
    print('{} {} Model Evaluation'.format(grp_class, i))
    print ("Model Accuracy:" "\n", accuracy)
    print("Confusion matrix:" "\n", confusion_matrix)
    print("Classification report:" "\n", classification)

for i, v in models:
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_testing))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_testing))
    classification = metrics.classification_report(Y_test, v.predict(X_testing))
    print('{} {} Model Test Results'.format(grp_class, i))
    print ("Model Accuracy:" "\n", accuracy)
    print("Confusion matrix:" "\n", confusion_matrix)
    print("Classification report:" "\n", classification)
####################################################################