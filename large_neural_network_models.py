from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Activation, Convolution1D,MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import model_from_yaml

desired_width= 200
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', desired_width)
myadam= tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(8,7)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    cm1=cm*100
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = (cm1 / cm_sum.astype(float))
    annot = np.empty_like(cm_perc).astype(str)
    nrows, ncols = cm_perc.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=3)
    sns.heatmap(cm_perc, annot=annot, annot_kws= {"size":15},fmt='' ,cbar=True,cmap='Blues',vmax=100)
    plt.tight_layout()
    plt.savefig(filename)



def pt_model(X_training,Y_train, cm_file,X_testing,Y_test):

    model = Sequential()

    model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=42))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=myadam,
                  metrics=['accuracy'])
    history= model.fit(X_training, Y_train, epochs=30, validation_split=0.3 , batch_size=16, verbose=2)
    score = model.evaluate(X_testing, Y_test)
    print('Accuracy: %0.2f%%' % (score[1] * 100))

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    predicted_targets = model.predict_classes(X_testing)
    true_targets = Y_test
    label = [0, 1]
    cm_analysis(true_targets,predicted_targets,  cm_file, label, ymap=None, figsize=(8, 7))
    f1_pt = metrics.f1_score(true_targets, predicted_targets)

    return f1_pt


def unseen_model(X_training, Y_train, cm_file,f1_uns,X_test_1, Y_test_1):

    unseen_model = Sequential()

    unseen_model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=42))
    unseen_model.add(tf.keras.layers.Dropout(0.5))
    unseen_model.add(tf.keras.layers.Dense(50, activation='relu'))
    unseen_model.add(tf.keras.layers.Dropout(0.5))
    unseen_model.add(tf.keras.layers.Dense(50, activation='relu'))
    unseen_model.add(tf.keras.layers.Dropout(0.5))
    unseen_model.add(tf.keras.layers.Dense(50, activation='relu'))
    unseen_model.add(tf.keras.layers.Dropout(0.5))
    unseen_model.add(tf.keras.layers.Dense(10, activation='relu'))
    unseen_model.add(tf.keras.layers.Dropout(0.5))
    unseen_model.add(tf.keras.layers.Dense(2, activation='softmax'))
    unseen_model.compile(loss='sparse_categorical_crossentropy', optimizer=myadam,
                  metrics=['accuracy'])
    history=unseen_model.fit(X_training, Y_train, epochs=30, validation_split=0.3 ,batch_size=16, verbose=2)
    score_x = unseen_model.evaluate(X_test_1, Y_test_1)
    print('Accuracy: %0.2f%%' % (score_x[1] * 100))

    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model Loss', fontsize=16)
    #plt.ylabel('Loss (mean squared error)')
    #plt.xlabel('Epoch',fontsize=14)
    #plt.legend(['train', 'val'], loc='upper right', fontsize=12)
    #plt.savefig(loss_curve_file)

    predicted_targets = unseen_model.predict_classes(X_test_1)
    true_targets = Y_test_1
    label=[0,1]
    cm_analysis(true_targets, predicted_targets,cm_file,label,ymap=None,figsize=(8,7))
    f1_uns = metrics.f1_score(true_targets, predicted_targets)

    return f1_uns



def model_tl(X_train, Y_trai,cm_file,f1_tl,X_test_1,Y_test_1):

    yaml_file = open('model.yaml', 'r')
    model_1_yaml = yaml_file.read()
    yaml_file.close()
    model_1 = model_from_yaml(model_1_yaml)
    # load weights into new model
    model_1.load_weights("model.h5")

    model_1.compile(loss='sparse_categorical_crossentropy', optimizer=myadam, metrics=['accuracy'])
    history = model_1.fit(X_train, Y_trai, epochs=30, validation_split=0.3, batch_size=16, verbose=2, callbacks=[callback])
    score_tl_1 = model_1.evaluate(X_test_1, Y_test_1)
    print('Test Accuracy: %0.2f%%' % (score_tl_1[1] * 100))


    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model Loss', fontsize=16)
    #plt.ylabel('Loss (mean squared error)', fontsize=14)
    #plt.xlabel('Epoch',fontsize=14)
    #plt.legend(['train', 'test'], loc='upper right', fontsize=12)
    #plt.savefig(loss_curve_file)

    predicted_targets = model_1.predict_classes(X_test_1)
    true_targets = Y_test_1
    label = [0, 1]
    cm_analysis(true_targets, predicted_targets ,cm_file, label, ymap=None, figsize=(8, 7))
    f1_tl = metrics.f1_score(true_targets, predicted_targets)

    return f1_tl
