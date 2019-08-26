import gc
import sys

import argparse
import numpy as np
import os
import pandas as pd
import warnings
from keras import callbacks
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
sys.path.insert(0, '../utils/')

from data import *
from model import get_dense_model

from keras import backend as K

import tensorflow as tf

'''
Compatible with tensorflow backend
'''


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--reproduce', type=bool, default=False)

    args = parser.parse_args()

    '''
    setting image and training parameters
    '''
    model_name = 'densenet'
    batch_size = 64
    img_width, img_height = 112, 112
    channel = 3
    epochs = 100
    lr = 1e-3
    log = []
    split_size = 2

    train_df, val_df, test_df = get_only_data_lists()
    train_df = pd.concat([train_df, val_df])

    '''
    loading models and weights
    '''
    num_classes = 5
    model5 = get_dense_model(num_classes, img_width,
                             img_height, channel, lr, loss=focal_loss())
    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=25,
        verbose=1,
        mode='auto')
    if not args.reproduce:
        weights_file = '../weights/weights_' + model_name + '_' + str(num_classes) + '_classes_' + str(
            img_width) + 'x' + str(
            img_height) + '.hdf5'

    else:
        weights_file = '../weights/reproduce.hdf5'
    checkpointer5 = callbacks.ModelCheckpoint(filepath=weights_file, verbose=1,
                                              save_best_only=True, period=1)
    if os.path.isfile(weights_file):
        try:
            model5.load_weights(weights_file)
            print('weights loaded.')
        except:
            print('could not load {}'.format(weights_file))

    '''
    Split training..............
    '''

    if not args.reproduce:
        for i in range(10):
            x_train, y_train3, y_train5 = get_partial_data(shuffle(train_df), img_width=img_width,
                                                           img_height=img_height, channel=channel)

            # x_val, y_val3, y_val5 = get_partial_data(val_df, img_width=img_width,
            # img_height=img_height, channel=channel)
            model5.fit(x_train,
                       y_train5,
                       epochs=epochs,
                       verbose=1,
                       batch_size=batch_size,
                       validation_split=0.15,
                       # validation_data=(x_val, y_val5),
                       callbacks=[checkpointer5])

            # loading model state where it achieved best performance on validation
            # dataset
            if os.path.isfile(weights_file):
                try:
                    model5.load_weights(weights_file)
                    print('weights loaded again.')
                except:
                    print('could not load {}'.format(weights_file))
                    sys.exit()

        del train_df, val_df, x_train, y_train3, y_train5
        gc.collect()

    print('loading testing data. Please wait' + '.' * 10)
    x_test, y_test3, y_test5 = get_partial_data(
        test_df, img_width=img_width, img_height=img_height, channel=channel)
    print('data loaded.')
    # print(model5.evaluate(x_test, y_test5, batch_size=64, verbose=1))

    predictions5 = model5.predict(x_test, verbose=1, batch_size=64)

    cnf_matrix = confusion_matrix(np.argmax(y_test5, axis=1),
                                  np.argmax(predictions5, axis=1),
                                  labels=[0, 1, 2, 3, 4])
    # print(cnf_matrix)

    cnf_matrix_norm = np.round(cnf_matrix / np.sum(cnf_matrix, axis=0), 3)
    print('confusion matrix for 5 classes: 0 1 2 4 5')
    print(cnf_matrix_norm)

    print('classification accuracy for type 1 is {}%.\nclassification accuracy for type 2 is {}%.'.format(
        cnf_matrix_norm[1][1] * 100, cnf_matrix_norm[2][2] * 100))

    acc = accuracy_score(np.argmax(y_test5, axis=1),
                         np.argmax(predictions5, axis=1))

    print('Accuracy Score: ')
    print(acc)
