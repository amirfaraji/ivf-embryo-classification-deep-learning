################ IMPORT LIBRARY ##################
import keras.backend as K
import tensorflow as tf
import numpy as np



################ FUNCTIONS ##################
def weighted_binary_loss_crossentropy(weights):
    """Weighted Binary Cross Entropy Loss Function"""
    def loss(y_true, y_pred):
        epsilon = tf.constant(K.epsilon())
        logloss = K.mean(-(y_true * K.log(y_pred + epsilon) * tf.constant(weights[0]) 
                    + (tf.constant(1.0) - y_true) * K.log(tf.constant(1.0) - y_pred + epsilon) * tf.constant(weights[1])),
                    axis=-1)
        return logloss
    return loss

def jaccard_index(y_true, y_pred, smooth=1e-12):
    """Jaccard Index"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_index_loss(y_true, y_pred):
    """Jaccard Loss Function"""
    return 1-jaccard_index(y_true, y_pred)

def binary_class_weights(data):
    """
    Calculates weights based on distribution
    Currently returns [1,1]
    """
    num_of_ones = np.count_nonzero(data)
    num_of_zeros = len(data) - num_of_ones
    print(num_of_ones, num_of_zeros)
    return [1, 1]#[1*(num_of_ones+num_of_zeros)/num_of_ones, 1.0]

def accuracy_confusion_matrix(cm):
    return (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[1][0] + cm[0][1] + cm[0][0])
        
def recall_confusion_matrix(cm):        
    return cm[1][1] / (cm[1][1] + cm[1][0])
        
def precision_confusion_matrix(cm):
    return cm[1][1] / (cm[1][1] + cm[0][1])

def jaccard_index_cm(cm):
    return cm[1][1] / (cm[1][1] + cm[0][1] + cm[1][0])
