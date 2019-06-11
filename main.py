################ IMPORT LIBRARY ##################
import tensorflow as tf
import numpy as np

import NeuralNetwork as NN
import IVFData as IVFD
import Metrics as MT
########## ERROR FIXED WITH THIS ##########
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


hyperparameters = {
    "batch_size"        : 32,
    "learning_rate"     : 0.0001,
    "epochs"            : 1,
    "img_wid"           : 32,
    "img_hgt"           : 32,
    "num_of_classes"    : 2,
    "classes"           : ['No Pregnancy','Pregnancy'],
    "model_to_train"    : 'Residual-Attention', # 'ConvNet', 'Concate-Attention', 'Residual-Attention'
    "weight_path"       : '',
    "preprocessing_flag": False,
    "data_setup_flag"   : True,
    "load_weight_flag"  : False
}

def k_folds_validation(folds):
    confmat = np.zeros((2,2,folds))
    accuracy = np.zeros((folds,))
    recall = np.zeros((folds))
    precision = np.zeros((folds))
    seeds = range(0, folds)
    jaccard = np.zeros((folds))

    for seed in seeds:
        cm = test_run(seed)
        confmat[:, :, seed] = cm
        accuracy[seed] = MT.accuracy_confusion_matrix(cm)
        recall[seed] = MT.recall_confusion_matrix(cm)
        precision[seed] = MT.precision_confusion_matrix(cm)
        jaccard[seed] =  MT.jaccard_index_cm(cm)

    print("Average Accuracy: ",str(np.sum(accuracy)/len(seeds)))
    print("Average Recall: ",str(np.sum(recall)/len(seeds)))
    print("Average Precision: ",str(np.sum(precision)/len(seeds)))
    print("Average Jaccard: ", str(np.sum(jaccard) / len(seeds)))

    pass

def test_run(fold):
    ################ Initialize Network ##################
    nn = NN.NeuralNetwork(**hyperparameters)

    ########## Data Setup ##########
    if (nn.flags["data_setup"]):
        myData = IVFD.IVFData(hyperparameters["img_wid"], hyperparameters["img_hgt"])
        y_data, x_data = myData.setup_data()

    ########## Split Data 70-15-15 ##########
    y_train_set, y_valid_set, y_test_set, x_train_set, x_valid_set, x_test_set = myData.split_k_folds(x_data, y_data, fold)

    ################ Build & Run ##################
    nn.run(y_train_set, y_valid_set, y_test_set, x_train_set, x_valid_set, x_test_set)

    ################ Predict & Save ##################
    cm = nn.predict(x_test_set, y_test_set)
    nn.save_graph_and_parameters(fold)

    return cm

def main():
    k_folds_flag = False
    num_of_folds = 6

    if (k_folds_flag):
        k_folds_validation(num_of_folds)
    else:
        test_run(0)
    pass

if __name__== "__main__":
   main()
