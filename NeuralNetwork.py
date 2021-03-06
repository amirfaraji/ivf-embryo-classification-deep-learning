################ IMPORT LIBRARY ##################
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Concatenate, UpSampling2D, Multiply, Input, Add
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator

from os import path
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np

import Metrics

################ Neural Net Class ##################

class NeuralNetwork:
    """
    Neural Network Model for Embryo Classification.
    Three models present: 
        Baseline Convolutional Network, Residual Attention Network & Concatenated Attention Network
    """
    def __init__(self, **hyperparameters):
        """Initialize Hyperparameters of the Model"""
        print("Initializing Hyperparameters")
        self.batch_size = hyperparameters["batch_size"]
        self.epochs = hyperparameters["epochs"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.classes = hyperparameters["classes"]
        self.num_of_classes = hyperparameters["num_of_classes"]
        self.model_to_train = hyperparameters["model_to_train"]
        self.weight_path = hyperparameters["weight_path"]
        self.flags = {
            "preprocessing": hyperparameters["preprocessing_flag"],
            "data_setup"   : hyperparameters["data_setup_flag"],
            "load_weight"  : hyperparameters["load_weight_flag"]
        }

        if K.image_data_format == 'channel_first':
            self.img_shape = (3, hyperparameters["img_wid"], hyperparameters["img_hgt"])
        else:
            self.img_shape = (hyperparameters["img_wid"], hyperparameters["img_hgt"], 3)
        
        self.filters = [16,32,64]
        pass

    def conv_bn_relu(self,inputs, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
        """Order of Common Layers"""
        x = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def conv_relu_bn(self,inputs, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
        """Order of Common Layers"""
        x = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding=padding)(inputs)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)
        return x

    def residual_connection(self, input1, input2, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
        conv_layer = Conv2D(nb_outputs, kernel_size=kernel, strides=(1,1), padding='same')(input1)
        conv_layer = BatchNormalization()(conv_layer)

        shortcut = Conv2D(nb_outputs, kernel_size=kernel, strides=(1,1), padding='same')(input2)
        shortcut = BatchNormalization()(shortcut)

        res = Add()([conv_layer, shortcut])
        res = Activation(activation)(res)
        return res

    def build_conv_net(self,):
        inputs = Input(self.img_shape)

        # Block 1
        conv1 = self.conv_bn_relu(inputs, 8, 3, (2,2))
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        # Block 2 
        conv2 = self.conv_bn_relu(pool1, 16, 1)
        conv2 = self.conv_bn_relu(conv2, 16, 3)

        res2 = self.residual_connection(conv2, pool1, 32, 1)
        pool2 = MaxPooling2D(pool_size=(2,2))(res2)

        # Block 3
        conv3 = self.conv_bn_relu(pool2, 32, 1)
        conv3 = self.conv_bn_relu(conv3, 32, 3)
        
        res3 = self.residual_connection(conv3, pool2, 64, 1)
        # pool3 = MaxPooling2D(pool_size=(2,2))(res3)

        fcn = GlobalAveragePooling2D()(res3)
        # Dense(32, activation='sigmoid')(fcn)
        # fcn = Dropout(0.3)(fcn)
        outputs = Dense(1, activation='sigmoid')(fcn)

        self.model = Model(input=inputs, output=outputs)
        pass
    
    def build_res_att_net(self,):
        inputs = Input(self.img_shape)

        # Block 1
        conv1 = self.conv_bn_relu(inputs, 8, 3, (2,2))
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        # Block 2 
        conv2 = self.conv_bn_relu(pool1, 16, 1)
        conv2 = self.conv_bn_relu(conv2, 16, 3)

        res2 = self.residual_connection(conv2, pool1, 32, 1)

        attent2 = self.attention_module_medium(pool1, 16, 32)
        multiply2 = Multiply()([attent2, res2])
        h2 = Add()([multiply2, res2])

        pool2 = MaxPooling2D(pool_size=(2,2))(h2)

        # Block 3
        conv3 = self.conv_bn_relu(pool2, 32, 1)
        conv3 = self.conv_bn_relu(conv3, 32, 3)
        
        res3 = self.residual_connection(conv3, pool2, 64, 1)

        attent3 = self.attention_module_medium(pool2, 32, 64)
        multiply3 = Multiply()([attent3, res3])
        h3 = Add()([multiply3, res3])

        # pool3 = MaxPooling2D(pool_size=(2,2))(h3)

        fcn = GlobalAveragePooling2D()(h3)
        # Dense(32, activation='sigmoid')(fcn)
        # fcn = Dropout(0.3)(fcn)
        outputs = Dense(1, activation='sigmoid')(fcn)

        self.model = Model(input=inputs, output=outputs)
        pass

    def build_con_att_net(self,):
        inputs = Input(self.img_shape)

        # Block 1
        conv1 = self.conv_bn_relu(inputs, 8, 3, (2,2))
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        # Block 2 
        conv2 = self.conv_bn_relu(pool1, 16, 1)
        conv2 = self.conv_bn_relu(conv2, 16, 3)

        res2 = self.residual_connection(conv2, pool1, 32, 1)

        attent2 = self.attention_module_medium(pool1, 16, 32)
        h2 = Concatenate()([attent2, res2])

        pool2 = MaxPooling2D(pool_size=(2,2))(h2)

        # Block 3
        conv3 = self.conv_bn_relu(pool2, 32, 1)
        conv3 = self.conv_bn_relu(conv3, 32, 3)
        
        res3 = self.residual_connection(conv3, pool2, 64, 1)

        attent3 = self.attention_module_medium(pool2, 32, 64)
        h3 = Concatenate()([attent3, res3])

        # pool3 = MaxPooling2D(pool_size=(2,2))(h3)

        fcn = GlobalAveragePooling2D()(h3)
        # Dense(32, activation='sigmoid')(fcn)
        # fcn = Dropout(0.3)(fcn)
        outputs = Dense(1, activation='sigmoid')(fcn)

        self.model = Model(input=inputs, output=outputs)
        pass

    def attention_module_medium(self,inputs, num_filts, out_filts):
        """U-Net Architecture"""
        conv1 = self.conv_bn_relu(inputs, 4*num_filts, 3)
        conv1 = self.conv_bn_relu(conv1, 4*num_filts, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_bn_relu(pool1, 8*num_filts, 3)
        conv2 = self.conv_bn_relu(conv2, 8*num_filts, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_bn_relu(pool2, 16*num_filts, 3)
        conv3 = self.conv_bn_relu(conv3, 16*num_filts, 3)
        
        up4 = self.conv_bn_relu(
            UpSampling2D(size=(2, 2))(conv3),
            8*num_filts, 
            3
        )
        merge4 = Concatenate()([conv2, up4])
        conv4 = self.conv_bn_relu(merge4, 8*num_filts, 3)
        conv4 = self.conv_bn_relu(conv4, 8*num_filts, 3)

        up5 = self.conv_bn_relu(
            UpSampling2D(size=(2, 2))(conv4),
            8*num_filts, 
            3
        )
        merge5 = Concatenate()([conv1, up5])
        conv5 = self.conv_bn_relu(merge5, 4*num_filts, 3)
        conv5 = self.conv_bn_relu(conv5, 4*num_filts, 3)

        outputs = self.conv_bn_relu(conv5, out_filts, 1, activation='sigmoid')

        return outputs

    def build_model(self,):
        """Build desired model"""
        if (self.model_to_train == 'Residual-Attention'):
            self.build_res_att_net()
        elif (self.model_to_train == 'Concate-Attention'):
            self.build_con_att_net()
        elif (self.model_to_train == 'ConvNet'):
            self.build_conv_net
        else:
            print("Model " + self.model_to_train + " not recognized. Building Residual Attention")
            self.build_res_att_net()

        if (self.flags["load_weight"]):
            print("Load weights when weights created")

        opti = optimizers.adam(lr=self.learning_rate)
        #loss = Metrics.weighted_binary_loss_crossentropy(Metrics.binary_class_weights(data))
        self.model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy', Metrics.jaccard_index])
        self.model.summary()

        pass

    def load_weights(self,):
        if path.exists(self.weight_path):
            self.model.load_weights(self.weight_path)
        else: 
            print("Weight Path does not exist")
        pass

    def train(self, y_train_set, y_valid_set, y_test_set, x_train_set, x_valid_set, x_test_set):
        train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=179,
        fill_mode='reflect',
        shear_range=0.05)

        validation_datagen = ImageDataGenerator()
        val_data = validation_datagen.flow(x_valid_set, y_valid_set, batch_size=self.batch_size)

        weight_checkpoint_path= self.model_to_train+"-weights-improvement.h5"
        checkpoint = ModelCheckpoint(
            weight_checkpoint_path, 
            monitor='val_acc', 
            verbose=1, 
            save_best_only=True,
            mode='max')
        earlystopping = EarlyStopping(monitor='val_acc',
                                    patience=150)

        # fits the model on batches with real-time data augmentation:
        self.history = self.model.fit_generator(train_datagen.flow(x_train_set, y_train_set, batch_size=self.batch_size),#, save_to_dir='./GeneratedImages', save_prefix='GenImg'),
                            steps_per_epoch=len(y_train_set) // self.batch_size, epochs=self.epochs, validation_data=val_data,
                            callbacks=[checkpoint, earlystopping])
        pass

    def predict(self, x_test, y_test):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        opti = optimizers.adam(lr=self.learning_rate)
        self.model.load_weights(self.model_to_train+'-weights-improvement.h5')
        self.model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy', Metrics.jaccard_index])
    
        array = self.model.predict(x_test)
        Results = [np.around(x) for x in array]
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, Results)

        print(cm)

        accuracy = (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[1][0] + cm[0][1] + cm[0][0])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        print("Accuracy: ", accuracy, "Recall: ", recall, "Precision: ", precision)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Embryo Prediction Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = 'd'
        thresh = cm.max()+cm.min() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('ConfusionMatrix' + '.png')
        plt.close()
        #plt.show()
        return cm

    def run(self,x_train_set, y_train_set, x_valid_set, y_valid_set, x_test_set, y_test_set): 
        self.build_model()
        self.train(x_train_set, y_train_set, x_valid_set, y_valid_set, x_test_set, y_test_set)
        pass

    def save_graph_and_parameters(self, fold, saveDir = './saved_data'):
        time = datetime.datetime.now().strftime("%Y-%m-%d%H-%M-%S")
        name = saveDir + '/TrainAcc_vs_ValAcc_' + str(np.amax(self.history.history['val_acc'])) + "_" + time + "_" +str(fold)

        f = open(name + '.txt','w')
        a = "Training Accuracy: " + str(np.amax(self.history.history['acc'])) + '\n' 
        a += "Validation Accuracy: " + str(np.amax(self.history.history['val_acc'])) + '\n'
        a += "Epochs: " + str(self.epochs) + '\n' 
        a += "Batch size: " + str(self.batch_size) + '\n' 
        a += "Learning Rate: " + str(self.learning_rate) + '\n'  

        f.write(str(a))
        f.close()

        # Plot Train vs Valid and Save
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        #plt.show()
        plt.savefig(name + '.png')
        plt.close()
        pass


# To Add & Refactor from my Original Code:
# Attention Modules: small, large