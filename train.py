import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import datetime
import math
import os
import csv
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import L1L2
from keras.models import load_model
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from keras.utils import to_categorical
import config as config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_amount_of_files_per_folder(path) :
    filesPerFolder = []
    for _, dirnames, filenames in os.walk(path):
        filesPerFolder.extend([len(filenames)])

    del filesPerFolder[0]
    return filesPerFolder

def get_labels(amount_of_files_per_type):
    labels = []
    labelCounter = 0
    for i in range(len(amount_of_files_per_type)):
        for j in range((amount_of_files_per_type[i])):
            labels.append([labelCounter])
        labelCounter += 1
    #return np.array(labels)    
    return labels

nb_train_samples = sum([len(files) for r, d, files in os.walk(config.trainPath)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(config.validationPath)])
train_samples_per_type = get_amount_of_files_per_folder(config.trainPath)
validation_samples_per_type = get_amount_of_files_per_folder(config.validationPath)
traintime = datetime.datetime.utcnow().strftime('%d%m%Y-%H%M%S')


print(f'{datetime.datetime.utcnow()} : loading train bottleneck')
train_data = np.load(open(f'{config.bottleneckFileName}.npy', 'rb'))
train_labels = get_labels(train_samples_per_type)

print(f'{datetime.datetime.utcnow()} : loading validation bottleneck')
validation_data = np.load(open(f'{config.bottleneckFileName}_validation.npy', 'rb'))
validation_labels = get_labels(validation_samples_per_type)


def create_model(train_data):
     
    model = Sequential()
    
    model.add(Flatten(input_shape=train_data.shape[1:]))

    model.add(Dense(128,
                    activation='sigmoid',
                    name='dense'))

    model.add(Dropout(0.75))

    model.add(Dense(config.num_classes, activation='sigmoid', 
                kernel_regularizer=L1L2(0, 0.01),
                activity_regularizer=L1L2(0, 0.01), name='out'))
    
    #optimizer = Adam(lr=0.01)
    optimizer = RMSprop(lr=0.005)
    #optimizer = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def fitness():

    global train_data, train_labels, validation_data, validation_labels

    model = create_model(train_data)
   
    history = model.fit(train_data, np.array(train_labels),
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_data=(validation_data, np.array(validation_labels)))

    accuracy = history.history['val_acc'][-1]

    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    model.save(config.modelFileName)
    print("Saved model to disk")

    del model
    
    K.clear_session()
    
    return -accuracy

fitness()