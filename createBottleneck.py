import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import math
import datetime
import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

train_data_dir = '../files/catsNdogs_data_v2/train'
validation_data_dir = '../files/catsNdogs_data_v2/validation'
img_width, img_height = 150, 150
bottleneckFileName = 'bottleneck_features_train_smallAmount_vgg'
labelsFileName = 'labels_smallAmount_vgg'

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 100
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

# build the VGG16 network
model = applications.VGG19(include_top=False, weights='imagenet')

generator = datagen.flow_from_directory(
    config.trainPath,
    target_size=(config.img_width, config.img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
print(f'{datetime.datetime.utcnow()} : started train bottelneck')
predict_size_train = int(math.ceil(nb_train_samples / batch_size))

bottleneck_features_train = model.predict_generator(generator, predict_size_train)

print(f'{datetime.datetime.utcnow()} : ended train bottleneck')
np.save(open(f'{config.bottleneckFileName}.npy', 'wb'),
        bottleneck_features_train)
print(f'{datetime.datetime.utcnow()} : saved')

print (f'{datetime.datetime.utcnow()} : saving labels')
np.save(f'{config.labelsFileName}.npy', generator.class_indices) 

generatorValidation = datagen.flow_from_directory(
    config.validationPath,
    target_size=(config.img_width, config.img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
print(f'{datetime.datetime.utcnow()} : started validation bottelneck')
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))
bottleneck_features_validation = model.predict_generator(generatorValidation, predict_size_validation)

print(f'{datetime.datetime.utcnow()} : ended validation bottelneck')
np.save(open(f'{config.bottleneckFileName}_validation.npy', 'wb'),
        bottleneck_features_validation)
print(f'{datetime.datetime.utcnow()} : saved')
