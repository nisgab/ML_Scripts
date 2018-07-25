from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import applications
from keras.models import load_model
import numpy as np
import os
import glob 
import config as config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

incModel = applications.InceptionResNetV2(include_top=False, weights='imagenet')

model = load_model(config.modelFileName)

# Load labels
labels = np.load(config.labelsFileName).item()
inv_lables = {v: k for k, v in labels.items()}
print(inv_lables)

types = os.listdir(config.testPath)

dogCounter = 0
catCounter = 0

for type in types:
    dogCounter = 0
    catCounter = 0
    print (f'working on type: {type}')

    
    for fileName in glob.glob(f'{config.testPath}\\{type}\\*.{config.testImagesType}'):
        img = load_img(fileName)  # this is a PIL image
        x = img_to_array(img.resize((config.img_height, config.img_width)))  # this is a Numpy array with shape (3, 150, 150)
        x = x/255
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        bottleneck_prediction = incModel.predict(x)  
       
        # use the bottleneck prediction on the top model to get the final classification  
        class_predicted = model.predict_classes(bottleneck_prediction)  
        if class_predicted[0] == 0:
            catCounter += 1
        else :
            dogCounter += 1
        
        #print(f'{class_predicted} - {inv_lables[class_predicted[0]]}')
    print (f'total cats : {catCounter}, dogs : {dogCounter}')
