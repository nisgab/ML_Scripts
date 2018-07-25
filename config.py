#photos meta data
img_width = 150
img_height = 150
testImagesType = 'jpeg'
originImagesType = 'jpg'
trainAmount = 0.7
validationAmount = 0.2
testAmount = 0.1


#file names data
bottleneckFileName = '../files/bottleneck_features_train_smallAmount_vgg'
modelFileName = '../files/model_cvd_v9.h5'
labelsFileName = '../files/labels.npy'

#training data
epochs = 10
batch_size = 30
num_classes = 2


#paths
originImagesFolder = '../files/catsNdogs'
mutatedPath = '../files/catsNdogs_data_v2'

trainPath = f'{mutatedPath}/train'
validationPath = f'{mutatedPath}/validation'
testPath = f'{mutatedPath}/test'
