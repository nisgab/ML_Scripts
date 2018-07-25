from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob 
import config 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def createPath(path, folderName = None) :
    if folderName is None:
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        if not os.path.exists(f'{path}\\{folderName}'):
            os.makedirs(f'{path}\\{folderName}')

def createDataFolder(folderName = None) :
    if folderName is None:
        createPath(config.trainPath)
        createPath(config.validationPath)
        createPath(config.testPath)
    else:
        createPath(config.trainPath, folderName)
        createPath(config.validationPath, folderName)
        createPath(config.testPath, folderName)

def getDestFolder (totalFiles, currentFileNum):
    if (totalFiles * config.trainAmount > currentFileNum):
        return config.trainPath
    if ((totalFiles * config.trainAmount) + (totalFiles * config.validationAmount) > currentFileNum):
        return config.validationPath
    else:
        return config.testPath


createDataFolder()

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# datagen = ImageDataGenerator(
#         rotation_range=180,
#         width_shift_range=0.1,
#         height_shift_range=0.1,

#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode='nearest')

types = os.listdir(config.originImagesFolder)

for type in types:
    createDataFolder(type)
    print (f'creating {type}')

    amountOfFilesInType = len(glob.glob(f'{config.originImagesFolder}\\{type}\\*.{config.originImagesType}'))

    fileCounter = 0

    for fileName in glob.glob(f'{config.originImagesFolder}\\{type}\\*.{config.originImagesType}'):
        fileCounter += 1
        img = load_img(fileName)  # this is a PIL image
        img =img.resize((config.img_height, config.img_width))
        
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 1
        filePrefix = os.path.splitext(os.path.basename(fileName))[0]
        destFolder = getDestFolder(amountOfFilesInType, fileCounter)

        if 'test' in destFolder: 
            img.save(f'{destFolder}\\{type}\\{filePrefix}.jpeg')
        else:
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)      
            for batch in datagen.flow(x, batch_size=5,
                                    save_to_dir=f'{destFolder}\\{type}', save_prefix=f'{filePrefix}', save_format='jpeg'):
                i += 1
                if i > 20:
                    break  # otherwise the generator would loop indefinitely