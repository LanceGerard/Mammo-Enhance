import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import keras.metrics as mtr

from skimage import color, data, restoration
from scipy.signal import convolve2d
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.resnet_v2 import ResNet152V2, preprocess_input, decode_predictions
from keras.models import Sequential, Model



img_height, img_width = (224, 224)
batch_size = 32
epochs = 10

results_dict = {}
resultkeys = []

train_data_dir = r"D:/Downloads/testMammo/train" # valid_data_dir = r""
test_data_dir = r"D:/Downloads/testMammo/test"

SMALL_data_dir = r"D:/Documents/Adamson University/2022-2023 SEM1/CSRP2/DDSM_small-dataset"
save_enhanced_dir = r"D:/Documents/Adamson University/2022-2023 SEM1/CSRP2/DDSM_enh-small-dataset"

def mammoEnhance(filename, where, what):
    raw_img = cv2.imread(filename)
    to_write = ""

    # greyed
    gray1_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    match what:
        case "gs":
            # 2D FILTERS
            sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gaussian = np.array([[0.075, 0.124, 0.075],[0.124, 0.204, 0.124],[0.075, 0.124, 0.075]])
            gaussed_img = cv2.filter2D(gray1_img, -1, gaussian)
            sharped_img = cv2.filter2D(gaussed_img, -1, sharpen)

            R_sharped_img = cv2.cvtColor(sharped_img, cv2.COLOR_GRAY2RGB) # To RGB
            to_write = R_sharped_img
        case "cl":
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            equalized_img = clahe.apply(gray1_img)

            R_equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB) # To RGB
            to_write = R_equalized_img
        case "wi":
            # WIENER FILTER
            psf = np.ones((5,5)) / 25
            convolved = convolve2d(gray1_img, psf,'same')
            convolved += 0.1 * convolved.std() * np.random.standard_normal(convolved.shape)
            wiener_img = restoration.wiener(convolved, psf, 1100, clip=False)

            R_wiener_img = cv2.cvtColor(wiener_img.astype(np.uint8), cv2.COLOR_GRAY2RGB) # To RGB
            to_write = R_wiener_img
        case "co":
            # CONTRAST
            contrasted_img = cv2.convertScaleAbs(gray1_img, alpha=1.0, beta=0)

            R_contrasted_img = cv2.cvtColor(contrasted_img, cv2.COLOR_GRAY2RGB)
            to_write = R_contrasted_img
        case _:
            to_write = gray1_img

    cv2.imwrite(save_enhanced_dir+"/"+where+name(filename, where), to_write)
    print("Successfully processed "+filename+" using "+what)

def name(filename, where):
    outname = filename.replace(SMALL_data_dir+"/"+where, '')
    return outname

def imgBarker(what):
    benigns = [file for file in glob.glob(SMALL_data_dir+"/benigns/*.png")]
    cancers = [file for file in glob.glob(SMALL_data_dir+"/cancers/*.png")]
    normals = [file for file in glob.glob(SMALL_data_dir+"/normals/*.png")]

    print("Found "+str(len(benigns))+" items in "+SMALL_data_dir+"/benigns")
    print("Found "+str(len(cancers))+" items in "+SMALL_data_dir+"/cancers")
    print("Found "+str(len(normals))+" items in "+SMALL_data_dir+"/normals")

    
    for image in benigns:
        # print(save_enhanced_dir+"/benigns"+name(image, "benigns"))
        mammoEnhance(image, "benigns", what)
    for image in cancers:
        mammoEnhance(image, "cancers", what)
    for image in normals:
        mammoEnhance(image, "normals", what)

def traintest():

    #   train_datagen = tf.data.Dataset(
    #   preprocessing_function=preprocess_input,
    #   shear_range=0.2,
    #   zoom_range=0.2,
    #   horizontal_flip=True)

    train_generator = tf.keras.utils.image_dataset_from_directory(
        save_enhanced_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=False, # FOR INITIAL TESTING
        validation_split=0.10, # FOR INITIAL TESTING
        subset="training", # FOR INITIAL TESTING
        image_size=(img_height, img_width)
        ) # training data, returns Dataset as BatchDataset

    """test_generator = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=1,
        image_size=(img_height, img_width)
        ) # test v data"""

    base_model = ResNet152V2(include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, input_dim=3, activation='relu')(x)
    x = preprocess_input(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss='categorical_crossentropy', 
        metrics = [
            'mean_squared_error',
            'categorical_crossentropy',
            'categorical_accuracy',
            mtr.Accuracy(),
            mtr.FalseNegatives(),
            mtr.Precision(),
            mtr.Recall()
            ]
        ) 
    history = model.fit(
        train_generator, 
        epochs = epochs
    #    ,callbacks = [learning_rate_reduction]
        )

    global resultkeys
    resultkeys = list(history.history.keys())

    model.save('D:/Documents/Adamson University/2022-2023 SEM1/CSRP2/DDSM_small-dataset/ResNet152_Mammo.h5')
    #model.evaluate(test_generator, verbose=2)

    return history.history

#imgBarker("gs")
#results_dict["gs"] = traintest()
#imgBarker("cl")
#results_dict["cl"] = traintest()
#imgBarker("wi")
results_dict["wi"] = traintest()
#imgBarker("co")
#results_dict["co"] = traintest()

print(resultkeys)

for which in results_dict:
    print(which, ': ')
    for key in resultkeys:
        print(key, ' = ', results_dict[which][key][epochs-1])



