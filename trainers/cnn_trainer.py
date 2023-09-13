import pandas as pd
import numpy as np
import itertools
import sys
import os

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from ModelGenerator import VGG16, ResNet152, Inception, Xception
# from VGG16 import VGG16
# from ResNet152 import ResNet152
# from Inception import Inception
# from Xception import Xception

import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 224

def data_generator(train_path, test_path):
    # create a new generator
    training_gen = ImageDataGenerator(rescale=1./255,
                                rotation_range=30,
                                width_shift_range=0.2, height_shift_range=0.2,
                                horizontal_flip=True, vertical_flip=True,
                                brightness_range=[0.1,1],
                                zoom_range=0.3,
                                validation_split=0.2
                                )

    test_gen = ImageDataGenerator(rescale=1./255)

    # load train data
    train = training_gen.flow_from_directory(train_path, class_mode="categorical", shuffle=True, batch_size=100, target_size=(IMG_SIZE, IMG_SIZE), subset='training')
    # load val data
    val = training_gen.flow_from_directory(train_path,  class_mode="categorical", shuffle=True, batch_size=100, target_size=(IMG_SIZE, IMG_SIZE), subset='validation')
    #load test data
    test = test_gen.flow_from_directory(test_path,  class_mode="categorical", shuffle=False, batch_size=50, target_size=(IMG_SIZE, IMG_SIZE))

    return train, val, test

def unbiased(model_name, saved_model_name):
    train_path = "Gender_Dataset/train/balanced/"
    test_path = "Gender_Dataset/test/"
    train, val, test = data_generator(train_path=train_path, test_path=test_path)

    if model_name == "VGG16":
        model = VGG16.VGG16(train, val, test, saved_model_name)
    elif model_name == "ResNet152":
        model = ResNet152.ResNet152(train, val, test, saved_model_name)
    elif model_name == "Inception":
        model = Inception.Inception(train, val, test, saved_model_name)
    elif model_name == "Xception":
        model = Xception.Xception(train, val, test, saved_model_name)

    return model.get_results()

def biased(model_name, saved_model_name):
    train_path = "Gender_Dataset/train/imbalanced/"
    test_path = "Gender_Dataset/test/"
    train, val, test = data_generator(train_path=train_path, test_path=test_path)
    if model_name == "VGG16":
        model = VGG16.VGG16(train, val, test, saved_model_name)
    elif model_name == "ResNet152":
        model = ResNet152.ResNet152(train, val, test, saved_model_name)
    elif model_name == "Inception":
        model = Inception.Inception(train, val, test, saved_model_name)
    elif model_name == "Xception":
        model = Xception.Xception(train, val, test, saved_model_name)

    return model.get_results()

def result_generator():

    models = ['VGG16', 'ResNet152', 'Inception', 'Xception']


    for model in models:
            model_ls = []
            model_name_biased_ls = []
            model_name_unbiased_ls = []
            acc_biased_ls = []
            acc_unbiased_ls = []
            bias_diff = []
            pc_bias_diff = []

            for i in range(1,6):

                model_ls.append(model)

                #Unbiased
                print("Starting unbiased training for "+model+" iteration: "+str(i))
                model_name = "models/cnn/unbiased/"+model+"unbiased_"+str(i)
                res_unbiased = unbiased(model_name=model, saved_model_name=model_name)
                model_name_unbiased_ls.append(model+"_unbiased_"+str(i))
                acc_unbiased = res_unbiased[1]
                acc_unbiased_ls.append(acc_unbiased)

                #biased
                print("Starting biased training for "+model+" iteration: "+str(i))
                model_name = "models/cnn/biased/"+model+"biased_"+str(i)
                res_biased = biased(model_name=model, saved_model_name=model_name)
                model_name_biased_ls.append(model+"_biased_"+str(i))
                acc_biased = res_biased[1]
                acc_biased_ls.append(acc_biased)

                #bias difference
                bias_diff.append(abs(acc_unbiased-acc_biased))
                pc_bias_diff.append((abs(acc_unbiased-acc_biased)/acc_unbiased)*100)

            res_file = pd.DataFrame(list(zip(model_ls, model_name_unbiased_ls, acc_unbiased_ls, model_name_biased_ls, acc_biased_ls, bias_diff, pc_bias_diff)),
                                    columns=['Model', 'Model_Name_Unbiased', 'Accuracy_Unbiased', 'Model_Name_Biased', 'Accuracy_Biased', 'Bias_Difference', 'Percent_Bias_Diff'])
            
            res_file.to_csv("./results/cnn/"+model+".csv",index=False)

