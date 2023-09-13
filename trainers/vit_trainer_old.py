import pandas as pd
import numpy as np
import itertools
import sys
import os
import gc
import argparse

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from ModelGenerator import VisionTransformer

import warnings
warnings.filterwarnings('ignore')

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)



IMG_SIZE = 112

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

def unbiased(transformer_type, saved_model_name):
    train_path = "Gender_Dataset/train/balanced/"
    test_path = "Gender_Dataset/test/"
    train, val, test = data_generator(train_path=train_path, test_path=test_path)
    vit = VisionTransformer.VisionTransformer(transformer_type, train, val, test, saved_model_name)
    return vit.get_results()

def biased(transformer_type, saved_model_name):
    train_path = "Gender_Dataset/train/imbalanced/"
    test_path = "Gender_Dataset/test/"
    train, val, test = data_generator(train_path=train_path, test_path=test_path)
    vit = VisionTransformer.VisionTransformer(transformer_type, train, val, test, saved_model_name)
    return vit.get_results()

def result_generator(model, run):

    


    #for model in models:
    model_ls = []
    model_name_biased_ls = []
    model_name_unbiased_ls = []
    acc_biased_ls = []
    acc_unbiased_ls = []
    bias_diff = []
    pc_bias_diff = []
    
    K.clear_session()
    i = run 
    model = model 
    # for i in range(1):

    model_ls.append(model)

    #Unbiased
    model_name = "models/vit/unbiased/vit_"+model+"unbiased_"+str(i)
    res_unbiased = unbiased(transformer_type=model, saved_model_name=model_name)
    model_name_unbiased_ls.append(model_name)
    acc_unbiased = res_unbiased[1]
    acc_unbiased_ls.append(acc_unbiased)

    #biased
    model_name = "models/vit/biased/vit_"+model+"biased_"+str(i)
    res_biased = biased(transformer_type=model, saved_model_name=model_name)
    model_name_biased_ls.append(model_name)
    acc_biased = res_biased[1]
    acc_biased_ls.append(acc_biased)

    #bias difference
    bias_diff.append(abs(acc_unbiased-acc_biased))
    pc_bias_diff.append((abs(acc_unbiased-acc_biased)/acc_unbiased)*100)

    res_file = pd.DataFrame(list(zip(model_ls, model_name_unbiased_ls, acc_unbiased_ls, model_name_biased_ls, acc_biased_ls, bias_diff, pc_bias_diff)),
                            columns=['Model', 'Model_Name_Unbiased', 'Accuracy_Unbiased', 'Model_Name_Biased', 'Accuracy_Biased', 'Bias_Difference', 'Percent_Bias_Diff'])

    res_file.to_csv("./results/vit/"+model+".csv",index=False)

    K.clear_session()
    gc.collect()

            





