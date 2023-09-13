import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.models import Model
from vit_keras import vit, utils
from numpy.linalg import norm
from sklearn.decomposition import PCA
from glob import glob


target_ls = ['ceo', 'engineer', 'nurse', 'school_teacher']

path_ls = ['./IIAS/Obscured/CEO/*',
              './IIAS/Obscured/Engineer/*',
              './IIAS/Obscured/Nurse/*',
              './IIAS/Obscured/SchoolTeacher/*']


def extract_features_cnn(img_path, model, target_size, preprocess_input):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(model.predict(x))

def extract_features_vit(img_path, model, target_size):
    img = utils.read(img_path, target_size)
    return model.predict(vit.preprocess_inputs(img)[np.newaxis])

def reduce_dimensions(features, model_name):
    pca = PCA()
    if model_name == 'vgg_16':
        features = features.reshape(7,7*512)
    elif model_name == 'inception':
        features = features.reshape(50,2*512)
    elif model_name == 'resnet':
        features = features.reshape(7,7*2048)
    elif model_name == 'xception':
        features = features.reshape(7,7*2048)
    
    pca.fit(features)
    features_trans = pca.transform(features)
    if model_name == 'vgg_16':
        features_trans_reshaped = np.squeeze(features_trans.reshape(1,49))
    elif model_name == 'inception':
        features_trans_reshaped = np.squeeze(features_trans.reshape(1,2500))
    elif model_name == 'resnet':
        features_trans_reshaped = np.squeeze(features_trans.reshape(1,49))
    elif model_name == 'xception':
        features_trans_reshaped = np.squeeze(features_trans.reshape(1,49))
    
    return(features_trans_reshaped)

def img_sim_score(features_1, features_2):
    sim = (np.dot(features_1,features_2))/(norm(features_1,2)*norm(features_2,2))
    return(sim) 


def iias_cnn(target_path, target_size, model, preprocess_input, model_name):
    man_images = glob('./IIAS/Unobscured/Man/*')
    woman_images = glob('./IIAS/Unobscured/Woman/*')
    man_feat_ls = []
    woman_feat_ls = []
    iias_ls = []
    
    for i in range(10):
        man_feat_ls.append(reduce_dimensions(extract_features_cnn(man_images[i], model, target_size, preprocess_input), model_name))
        woman_feat_ls.append(reduce_dimensions(extract_features_cnn(woman_images[i], model, target_size, preprocess_input), model_name))
    
    target_feat = reduce_dimensions(extract_features_cnn(target_path, model, target_size, preprocess_input), model_name)
    
    for i in range(10):
        iias_ls.append(img_sim_score(man_feat_ls[i],target_feat) - img_sim_score(woman_feat_ls[i],target_feat))
    
    return np.mean(iias_ls)

def iias_vit(target_path, target_size, model):
    man_images = glob('./IIAS/Unobscured/Man/*')
    woman_images = glob('./IIAS/Unobscured/Woman/*')
    man_feat_ls = []
    woman_feat_ls = []
    iias_ls = []
    
    for i in range(10):
        man_feat_ls.append(extract_features_vit(man_images[i], model, target_size))
        woman_feat_ls.append(extract_features_vit(woman_images[i], model, target_size))
    
    target_feat = np.squeeze(extract_features_vit(target_path, model, target_size))
    
    for i in range(10):
        iias_ls.append(img_sim_score(np.squeeze(man_feat_ls[i]),target_feat) - img_sim_score(np.squeeze(woman_feat_ls[i]),target_feat))
    
    return np.mean(iias_ls)

def cnn_iias_calculator(biased_model, unbiased_model, preprocess_input, model_name):

    target_size = target_size = (224,224,3)
    biased_ls = []
    unbiased_ls = []
    
    for path in path_ls:
        biased = []
        unbiased = []
        for img in glob(path):
            biased.append(iias_cnn(img, target_size, biased_model, preprocess_input, model_name))
            unbiased.append(iias_cnn(img, target_size, unbiased_model, preprocess_input, model_name))
            
        biased_ls.append(np.mean(biased))
        unbiased_ls.append(np.mean(unbiased))
    
    return list([biased_ls , unbiased_ls])

def vit_iias_calculator(biased_model, unbiased_model, target_size):
    
    biased_ls = []
    unbiased_ls = []
    
    for path in path_ls:
        biased = []
        unbiased = []
        for img in glob(path):
            biased.append(iias_vit(img, target_size, biased_model))
            unbiased.append(iias_vit(img, target_size, unbiased_model))
            
        biased_ls.append(np.mean(biased))
        unbiased_ls.append(np.mean(unbiased))
    
    return list([biased_ls , unbiased_ls])

