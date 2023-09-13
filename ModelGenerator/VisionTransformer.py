
import time as time

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from vit_keras import vit

import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

"""
Keras implementation of Vision Transformer: Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
Keras Implementation by faustomorales: https://github.com/faustomorales/vit-keras. Transfer learning using weights pretrained on Imagenet.

"""

N_CLASSES = 4
N_EPOCHS = 100

class VisionTransformer:

    def __init__(self, transformer_type, train, val, test, saved_model_name) -> None:

        self.transformer_type = transformer_type
        self.train = train
        self.val = val
        self.test = test
        self.saved_model_name = saved_model_name

    def create_model(self, vit_model):

        print("Instatiating Vision Transformer: "+ self.transformer_type)
        
        model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(11, activation = tfa.activations.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(N_CLASSES, 'softmax')
        ],
        name = 'vision_transformer')
        
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

        model.summary()
        
        return model
    
    def model_selector(self):

        if self.transformer_type == 'vit_l32':
            vit_model = vit.vit_l32(
            image_size = 224,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = N_CLASSES)
        elif self.transformer_type == 'vit_l16':
            vit_model = vit.vit_l16(
            image_size = 112,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = N_CLASSES)
        elif self.transformer_type == 'vit_b32':
            vit_model = vit.vit_b32(
            image_size = 224,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = N_CLASSES)
        elif self.transformer_type == 'vit_b16':
            vit_model = vit.vit_b16(
            image_size = 112,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = N_CLASSES)
        
        return vit_model

    def fine_tune(self):

        model = self.create_model(self.model_selector())
        model_name = self.saved_model_name+'.h5'

        t0 = time.time()
        callback_params   = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
            ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min')
        ]

        history = model.fit(self.train, epochs=N_EPOCHS, callbacks= callback_params, validation_data=self.val)

        print("Model training time: ", int(time.time()-t0), 's')

        return model

    def get_results(self):
        model = self.fine_tune()
        preds = model.evaluate(self.test, workers=1)
        K.clear_session()
        del model
        return preds
