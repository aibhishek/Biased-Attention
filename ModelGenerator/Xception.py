import time as time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
#from tensorflow.keras.applications import Xception
from keras.layers import  Flatten, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

N_CLASSES = 4
IMG_SIZE = 224
N_EPOCHS = 50

class Xception:
    def __init__(self, train, val, test, saved_model_name) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.saved_model_name = saved_model_name

    def get_results(self):

        baseModel = tf.keras.applications.Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        

        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(1024, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        #Freeze base layers
        print("Freezing Xception base layers")
        for layer in baseModel.layers:
            layer.trainable = False
            
        opt = optimizers.Adam()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        #Warm-up
        print("Warm-up: firing neurons")
        history = model.fit_generator(self.train, epochs=N_EPOCHS, validation_data=self.val)

        #Unfreeze layers
        print("Unfreezing Xception layers")
        for layer in baseModel.layers[126:]:
            layer.trainable = True

        opt = optimizers.Adam(lr=1e-5)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        #Fine-tuning
        print("Fine-tuning Xception")
        model_name = self.saved_model_name+".h5"
        t0 = time.time()
        callback_params   = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
            ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min')
        ]

        history = model.fit_generator(self.train, epochs=N_EPOCHS, callbacks= callback_params, validation_data=self.val)

        print("Model training time: ", int(time.time()-t0), 's')

        preds = model.evaluate_generator(self.test, workers=1)
        K.clear_session()
        del model
        return preds

