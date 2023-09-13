import time as time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.applications import resnet_v2
from keras.layers import  Flatten, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')


N_CLASSES = 4
IMG_SIZE = 224
N_EPOCHS = 50


class ResNet152:

    def __init__(self, train, val, test, saved_model_name) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.saved_model_name = saved_model_name

    def get_results(self):

        print("Instantiating ResNet 152V2")

        baseModel = resnet_v2.ResNet152V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(1024, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.7)(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.7)(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.3)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        #Freeze base layers
        print("Freezing ResNet152V2 base layers")

        for layer in baseModel.layers:
            layer.trainable = False
            
        opt = optimizers.Adam(lr=0.0003, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        #Warm-up
        print("Warm-up: firing neurons")
        history = model.fit(self.train, epochs=N_EPOCHS, validation_data=self.val)

        #Unfreeze layers
        print("Unfreezing ResNet152V2 layers")
        for layer in baseModel.layers[400:]:
            layer.trainable = True

        opt = optimizers.Adam(lr=0.00003, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        #Fine-tuning
        print("Fine-tuning ResNet152V2")
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

