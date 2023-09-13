import time as time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.applications import vgg16
from keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

N_CLASSES = 4
IMG_SIZE = 224
N_EPOCHS = 50

class VGG16:

    def __init__(self, train, val, test, saved_model_name) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.saved_model_name = saved_model_name

    def get_results(self):

        print("Instantiating VGG16")

        baseModel = vgg16.VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax")(headModel)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        #Freeze base layers
        print("Freezing VGG16 base layers")
        for layer in baseModel.layers:
            layer.trainable = False
            
        opt = SGD(lr=1e-4, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])
        
        #Warm up
        print("Warm-up: firing neurons")
        history = model.fit(self.train, epochs=N_EPOCHS, validation_data=self.val)
        
        #Unfreeze layers
        print("Unfreezing VGG16 layers")
        for layer in baseModel.layers[15:]:
            layer.trainable = True
                
        model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])
        
        #Fine-tuning
        print("Fine-tuning VGG16")
        t0 = time.time()
        model_name = self.saved_model_name+".h5"
        callback_params   = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
            ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min')
        ]

        history = model.fit(self.train, epochs=N_EPOCHS, callbacks= callback_params, validation_data=self.val)

        print("Model training time: ", int(time.time()-t0), 's')

        preds = model.evaluate_generator(self.test, workers=1)
        K.clear_session()
        del model
        return preds
        
        
    


    

        
                
        
