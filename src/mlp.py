import numpy as np
import pandas as pd
import time
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Embedding,
                                     Conv1D,
                                     Flatten,
                                     MaxPooling1D,
                                     Dense,
                                     Dropout,
                                     Activation,
                                     SpatialDropout1D,
                                     LSTM)

from tensorflow.keras import layers
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint

class MLP():
    def __init__(self):
        pass

    def prepare_data(self, data):
        y = data['DEATH_EVENT']
        X = data.drop(['DEATH_EVENT', 'time'], axis=1).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def _build_model(self):
        model = Sequential()

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
        # print(model.summary())

        return model
    
    def _get_callbacks(self):
        earlystopping = EarlyStopping(monitor='val_accuracy', patience=10)
        
        return earlystopping

    def fit(self, X_train, X_test, y_train, y_test, epochs=50):
        model = self._build_model()

        callbacks = self._get_callbacks()

        history = model.fit(X_train, y_train,
                            batch_size=4,
                            epochs=epochs,
                            validation_split=0.2,
                            callbacks=[callbacks])

        accuracies = history.history['val_accuracy']
        print(f'accuracies : {accuracies}')

        return model
        
