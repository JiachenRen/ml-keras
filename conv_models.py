from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.utils import np_utils

from base_model import BaseModel


class ConvModel(BaseModel):

    def __init__(self, training_data, testing_data, epochs=10, batch_size=200):
        super().__init__(training_data, testing_data, epochs, batch_size)
        self.name = "cnn_baseline"

    # Baseline model for CNN with one convolution layer, max pooling, and dropout.
    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _prepare_data(self):
        self.x_train = np.expand_dims(self.x_train, 3)
        self.x_test = np.expand_dims(self.x_test, 3)

    def _testing_img_at(self, idx):
        return self.x_test[idx].squeeze()


def lr_scheduler(epoch, lr):
    if epoch < 5:
        lr = 0.005
    elif epoch <= 10:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


class CustomConvModel(ConvModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data, 20, 200)
        self.name = "cnn_custom1"

    def train(self):
        print("Training " + self.name)
        self._prepare_data()
        self.keras_model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[LearningRateScheduler(lr_scheduler, verbose=1)]
        )

    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(20, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class LargeConvModel(ConvModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data)
        self.name = "cnn_large"

    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class SuperConvModel(ConvModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data, epochs=10, batch_size=20)
        self.name = "cnn_super"

    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Conv2D(20, (5, 5), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(10, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class LettersConvModel(ConvModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data, epochs=5, batch_size=200)
        self.name = "cnn_letter"
        self.input_description = "An image of capital letter"
        self.output_description = "Prediction of letter from A to Z"
        self.short_description = 'Predicts a handwritten capital letter'
        self.output_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def _one_hot_encode_labels(self):
        to_ints = np.vectorize(lambda x: ord(x.lower()) - 97)
        def encode(y): return np_utils.to_categorical(to_ints(y), 26)
        self.y_train = encode(self.y_train)
        self.y_test = encode(self.y_test)

    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Conv2D(30, (10, 10), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(20, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
