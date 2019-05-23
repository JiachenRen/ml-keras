from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from base_model import BaseModel


class ConvModel(BaseModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data)
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


class CustomConvModel(ConvModel):

    def __init__(self, training_data, testing_data):
        super().__init__(training_data, testing_data)
        self.name = "cnn_custom1"

    def _make_model(self):
        model = Sequential()
        model.add(Conv2D(10, (20, 20), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

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
