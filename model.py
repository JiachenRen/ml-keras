from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten
from keras.utils import np_utils
import numpy as np


class Model:

    def __init__(self, name, training_data, testing_data):
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_test = testing_data[0]
        self.y_test = testing_data[1]

        # Normalize inputs from 0-255 to 0-1
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        # One hot encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)

        self.input_shape = (28, 28)
        self.num_classes = self.y_test.shape[1]
        self.name = name

        if name == "baseline":
            self.keras_model = self.baseline_model()
        elif name == "cnn_baseline":
            self.keras_model = self.baseline_cnn_model()
        elif name == "cnn_large":
            self.keras_model = self.large_cnn_model()
        elif name == "cnn_custom1":
            self.keras_model = self.custom_cnn_model1()
        else:
            raise Exception("Model %s does not exist." % name)

    # Baseline model with 1 hidden fully connected layer
    def baseline_model(self):
        num_pixels = self.input_shape[0] * self.input_shape[1]

        # Setup
        model = Sequential()
        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dense(self.num_classes, kernel_initializer='normal', activation='softmax'))

        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Baseline model for CNN with one convolution layer, max pooling, and dropout.
    def baseline_cnn_model(self):

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

    # CNN model with state of the art performance (I made slight adjustments)
    def large_cnn_model(self):

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

    # Custom CNN model
    def custom_cnn_model1(self):
        model = Sequential()
        model.add(Conv2D(10, (20, 20), input_shape=(self.input_shape[0], self.input_shape[1], 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Train/fit the model
    def train(self):
        print("Training " + self.name)
        if self.name == "baseline":
            # Flatten 28*28 images to a 784 vector for each image
            num_pixels = self.x_train.shape[1] * self.x_train.shape[2]
            self.x_train = self.x_train.reshape(self.x_train.shape[0], num_pixels).astype('float32')
            self.x_test = self.x_test.reshape(self.x_test.shape[0], num_pixels).astype('float32')
        elif self.name == "cnn_baseline" or self.name == "cnn_large" or self.name == "cnn_custom1":
            self.x_train = np.expand_dims(self.x_train, 3)
            self.x_test = np.expand_dims(self.x_test, 3)

        self.keras_model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=10,
            batch_size=200,
            verbose=2)

    def evaluate(self):
        scores = self.keras_model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Error: %.2f%%" % (100 - scores[1] * 100))

    # Save model to disk
    def serialize(self):
        print("Serializing...")
        model_json = self.keras_model.to_json()
        with open("models/%s/model.json" % self.name, "w") as json_file:
            json_file.write(model_json)
        self.keras_model.save_weights("models/%s/model.h5" % self.name)
        print("Saved model to disk")

    def run(self):
        self.train()
        self.evaluate()
        self.serialize()
