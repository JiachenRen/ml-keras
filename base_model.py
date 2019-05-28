from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils
import matplotlib.pyplot as plt
import coremltools
import numpy as np


class BaseModel:

    def __init__(self, training_data, testing_data, epochs=10, batch_size=200):
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_test = testing_data[0]
        self.y_test = testing_data[1]
        self.epochs = epochs
        self.batch_size = batch_size

        self._normalize_input()
        self._one_hot_encode_labels()

        self.input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.num_pixels = self.input_shape[0] * self.input_shape[1]
        self.num_classes = self.y_test.shape[1]
        self.keras_model = self._make_model()

        # metadata
        self.name = "base_model"
        self.output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.short_description = 'Digit Recognition with MNIST'
        self.input_description = 'An image of a handwritten digit'
        self.output_description = 'Prediction of digit from 0 ~ 9'

    # Normalize inputs from 0-255 to 0-1
    def _normalize_input(self):
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

    # One hot encode outputs
    def _one_hot_encode_labels(self):
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)

    def _make_model(self):
        # Setup
        model = Sequential()
        model.add(Dense(self.num_pixels, input_dim=self.num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dense(self.num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _prepare_data(self):
        # Flatten 28*28 images to a 784 vector for each image
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.num_pixels).astype('float32')
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.num_pixels).astype('float32')

    # Train/fit the model
    def train(self):
        print("Training " + self.name)
        self._prepare_data()
        self.keras_model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2)

    def evaluate(self):
        print("Evaluating...")
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

    def export_as_mlmodel(self):
        base_url = "models/%s/" % self.name
        self.load_from_disk()
        h5_url = base_url + "digit_recognition.h5"
        self.keras_model.save(h5_url)
        model = coremltools.converters.keras.convert(
            h5_url,
            input_names=['image'],
            output_names=['output'],
            class_labels=self.output_labels,
            image_input_names='image'
        )

        # Set model meta data
        model.author = 'Jiachen Ren'
        model.short_description = self.short_description
        model.input_description['image'] = self.input_description
        model.output_description['output'] = self.output_description

        # Export model
        model.save(base_url + 'digit_recognition.mlmodel')

    def train_and_save(self):
        self.train()
        self.evaluate()
        self.serialize()

    def load_from_disk(self):
        # Load & compile model from disk
        print("Loading %s from disk..." % self.name)
        json_file = open('models/%s/model.json' % self.name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("models/%s/model.h5" % self.name)
        print("Loaded model from disk.")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.keras_model = loaded_model

    def _testing_img_at(self, idx):
        return self.x_test[idx].reshape(self.input_shape[0], self.input_shape[1])

    def run_demo(self):
        self._prepare_data()
        indices = list(range(0, self.x_test.shape[0], 10))
        np.random.shuffle(indices)
        num_correct = 0
        num_classified = 0
        for q in indices:
            for i in range(10):
                idx = i + q
                x = self.x_test[idx]
                img = self._testing_img_at(idx)
                plt.subplot(2, 5, i + 1)
                plt.imshow(img, cmap=plt.get_cmap('gray'))
                predicted = self.keras_model.predict_classes(np.expand_dims(x, axis=0))[0]
                is_correct = self.y_test[idx][predicted] == 1
                color = "green" if is_correct else "red"
                plt.text(14, -5, "Predicted: %s" % predicted, horizontalalignment='center', fontsize=12, color=color)
                num_classified += 1
                num_correct += 1 if is_correct else 0
            print("accuracy = %s" % (num_correct / num_classified))
            plt.show()
