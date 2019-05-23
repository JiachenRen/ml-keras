from __future__ import division
from keras.models import model_from_json
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load testing data
_, (x_test, y_test) = mnist.load_data()

cnn_model_name = input("Enter the name of cnn model: ")

# Load & compile model from disk
print("Loading")
json_file = open('models/%s/model.json' % cnn_model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/%s/model.h5" % cnn_model_name)
print("Loaded model from disk.")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Make predictions on test data set.

indices = list(range(0, x_test.shape[0], 10))
np.random.shuffle(indices)
num_correct = 0
num_classified = 0
for q in indices:
    for i in range(10):
        idx = i + q
        x = x_test[idx]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x, cmap=plt.get_cmap('gray'))
        x = np.expand_dims(x, 2)
        predicted = loaded_model.predict_classes(np.expand_dims(x, 0))[0]
        isCorrect = predicted == y_test[idx]
        color = "green" if isCorrect else "red"
        plt.text(14, -5, "Predicted: %s" % predicted, horizontalalignment='center', fontsize=12, color=color)
        num_classified += 1
        num_correct += 1 if isCorrect else 0
    print("accuracy = %s" % (num_correct / num_classified))
    plt.show()
