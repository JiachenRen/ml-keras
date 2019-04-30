from __future__ import division
from keras.models import model_from_json
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load testing data
_, (X_test, y_test) = mnist.load_data()

# Load & compile model from disk
print("Loading")
json_file = open('models/baseline/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/baseline/model.h5")
print("Loaded model from disk.")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Make predictions on test data set.
X_test_lin = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype('float32')
X_test_lin /= 255
indices = list(range(0, X_test.shape[0], 10))
np.random.shuffle(indices)
num_correct = 0
num_classified = 0
for q in indices:
    for i in range(10):
        idx = i + q
        x = X_test[idx]
        x_lin = X_test_lin[idx]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x, cmap=plt.get_cmap('gray'))
        predicted = loaded_model.predict_classes(np.expand_dims(x_lin, axis=0))[0]
        isCorrect = predicted == y_test[idx]
        color = "green" if isCorrect else "red"
        plt.text(14, -5, "Predicted: %s" % predicted, horizontalalignment='center', fontsize=12, color=color)
        num_classified += 1
        num_correct += 1 if isCorrect else 0
    print("accuracy = %s" % (num_correct / num_classified))
    plt.show()
