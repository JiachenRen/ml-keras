import numpy
from model import Model
from keras.datasets import mnist

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load MNIST data
training_data, testing_data = mnist.load_data()

# Create, fit, evaluate, and then save the specified model.
model_name = input("Please enter the name of the model to train: ")
model = Model(model_name, training_data, testing_data)
model.run()
