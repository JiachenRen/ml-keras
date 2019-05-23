from base_model import BaseModel
from keras.datasets import mnist

# Load MNIST data
training_data, testing_data = mnist.load_data()

# Create, fit, evaluate, and then save the specified model.
model = BaseModel(training_data, testing_data)
model.run()
