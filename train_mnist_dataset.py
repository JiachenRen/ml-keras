from conv_models import ConvModel
from keras.datasets import mnist

# Load MNIST data
training_data, testing_data = mnist.load_data()

# Create, fit, evaluate, and then save the specified model.
model = ConvModel(training_data, testing_data)
# model.train_and_save()
model.export_as_mlmodel()
