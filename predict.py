from keras.datasets import mnist
from conv_models import ConvModel

# Load testing data
training_data, testing_data = mnist.load_data()

# Load model from disk and run prediction demonstration
model = ConvModel(training_data, testing_data)
model.load_from_disk()
model.demo()
