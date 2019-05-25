from os import listdir
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from conv_models import CustomConvModel
import numpy as np


# Load my own dataset
def load_dataset(directory):
    images = []
    labels = []
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(60, 60))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # delete red and green channels; we don't need them
        image = np.delete(image, np.s_[0:2], 2)
        # remove the extra dimension
        image = image.squeeze()
        # invert the image - images in my dataset have white backgrounds;
        invert = np.vectorize(lambda x: 255 - x)
        image = invert(image)
        images.append(image)
        # get image label
        label = int(name.split('.')[0][6])
        labels.append(label)

    return np.array(images), np.array(labels)


print("Loading & processing dataset...")
(data, labels) = load_dataset("dataset")

# Split the data
print("Splitting dataset...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, shuffle=True)

# Train using my custom dataset!
model = CustomConvModel((x_train, y_train), (x_test, y_test))
model.train_and_save()
model.export_as_mlmodel()
