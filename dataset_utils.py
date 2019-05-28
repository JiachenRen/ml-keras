
import numpy as np
from os import listdir
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def expand_dataset(dataset, trans=(-5, 6, 2), rot=(-20, 20, 5)):
    (images, labels) = dataset
    new_images = []
    new_labels = []
    size = images.shape[0]
    count = 1
    for (image, label) in zip(images, labels):
        img = Image.fromarray(image)
        if count % 10 == 0:
            print("Processing %s of %s" % (count, size))
        # translate the image left, right, up, and down
        for i in range(*trans):
            for j in range(*trans):
                translated = translate(img, i, j)
                img_arr = np.array(translated)
                new_images.append(img_arr)
                new_labels.append(label)
        # rotate the image left and right by varying degrees
        for i in range(*rot):
            rotated = img.rotate(i)
            img_arr = np.array(rotated)
            new_images.append(img_arr)
            new_labels.append(label)
        count += 1
    return np.array(new_images), np.array(new_labels)


def translate(img, x, y):
    img = img.transform(img.size, Image.AFFINE, (1, 0, x, 0, 1, y))
    return img


# Load custom dataset from directory
def load_dataset(directory, label_idx, size=(40, 40)):
    images = []
    labels = []
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=size, interpolation='hamming')
        # plt.imshow(image, cmap=plt.get_cmap('gray'))
        # plt.show()
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
        label = name.split('.')[0][label_idx]
        labels.append(label)

    return np.array(images), np.array(labels)