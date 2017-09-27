from skimage.data import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bson
import io
import sys

DATA_DIR = './data/'

def parse_train_example():

    # train_example_file = open(DATA_DIR + 'train_example.bson', 'rb')
    # data = bson.loads(train_example_file.read())
    #
    train_example = bson.decode_file_iter(open(DATA_DIR + 'train_example.bson', 'rb'))
    # data = bson.decode_document(open(DATA_DIR + 'train_example.bson', 'rb'))

    data = []
    for key, value in enumerate(train_example):
        product_id = value['_id']
        category_id = value['category_id']  # This won't be in Test data
        # prod_to_category[product_id] = category_id
        pics = []
        for e, pic in enumerate(value['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            # do something with the picture, etc
            # plt.imshow(picture)
            # plt.title(category_id)
            # plt.show()
            pics.append(picture)
        data.append((product_id, category_id, pics))

    return data

def get_training_data():

    data = parse_train_example()

    images = []
    classes = []
    for triplet in data:
        for image in triplet[2]:
            images.append(image)
            classes.append(triplet[1])

    unique, return_inverse = np.unique(classes, return_inverse=True)
    labels = []
    for index in return_inverse:
        target = np.zeros(shape=np.max(return_inverse)+1)
        target[index] = 1
        labels.append(target)

    return np.array(images), np.array(labels)

if __name__ == '__main__':

    # parse_train_example()
    get_training_data()