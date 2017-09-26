import bson
import sys

DATA_DIR = './data/'

def show_data():

    train_example_file = open(DATA_DIR + 'train_example.bson')
    train_example = bson.loads(train_example_file.read())

    for key, value in train_example.iteritems():
        print(key)
        print(value)
        # sys.exit('')

if __name__ == '__main__':

    show_data()