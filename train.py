from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from data import get_training_data

def build_model():

    kernel_size = 4
    conv_depth_1 = 50
    pool_size = 3
    drop_prob_1 = 0.3
    conv_depth_2 = 70
    hidden_size = 1000
    drop_prob_2 = 0.3
    num_classes = 36

    inp = Input(shape=(180, 180, 3))  # depth goes last in TensorFlow back-end (first in Theano)
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers

    return model

def train():

    batch_size = 10
    num_epochs = 5

    model = build_model()
    X_train, Y_train = get_training_data()


    model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                  optimizer='adam',  # using the Adam optimiser
                  metrics=['accuracy'])  # reporting the accuracy

    model.fit(X_train, Y_train,  # Train the model using the training set...
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)  # ...holding out 10% of the data for validation
    # model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

if __name__ == '__main__':

    train()