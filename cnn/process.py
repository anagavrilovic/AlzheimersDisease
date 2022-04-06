# import libraries here
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

'''from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model'''


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def resize_image(image):
    image = cv2.resize(image, (256, 256))
    return image


def create_cnn(class_number):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(256, 256, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=class_number, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def train_or_load_diabetic_retinopathy_stage_recognition_model(train_image_paths, train_image_labels, class_number):
    """
    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """

    train_image_paths, train_image_labels = shuffle(np.array(train_image_paths), np.array(train_image_labels))

    try:
        model = load_model('cnn.h5')
    except:
        images = []
        for image_path in train_image_paths:
            img = load_image(image_path)
            img = resize_image(img)
            img = np.reshape(img, img.shape + (1, ))
            # display_image(img)
            images.append(img)
        print("Images loaded")

        model = create_cnn(class_number)

        train_images = np.asarray(images)
        train_image_labels = np.asarray(train_image_labels)

        model.fit(train_images, train_image_labels, batch_size=32, epochs=15)
        model.save('cnn.h5')

        pd.DataFrame(model.history.history).plot()
        
    return model


def extract_diabetic_retinopathy_stage_from_image(trained_model, test_image_paths, test_image_labels):
    """
    :param trained_model: <Model> Istrenirani model za prepoznavanje faze dijabetesne retinopatije
    :param test_image_paths: <String> Putanja do fotografije sa koje treba prepoznati fazu dijabetesne retinopatije
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: '0', '1', '2', '3', '4')
    """

    images = []
    for image_path in test_image_paths:
        img = load_image(image_path)
        img = resize_image(img)
        img = np.reshape(img, img.shape + (1,))
        # display_image(img)
        images.append(img)

    predicted_retinopathy_stages_probabilities = trained_model.predict(np.asarray(images))

    predicted_retinopathy_stages = []
    for probability in predicted_retinopathy_stages_probabilities:
        index = np.where(probability == max(probability))[0][0]
        predicted_retinopathy_stages.append(index)

    test_loss, test_acc = trained_model.evaluate(np.array(images), np.array(test_image_labels), verbose=2)
    print(test_acc)

    return predicted_retinopathy_stages
