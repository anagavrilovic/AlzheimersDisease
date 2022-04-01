# import libraries here
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from joblib import dump, load

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

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
    image = cv2.resize(image, (64, 64))
    return image


def create_cnn():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(64, 64, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(64, 64, 1), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_or_load_diabetic_retinopathy_stage_recognition_model(train_image_paths, train_image_labels):
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

        model = create_cnn()

        train_images = np.asarray(images)
        train_image_labels = np.asarray(train_image_labels)

        model.fit(train_images, train_image_labels, batch_size=32, epochs=20)
        model.save('cnn.h5')
        
    return model


def extract_diabetic_retinopathy_stage_from_image(trained_model, image_path):
    """
    :param trained_model: <Model> Istrenirani model za prepoznavanje faze dijabetesne retinopatije
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati fazu dijabetesne retinopatije
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: '0', '1', '2', '3', '4')
    """
    retinopathy_stage = ""

    image = load_image(image_path)
    image = resize_image(image)
    image = np.reshape(image, image.shape + (1,))
    # display_image(img)

    # image = np.asarray(image)[0]
    retinopathy_stage = trained_model.predict(image)

    print(image_path, '\t\t', retinopathy_stage)

    return retinopathy_stage
