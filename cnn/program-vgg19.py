import numpy as np
import cv2
import os

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.applications import VGG19
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


DATASET_PATH = '..' + os.path.sep + 'dataset'
IMAGE_SIZE = (128, 128)


# Read image paths and labels and write to dictionary
images_dict = dict()
classes = []
for root, dirs, files in os.walk(DATASET_PATH):
    path = root.split(os.sep)
    for directory in dirs:
        classes.append(directory)

    for file in files:
        if len(path) == 3:
            if path[2] == 'MildDemented':
                images_dict[os.path.join(DATASET_PATH, path[2], file)] = 0
            elif path[2] == 'ModerateDemented':
                images_dict[os.path.join(DATASET_PATH, path[2], file)] = 1
            elif path[2] == 'NonDemented':
                images_dict[os.path.join(DATASET_PATH, path[2], file)] = 2
            elif path[2] == 'VeryMildDemented':
                images_dict[os.path.join(DATASET_PATH, path[2], file)] = 3

class_number = len(classes)


# Load images
train_images = []
train_image_labels = []
for image_path, image_label in images_dict.items():
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    train_images.append(img)
    train_image_labels.append(image_label)
    print(image_path)

train_images = np.array(train_images, dtype=float)
train_image_labels = np.array(train_image_labels, dtype=float)


# Split to train and test
train_images, test_images, train_image_labels, test_image_labels = train_test_split(
    train_images,
    train_image_labels,
    test_size=0.2,
    random_state=15,
    stratify=train_image_labels)

train_images = train_images / 255
test_images = test_images / 255
print("Train test split")


# Oversampling test data
sm = SMOTE(random_state=42)

test_images, test_image_labels = sm.fit_resample(test_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), test_image_labels)
test_images = test_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


# Create and train model
try:
    model = load_model('models-vgg19/cnn.h5')
except:

    # Oversampling train data
    train_images, train_image_labels = sm.fit_resample(train_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), train_image_labels)
    train_images = train_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    # Shuffle train data
    train_images, train_image_labels = shuffle(train_images, train_image_labels)

    vgg19 = VGG19(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=IMAGE_SIZE+(3,),
        pooling=None,
        classes=1000
    )
    vgg19.trainable = False

    model = Sequential()
    model.add(vgg19)
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=class_number, activation='softmax'))
    model.summary()

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_image_labels, epochs=20, verbose=1)
    model.save('models-vgg19/cnn.h5')

    # Plotting accuracy and loss during training
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Evaluate and predict
results = model.evaluate(test_images, test_image_labels)
print("test loss, test acc:", results)

predictions = model.predict(test_images)
print(predictions)

predicted_labels = []
for prediction in predictions:
    predicted_value = max(prediction)
    index = np.where(prediction == predicted_value)[0][0]
    predicted_labels.append(index)


# Statistics
test_image_labels_str = []
for label in test_image_labels:
    test_image_labels_str.append(classes[int(label)])
predicted_labels_str = []
for label in predicted_labels:
    predicted_labels_str.append(classes[label])

# Confusion matrix
conf_mat = confusion_matrix(test_image_labels_str, predicted_labels_str, labels=classes)
print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
disp.plot(cmap="Blues", values_format='', xticks_rotation=90)
plt.title('Alzheimer\'s Disease Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show()

# Accuracy
percentage = accuracy_score(test_image_labels_str, predicted_labels_str)*100
print('Accuracy:', percentage, '%')

# Classification report
print(classification_report(y_true=list(map(str, test_image_labels_str)),
                            y_pred=list(map(str, predicted_labels_str)),
                            target_names=list(map(str, classes))))
