import numpy as np
import cv2
import os

import tensorflow_hub as hub

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# PRETRAINED_MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" 74.31%
PRETRAINED_MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5"
DATASET_PATH = '..' + os.path.sep + 'dataset'
IMAGE_SIZE = (224, 224)


# Read image paths and labels and write to dictionary
train_images_dict = dict()
classes = []
for root, dirs, files in os.walk(DATASET_PATH):
    path = root.split(os.sep)
    for directory in dirs:
        classes.append(directory)

    for file in files:
        if len(path) == 3:
            if path[2] == 'MildDemented':
                train_images_dict[os.path.join(DATASET_PATH, path[2], file)] = 0
            elif path[2] == 'ModerateDemented':
                train_images_dict[os.path.join(DATASET_PATH, path[2], file)] = 1
            elif path[2] == 'NonDemented':
                train_images_dict[os.path.join(DATASET_PATH, path[2], file)] = 2
            elif path[2] == 'VeryMildDemented':
                train_images_dict[os.path.join(DATASET_PATH, path[2], file)] = 3

class_number = len(classes)


# Load images
train_images = []
train_image_labels = []
for image_path, image_label in train_images_dict.items():
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    train_images.append(img)
    train_image_labels.append(image_label)
    print(image_path)

train_images = np.array(train_images, dtype=float)
train_image_labels = np.array(train_image_labels, dtype=float)

# Oversampling
sm = SMOTE(random_state=42)
train_images, train_image_labels = sm.fit_resample(train_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), train_image_labels)
train_images = train_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


# Shuffle data
train_images, train_image_labels = shuffle(train_images, train_image_labels)


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


# Create and train model
try:
    model = load_model('cnn.h5')
except:
    vgg19 = VGG19(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=IMAGE_SIZE+(3,),
        pooling=None,
        classes=1000
    )
    vgg19.trainable = False

    pretrained_model = Sequential([
        hub.KerasLayer(PRETRAINED_MODEL_URL, input_shape=IMAGE_SIZE+(3,), trainable=False)
    ])

    model = Sequential()
    model.add(vgg19)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=class_number, activation='softmax'))
    model.summary()

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_image_labels, epochs=10, verbose=1)
    model.save('cnn.h5')


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
    test_image_labels_str.append(classes[label])
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

print(classification_report(y_true=list(map(str, test_image_labels_str)),
                            y_pred=list(map(str, predicted_labels_str)),
                            target_names=list(map(str, classes))))
