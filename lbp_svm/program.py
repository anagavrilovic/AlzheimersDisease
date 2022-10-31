import numpy as np
import cv2
import os

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from joblib import dump, load
from localbinarypatterns import LocalBinaryPatterns
import helper

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


# Oversampling
sm = SMOTE(random_state=42)
train_images, train_image_labels = sm.fit_resample(train_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3),
                                                   train_image_labels)
train_images = train_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

test_images, test_image_labels = sm.fit_resample(test_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3),
                                                 test_image_labels)
test_images = test_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


# Shuffle data
train_images, train_image_labels = shuffle(train_images, train_image_labels)


# Create and train model
try:
    clf_svm = load('models/svm.joblib')
except:
    train_image_features = []

    desc = LocalBinaryPatterns(30, 1)

    for img in train_images:
        img = helper.reshape_data(img)
        hist = desc.describe(img)
        train_image_features.append(hist)
    print("LBP done and SVM started")

    x = np.array(train_image_features)
    y = np.array(train_image_labels)

    clf_svm = LinearSVC(C=100.0, random_state=42, verbose=1, max_iter=20000)
    clf_svm.fit(x, y)
    print("SVM done")

    dump(clf_svm, 'models/svm.joblib')


# Predict
test_image_features = []
desc_test = LocalBinaryPatterns(40, 2)

for img in test_images:
    img = helper.reshape_data(img)
    hist = desc_test.describe(img)
    test_image_features.append(hist)
print("LBP done for test images")

x = np.array(test_image_features)
y = np.array(test_image_labels)

predictions = clf_svm.predict(x)
print(predictions)

# Statistics
test_image_labels_str = []
for label in test_image_labels:
    test_image_labels_str.append(classes[int(label)])
predicted_labels_str = []
for label in predictions:
    predicted_labels_str.append(classes[int(label)])


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
