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
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import copy

# from heatmaps import make_gradcam_heatmap, decode_predictions, crop_image, display_gradcam
from heatmaps2 import make_gradcam_heatmap, save_and_display_gradcam

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
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
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

test_image = copy.deepcopy(test_images[900])

train_images = train_images / 255
test_images = test_images / 255
print("Train test split")

# Oversampling test data
sm = SMOTE(random_state=42)

test_images, test_image_labels = sm.fit_resample(test_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1), test_image_labels)
test_images = test_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


# Create and train model
try:
    model = load_model('models/cnn.h5')

    # Oversampling train data
    train_images, train_image_labels = sm.fit_resample(train_images.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1),
                                                       train_image_labels)
    train_images = train_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    # Shuffle train data
    train_images, train_image_labels = shuffle(train_images, train_image_labels)
except:
    model = Sequential()

    model.add(Conv2D(input_shape=IMAGE_SIZE + (1,), filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

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
    model.save('models/cnn.h5')

    print(history.history.keys())

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


# Heat maps
image = copy.deepcopy(test_images[900])
image = np.expand_dims(image, axis=0)

model.layers[-1].activation = None

preds = model.predict(image)
heatmap = make_gradcam_heatmap(image, model, 'conv2d_2')

# display_gradcam(test_image, heatmap, preds, classes)

save_and_display_gradcam(test_image, heatmap, preds, classes)

''''INTENSITY = 0.5
heatmap = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
heatmapshow = None
heatmap = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)

image_with_heatmap = heatmap * INTENSITY + cv2.cvtColor(np.float32(crop_image(test_image)), cv2.COLOR_GRAY2RGB)

fig = plt.figure(figsize=(8, 5))
rows = 1
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title("Original image")

fig.add_subplot(rows, columns, 2)
plt.imshow(image_with_heatmap.astype(np.uint8))
plt.axis('off')
plt.title(decode_predictions(preds, classes))

plt.show()'''
