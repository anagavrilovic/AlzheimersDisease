# import libraries here
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump, load
from localbinarypatterns import LocalBinaryPatterns
# import ann_functions


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def resize_image(image):
    image = cv2.resize(image, (256, 256))
    return image


'''def define_hog(shape):
    nbins = 9  # broj binova
    cell_size = (20, 20)  # broj piksela po celiji
    block_size = (8, 8)  # broj celija po bloku

    hog = cv2.HOGDescriptor(
        _winSize=(shape[1] // cell_size[1] * cell_size[1], shape[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins)

    return nbins, cell_size, block_size, hog'''


def train_or_load_diabetic_retinopathy_stage_recognition_model(train_image_paths, train_image_labels):
    """
    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """

    train_image_paths, train_image_labels = shuffle(np.array(train_image_paths), np.array(train_image_labels))

    try:
        clf_svm = load('svm.joblib')
    except:
        images = []
        for image_path in train_image_paths:
            img = load_image(image_path)
            img = resize_image(img)
            # plt.imshow(img)
            # plt.show()
            images.append(img)
        print("Images loaded")

        image_features = []

        '''nbins, cell_size, block_size, hog = define_hog(images[0].shape)

        for img in images:
            hog_comp = hog.compute(img)
            image_features.append(hog_comp)
            print(hog_comp)'''

        desc = LocalBinaryPatterns(16, 2)

        for img, image_path in zip(images, train_image_paths):
            hist = desc.describe(img)
            image_features.append(hist)
            print(image_path)
        print("LBP done and SVM started")

        x = np.array(image_features)
        y = np.array(train_image_labels)
        print(x.shape)
        print(x[0].shape)

        # print('Train shape: ', x.shape, y.shape)
        # x = reshape_data(x)

        clf_svm = LinearSVC(C=100.0, random_state=42, verbose=1, max_iter=10000)
        clf_svm.fit(x, y)
        print("SVM done")

        dump(clf_svm, 'svm.joblib')
        y_train_pred = clf_svm.predict(x)
        print("Train accuracy: ", accuracy_score(y, y_train_pred))
        
    return clf_svm

    '''model = ann_functions.load_trained_ann()

    if model is None:
        images = []
        for image_path in train_image_paths:
            img = load_image(image_path)
            img = resize_image(img)
            # plt.imshow(img)
            # plt.show()
            images.append(img)
        print("Images loaded")

        image_features = []

        desc = LocalBinaryPatterns(8, 1)
        for img, image_path in zip(images, train_image_paths):
            hist = desc.describe(img)
            image_features.append(hist)
            print(image_path)
        print("LBP done and SVM started")

        inputs = np.array(image_features).reshape(1, -1)
        outputs = np.array(train_image_labels).reshape(1, -1)

        print("Treniranje modela zapoceto.")
        model = ann_functions.create_ann(len(image_features))
        print("Kreirana")
        model = ann_functions.train_ann(model, inputs, outputs)
        print("Treniranje modela zavrseno.")
        ann_functions.serialize_ann(model)

    return model'''


def extract_diabetic_retinopathy_stage_from_image(trained_model, image_path):
    """
    :param trained_model: <Model> Istrenirani model za prepoznavanje faze dijabetesne retinopatije
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati fazu dijabetesne retinopatije
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: '0', '1', '2', '3', '4')
    """
    retinopathy_stage = ""

    image = load_image(image_path)
    image = resize_image(image)
    # plt.imshow(image)
    # plt.show()

    '''nbins, cell_size, block_size, hog = define_hog(image.shape)
    image_feature = hog.compute(image)'''

    desc = LocalBinaryPatterns(16, 2)
    hist = desc.describe(image)
    retinopathy_stage = trained_model.predict(hist.reshape(1, -1))

    '''x = np.array(image_feature)
    x = x.transpose()
    x = x.reshape((x.shape[0], 1))
    retinopathy_stage = trained_model.predict(x)'''

    print(image_path, '\t\t', retinopathy_stage)

    return retinopathy_stage[0]
