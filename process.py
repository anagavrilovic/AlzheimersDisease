# import libraries here
from imutils import face_utils
import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def resize_image(image):
    image = cv2.resize(image, (400, 450), interpolation=cv2.INTER_AREA)
    return image


def define_hog(shape):
    nbins = 9  # broj binova
    cell_size = (20, 20)  # broj piksela po celiji
    block_size = (8, 8)  # broj celija po bloku

    hog = cv2.HOGDescriptor(
        _winSize=(shape[1] // cell_size[1] * cell_size[1], shape[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins)

    return nbins, cell_size, block_size, hog


def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran. 
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu). 
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran

    try:
        clf_svm = load('svm.joblib')
    except:
        images = []
        for image_path in train_image_paths:
            # img = crop_face(image_path)
            img = load_image(image_path)
            if img.shape[0] != 762 and img.shape[1] != 562:
                img = cv2.resize(img, (562, 762), interpolation=cv2.INTER_AREA)
            img = img[200:700, 60:500]
            img = resize_image(img)
            # plt.imshow(img)
            # plt.show()
            images.append(img)

        image_features = []
        nbins, cell_size, block_size, hog = define_hog(images[0].shape)

        for img in images:
            hog_comp = hog.compute(img)
            image_features.append(hog_comp)
            print(hog_comp)

        x = np.array(image_features)
        y = np.array(train_image_labels)

        # print('Train shape: ', x.shape, y.shape)
        x = reshape_data(x)

        clf_svm = SVC(kernel='linear', probability=True, verbose=True)
        clf_svm.fit(x, y)

        dump(clf_svm, 'svm.joblib')
        y_train_pred = clf_svm.predict(x)
        print("Train accuracy: ", accuracy_score(y, y_train_pred))

    return clf_svm


def crop_face(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = None
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        # shape predstavlja 68 koordinata
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz

        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if x + w > image.shape[1] or y + h > image.shape[0]:
            continue

        around_h = h // 4
        around_w = w // 8

        y_from = y - around_h
        y_to = y + h + around_h
        x_from = x - around_w
        x_to = x + w + around_w

        if y_from < 0:
            y_from = 0
        if y_to > image.shape[0]:
            y_to = image.shape[0]
        if x_from < 0:
            x_from = 0
        if x_to > image.shape[1]:
            x_to = image.shape[1]

        img = gray[y_from:y_to, x_from:x_to]

    if img is not None:
        # plt.imshow(img)
        # plt.show()
        return img
    return gray


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    facial_expression = ""
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    image = load_image(image_path)

    if image.shape[0] != 762 and image.shape[1] != 562:
        image = crop_face(image_path)
    else:
        image = image[200:700, 60:500]

    image = resize_image(image)
    # plt.imshow(image)
    # plt.show()

    nbins, cell_size, block_size, hog = define_hog(image.shape)

    image_feature = hog.compute(image)
    x = np.array(image_feature)
    x = x.transpose()
    facial_expression = trained_model.predict(x)

    print(image_path, ' ', facial_expression)

    return facial_expression[0]
