from process import extract_diabetic_retinopathy_stage_from_image, train_or_load_diabetic_retinopathy_stage_recognition_model
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_PATH = '..' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep
TEST_DATASET_PATH = '..' + os.path.sep + 'dataset' + os.path.sep + 'test' + os.path.sep


# indeksiranje labela za brzu pretragu
# priprema skupa podataka za metodu za treniranje
train_image_paths = []
train_image_labels = []
diseases_classes = []
for root, dirs, files in os.walk(TRAIN_DATASET_PATH):
    path = root.split(os.sep)
    for directory in dirs:
        diseases_classes.append(directory)
    for file in files:
        if len(path) == 4:
            train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, path[3], file))
            if path[3] == 'adenocarcinoma':
                train_image_labels.append(0)
            elif path[3] == 'large.cell.carcinoma':
                train_image_labels.append(1)
            elif path[3] == 'normal':
                train_image_labels.append(2)
            elif path[3] == 'squamous.cell.carcinoma':
                train_image_labels.append(3)
            # train_image_labels.append(int(path[3]))


# istrenirati model za prepoznavanje faze dijabetesne retinopatije
model = train_or_load_diabetic_retinopathy_stage_recognition_model(train_image_paths, train_image_labels, len(diseases_classes))


# izvrsiti odredjivanje faze bolesti sa svih fotografija iz test skupa podataka, koriscenjem istreniranog modela
processed_image_names = []
test_image_paths = []
test_image_labels = []
for root, dirs, files in os.walk(TEST_DATASET_PATH):
    path = root.split(os.sep)
    for file in files:
        if len(path) == 4:
            processed_image_names.append(file)
            image_path = os.path.join(TEST_DATASET_PATH, path[3], file)
            test_image_paths.append(image_path)
            if path[3] == 'adenocarcinoma':
                test_image_labels.append(0)
            elif path[3] == 'large.cell.carcinoma':
                test_image_labels.append(1)
            elif path[3] == 'normal':
                test_image_labels.append(2)
            elif path[3] == 'squamous.cell.carcinoma':
                test_image_labels.append(3)
            # test_image_labels.append(int(path[3]))

extracted_retinopathy_stages = extract_diabetic_retinopathy_stage_from_image(model, test_image_paths, test_image_labels)

# Statistika
test_image_labels_str = []
for label in test_image_labels:
    test_image_labels_str.append(diseases_classes[label])
extracted_retinopathy_stages_str = []
for stage in extracted_retinopathy_stages:
    extracted_retinopathy_stages_str.append(diseases_classes[stage])

# Confusion matrix
conf_mat = confusion_matrix(test_image_labels_str, extracted_retinopathy_stages_str, labels=diseases_classes)
print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=diseases_classes)
disp.plot(cmap="Blues", values_format='', xticks_rotation=90)
plt.show()

percentage = accuracy_score(test_image_labels_str, extracted_retinopathy_stages_str)*100
print('Accuracy:', percentage, '%')

print(classification_report(y_true=list(map(str, test_image_labels_str)),
                            y_pred=list(map(str, extracted_retinopathy_stages_str)),
                            target_names=list(map(str, diseases_classes))))


# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
'''result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, extracted_retinopathy_stages[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)'''
