from process import extract_diabetic_retinopathy_stage_from_image, train_or_load_diabetic_retinopathy_stage_recognition_model
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '..' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '..' + os.path.sep + 'dataset' + os.path.sep + 'test' + os.path.sep
# -------------------------------------------------------------------

# indeksiranje labela za brzu pretragu
# priprema skupa podataka za metodu za treniranje
train_image_paths = []
train_image_labels = []
label_dict = dict()

for root, dirs, files in os.walk(TRAIN_DATASET_PATH):
    path = root.split(os.sep)
    for file in files:
        if len(path) == 4:
            label_dict[file] = path[3]
            train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, path[3], file))
            train_image_labels.append(int(path[3]))

# istrenirati model za prepoznavanje faze dijabetesne retinopatije
model = train_or_load_diabetic_retinopathy_stage_recognition_model(train_image_paths, train_image_labels)

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka, koriscenjem istreniranog modela
processed_image_names = []
extracted_retinopathy_stages = []
test_image_paths = []

for root, dirs, files in os.walk(VALIDATION_DATASET_PATH):
    path = root.split(os.sep)
    for file in files:
        if len(path) == 4:
            processed_image_names.append(file)
            image_path = os.path.join(VALIDATION_DATASET_PATH, path[3], file)
            test_image_paths.append(image_path)

extracted_retinopathy_stage_probabilities = extract_diabetic_retinopathy_stage_from_image(model, test_image_paths)

for probability in extracted_retinopathy_stage_probabilities:
    index = probability.index(max(probability))
    extracted_retinopathy_stages.append(index)

# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, extracted_retinopathy_stages[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
