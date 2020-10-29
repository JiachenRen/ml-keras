import use_plaidml_backend
from sklearn.model_selection import train_test_split
from conv_models import LettersConvModel
from dataset_utils import expand_dataset, load_dataset

print("Loading & processing dataset...")
(data, labels) = load_dataset("letters_dataset", label_idx=8)

# Split the data
print("Splitting dataset...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, shuffle=True)
print("Initial training dataset size: %s" % x_train.shape[0])

# Expand the training dataset
print("Expanding training dataset... this may take a while...")
(x_train_ex, y_train_ex) = expand_dataset((x_train, y_train))
print("Expanded training dataset size: %s" % x_train_ex.shape[0])

# Train using my custom dataset!
model = LettersConvModel((x_train_ex, y_train_ex), (x_test, y_test))
model.train_and_save()
model.export_as_mlmodel()
