from architectures.helpers.constants import threshold
from architectures.helpers.model_handler import get_model
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import selected_model

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf


MODEL_PATH = "1741609964-tl8-pd64-p8-e"
THRESHOLD = threshold
hyperparameters = hyperparameters[selected_model]
i = 2  # Using epoch 2 which we know exists
run = "neat-planet-23"  # This might not be needed with the new path structure


def load_dataset():
    x_test = []
    y_test = []
    for etf in etf_list:
        x_test.extend(np.load(f"ETF/strategy/{threshold}/TestData/x_{etf}.npy"))
        y_test.extend(np.load(f"ETF/strategy/{threshold}/TestData/y_{etf}.npy"))
    return x_test, y_test


def make_datasets(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(hyperparameters["batch_size"])
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_finalized_datasets(x_test, y_test):
    test_dataset = make_datasets(x_test, y_test)
    return test_dataset


x_test, y_test = load_dataset()
test_dataset = get_finalized_datasets(x_test, y_test)
model = get_model()

# Build the model by calling it on some dummy data
input_shape = hyperparameters["input_shape"]
dummy_input = np.zeros((1, *input_shape))
_ = model(dummy_input)  # This will build the model

# Try to load the weights
try:
    # First try to load the weights from the specified path
    if hasattr(model, 'inner_model'):
        model.inner_model.load_weights(
            f"saved_models/{selected_model}/{THRESHOLD}/{run}/{MODEL_PATH}{i}.h5")
    else:
        model.load_weights(
            f"saved_models/{selected_model}/{THRESHOLD}/{run}/{MODEL_PATH}{i}.h5")
    print(f"Successfully loaded weights from specific path")
except Exception as e:
    print(f"Error loading weights from specific path: {e}")
    # Try loading from a more general path
    try:
        if hasattr(model, 'inner_model'):
            model.inner_model.load_weights(
                f"saved_models/{selected_model}/{THRESHOLD}/{MODEL_PATH}{i}.weights.h5")
        else:
            model.load_weights(
                f"saved_models/{selected_model}/{THRESHOLD}/{MODEL_PATH}{i}.weights.h5")
        print(f"Successfully loaded weights from general path")
    except Exception as e2:
        print(f"Error loading weights from general path: {e2}")
        # Try loading the model directly
        try:
            model = tf.keras.models.load_model(f"saved_models/{selected_model}/{THRESHOLD}/model.keras")
            print(f"Successfully loaded full model")
        except Exception as e3:
            print(f"Error loading full model: {e3}")
            print("Using untrained model")

model.evaluate(test_dataset)
predictions = model.predict(test_dataset)
classes = np.argmax(predictions, axis=1)
cf = confusion_matrix(y_test, classes)
print(f"\n{cf}")
cr = classification_report(y_test, classes)
print(f"\n{cr}")
f1 = f1_score(y_test, classes, average='weighted')
print(f"\nF1 score: {f1}")
rc = recall_score(y_test, classes, average='weighted')
print(f"\nRecall score: {rc}")
pr = precision_score(y_test, classes, average='weighted')
print(f"\nPrecision score: {pr}")
