import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report

from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model
from architectures.helpers.wandb_handler import initialize_wandb
from architectures.helpers.custom_callbacks import CustomCallback

from architectures.helpers.model_handler import get_model

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import wandb
from wandb.integration.keras import WandbCallback

hyperparameters = hyperparameters[selected_model]
t = time.time()
epoch_counter = 1

''' Dataset Preperation
'''


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for etf in etf_list:
        x_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/x_{etf}.npy"))
        y_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/y_{etf}.npy"))
        x_test.extend(
            np.load(f"ETF/strategy/{threshold}/TestData/x_{etf}.npy"))
        y_test.extend(
            np.load(f"ETF/strategy/{threshold}/TestData/y_{etf}.npy"))
    x_train_new = []
    y_train_new = []
    for x_t, y_t in zip(x_train, y_train):
        if y_t != 1:
            x_train_new.append(x_t)
            y_train_new.append(y_t)
            x_train_new.append(x_t)
            y_train_new.append(y_t)

    x_train.extend(x_train_new)
    y_train.extend(y_train_new)
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return x_train, y_train, x_test, y_test


def prepare_dataset(x_train, y_train, x_test):
    val_split = 0.1

    val_indices = int(len(x_train) * val_split)
    new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
    x_val, y_val = x_train[:val_indices], y_train[:val_indices]

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    return new_x_train, new_y_train, x_val, y_val


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(hyperparameters["batch_size"] * 10)
    dataset = dataset.batch(hyperparameters["batch_size"])
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
    val_dataset = make_datasets(x_val, y_val)
    test_dataset = make_datasets(x_test, y_test)
    return train_dataset, val_dataset, test_dataset


def run_experiment(model, test_dataset):
    # Create a ModelCheckpoint callback
    checkpoint_path = f"saved_models/{selected_model}/{threshold}/model.keras"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    callback_list = [
        CustomCallback(test_dataset, epoch_counter, t, y_test),
        checkpoint_callback
    ]
    
    if hyperparameters["learning_rate_type"] != "WarmUpCosine" and hyperparameters["learning_rate_type"] != "Not found":
        callback_list.append(hyperparameters["learning_rate_scheduler"])

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=hyperparameters["num_epochs"],
        callbacks=callback_list,
    )

    # Evaluate the model
    metrics = model.evaluate(test_dataset)
    
    # Print test metrics - handle the specific structure returned
    try:
        # The metrics structure is [loss_tensor, {metric_dict}]
        if isinstance(metrics, list) and len(metrics) > 1:
            loss = metrics[0]
            if isinstance(metrics[1], dict) and 'accuracy' in metrics[1]:
                accuracy = metrics[1]['accuracy']
            else:
                # If second element is not a dict or doesn't have 'accuracy'
                accuracy = metrics[1] if len(metrics) > 1 else 0
        else:
            # Fallback if structure is different
            loss = metrics if not isinstance(metrics, (list, dict)) else 0
            accuracy = 0
            
        # Convert tensors to numpy if needed
        if hasattr(loss, 'numpy'):
            loss = loss.numpy()
        if hasattr(accuracy, 'numpy'):
            accuracy = accuracy.numpy()
            
        print(f"Test accuracy: {round(float(accuracy) * 100, 2)}%, Test loss: {round(float(loss), 4)}")
    except Exception as e:
        print(f"Error processing metrics: {e}")
        print(f"Raw metrics: {metrics}")

    return history, model


if __name__ == "__main__":
    initialize_wandb()
    x_train, y_train, x_test, y_test = load_dataset()
    new_x_train, new_y_train, x_val, y_val = prepare_dataset(
        x_train, y_train, x_test)
    train_dataset, val_dataset, test_dataset = get_finalized_datasets(
        new_x_train, new_y_train, x_val, y_val, x_test, y_test)
    model = get_model()
    history, trained_model = run_experiment(model, test_dataset)
    #save edilecek,isimlendirme tutarlı olmalı

    predictions = trained_model.predict(test_dataset)
    classes = np.argmax(predictions, axis=1)
    cf = confusion_matrix(y_test, classes)
    print(f"\n{cf}")
    cr = classification_report(y_test, classes)
    print(f"\n{cr}")
    f1 = f1_score(y_test, classes, average='micro')
    print(f"\nF1 score: {f1}")
