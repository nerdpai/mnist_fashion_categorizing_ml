import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd


def normalizeData(data: np.ndarray):
    minimum = data.min()
    maximum = data.max()
    data = (data - minimum) / (maximum - minimum)
    return data


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = normalizeData(x_train)
    x_test = normalizeData(x_test)

    y_unique_count = len(np.unique(y_test))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(60, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(40, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(
                y_unique_count, activation=tf.keras.activations.softmax
            ),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
    )

    callback = tf.keras.callbacks.EarlyStopping(
        "val_sparse_categorical_accuracy", patience=25
    )

    model.fit(
        batch_size=128,
        epochs=500,
        callbacks=[callback],
        validation_split=0.1,
        verbose=0,
        x=x_train,
        y=y_train,
    )

    model.evaluate(x_test, y_test, batch_size=128, verbose=1)


if __name__ == "__main__":
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta", "history"))
    if not is_conda:
        raise EnvironmentError()
    main()
