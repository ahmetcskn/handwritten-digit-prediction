"""Train MNIST from scratch only when explicitly called (or on first app startup)."""
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / 'digit_classifier.h5'


class DigitClassification:
    def __init__(self):
        (X_train, self.y_train), (X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.X_train = X_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        self.X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        return self.model.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2, verbose=2)

    def save_model(self, path=MODEL_PATH):
        self.model.save(str(path))
        print(f'Model successfully saved as {path}')

    def predict(self):
        return np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)


def main():
    classifier = DigitClassification()
    classifier.train()
    _, accuracy = classifier.model.evaluate(classifier.X_test, classifier.y_test, verbose=0)
    print(f'MNIST test accuracy: {accuracy:.2%}')
    classifier.save_model()


if __name__ == '__main__':
    main()
