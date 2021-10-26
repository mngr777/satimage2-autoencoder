#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='Data file path')
    parser.add_argument('model_file', help='Trained autoencoder model file path')
    parser.add_argument('--bottleneck', type=int, default=16, help='Encoder output size (integer)')
    return parser.parse_args()

def main():
    # Get arguments
    args = parse_args()

    # Load data
    data = scipy.io.loadmat(args.data_file)
    if not ('X' in data and data['X'].shape[1] == 36):
        print('Invalid data file')
        return

    # Create model
    autoenc = Sequential()
    autoenc.add(Dense(args.bottleneck, activation='relu', name='encoder', input_shape=(36,)))
    autoenc.add(Dense(36, activation='sigmoid', name='decoder'))
    autoenc.compile(loss='mean_squared_error', optimizer=Adam())

    # Print model summary
    print(autoenc.summary())

    normal_rows = (np.where(data['y'] == 0)[0],)
    data_x = data['X'][normal_rows]
    data_x = data_x / 255

    print(data_x.shape)

    # Train
    history = History()
    autoenc.fit(
        data_x,
        data_x,
        batch_size=100,
        epochs=15,
        verbose=1,
        validation_split=0.3,
        callbacks=[history])

    # Plot stat. history
    plt.title('Accuracy')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Save model
    autoenc.save(args.model_file)

if __name__ == '__main__':
    main()
