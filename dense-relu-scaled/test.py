#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow.keras.models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='Data file path')
    parser.add_argument('model_file', help='Trained autoencoder model file path')
    return parser.parse_args()

def load_model(path):
    return tensorflow.keras.models.load_model(path, compile=False)

def score(x, autoenc):
    x = (x / 255).reshape(1, 36)
    decoded = autoenc.predict(x)
    return np.linalg.norm(x - decoded)

def main():
    # Get arguments
    args = parse_args()

    # Load data
    data = scipy.io.loadmat(args.data_file)
    if not ('X' in data and data['X'].shape[1] == 36):
        print('Invalid data file')
        return

    # Load model
    autoenc = load_model(args.model_file)

    # Get norman/anomaly rows
    anomaly_rows = (np.where(data['y'] == 1)[0],)
    anomaly_num = anomaly_rows[0].shape[0]
    normal_rows = (np.where(data['y'] == 0)[0][0:anomaly_num],) # get same number as anomalies

    # Calc. scores for anomalies an sample of normal points
    normal_scores = np.apply_along_axis(score, 1, data['X'][normal_rows], autoenc)
    anomaly_scores = np.apply_along_axis(score, 1, data['X'][anomaly_rows], autoenc)

    # Show loss distributions
    plt.title('Compression loss')
    plt.xlabel('loss')
    plt.ylabel('')
    plt.hist(normal_scores, bins=20)
    plt.hist(anomaly_scores, bins=20)
    plt.legend(['normal', 'anomaly'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
