import os
import pathlib
import numpy as np
import tensorflow as tf
# Set the seed value for experiment reproducibility.
SEED =24
tf.random.set_seed(SEED)
np.random.seed(SEED)
import shutil
import json
import librosa
import soundfile as sf
from src.simple1D import create_model
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import keras

class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)

if __name__=='__main__':

    PATH = 'dataset'
    dataset = []
    labels=[]

    labels_map = {"silence": [1,0,0], "sing": [0,1,0], "speech":  [0,0,1]}

    # Place tensors on the CPU

    for fpath in pathlib.Path(PATH).rglob('*.wav'):

        audio, fs = librosa.core.load(fpath)
        label = labels_map.get(str(fpath.parts[-2]), None)
        dataset.append(audio)
        labels.append(label)

    X, X_test, y, y_test = train_test_split(np.array(dataset), np.array(labels), test_size = 0.2, random_state = SEED)
    X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size = 0.1, random_state = SEED)

    # Load TFLite model and allocate tensors.
    with open('classificaiton_model.tflite', 'rb') as f:
        tflite_model = f.read()

    # get predict prob and label
    ypred = tflite_model.predict(X_test, verbose=1)
    ypred = np.argmax(ypred, axis=1)

    print(classification_report(np.argmax(y_test, axis=1), ypred, target_names=['silence', "sing", 'speech']))
    cm = confusion_matrix(np.argmax(y_test, axis=1), ypred)

    # Display the Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_map.keys())
    disp.plot(xticks_rotation='vertical', cmap=plt.cm.Greens)
    disp.savefig('confusion_matrix_green.jpg')
    plt.matshow(cm)
    plt.title('Problem 1: Confusion Matrix Audio Clasifier Recognition')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix.jpg')
    #model.fit(X, y, epochs=50, batch_size=20)
    # test_file = tf.io.read_file(TARGET_PATH + '/ADIZ/read/01.wav')
    # test_audio, _ = tf.audio.decode_wav(contents=test_file)
    # test_audio.shape

    # wav = tf.squeeze(test_audio, axis=-1)
