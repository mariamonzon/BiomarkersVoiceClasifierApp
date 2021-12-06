import os
import pathlib
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
SEED =24
tf.random.set_seed(SEED)
np.random.seed(SEED)
import shutil
import json
import librosa
import soundfile as sf
from src.simple1D import create_model

def copy_files(src_path, dst_path):
    pathlib.Path(dst_path).parent.mkdir(exist_ok=True, parents=True)

    try:
        shutil.copy(src_path, dst_path)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except:
        print("Error occurred while copying file.")


def organized_dataset(data_dir, target_dir):
    labels = {}
    for f in data_dir.rglob('*.wav'):
        label = f.parent.name
        singer = f.parts[-3]
        fname = f"{singer}_{label}_{f.name}"
        copy_files(f, target_dir / fname)
        with open(f"{(target_dir / fname).with_suffix('.txt')}", "w") as lbl:
            lbl.write(f"{label}\n")
        labels[str(fname)] = label
    json.dump(labels, open(f"{target_dir}/labels.txt", 'w'))

def split_audio(audio, fs, file_path, seg = 2, stride =0.25, target_dir ="split_dir"):
    target_dir.mkdir(exist_ok=True, parents=True)
    # Get number of samples for 2 seconds; replace 2 by any number
    tmp_samples = seg * fs

    samples_total = len(audio)
    start_sample = 0
    stride_samples = int(stride*seg * fs)
    i = 1

    while start_sample < (samples_total-tmp_samples):

        # check if the buffer is not exceeding total samples
        if tmp_samples > (samples_total - start_sample):
            start_sample = samples_total - tmp_samples

        start_sample  =  int(start_sample + stride_samples)
        end_sample = int(start_sample + tmp_samples)

        block = audio[start_sample: end_sample]
        fname =  f"{file_path.stem}_block_{str(i).zfill(2)}.wav"
        sf.write(target_dir/fname, block, fs)
        i += 1


def create_dataset():
    global FS
    DATA_PATH = 'data/sing-read'
    TARGET_PATH = 'dataset'
    data_dir = pathlib.Path(DATA_PATH)
    target_dir = pathlib.Path(TARGET_PATH)
    target_dir.mkdir(exist_ok=True, parents=True)
    # organized_dataset(data_dir, target_dir)
    FS = 44100
    for fpath in data_dir.rglob('*.wav'):
        audio, fs = librosa.core.load(fpath, sr=FS)
        audio = np.float64(audio)
        if "read" in str(fpath):
            dir = target_dir / "speech"
        else:
            dir = target_dir / "sing"
        split_audio(audio, fs, fpath, seg=2, target_dir=dir)

    for fpath in pathlib.Path("data/ESC-50/audio").rglob('*.wav'):
        audio, fs = librosa.core.load(fpath)
        audio = np.float64(audio)
        fpath = fpath.parent / f"{fpath.stem}_silence.wav"
        split_audio(audio, fs, fpath, stride=1, seg=2, target_dir=target_dir / "silence")

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler


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
    print("Development of App")
    # create_dataset()
    PATH = 'dataset'
    dataset = []
    labels= []
    create_dataset()
    labels_map = {"silence": 0, "sing":1, "speech": 2}
    for fpath in pathlib.Path(PATH).rglob('*.wav'):
        audio, fs = librosa.core.load(fpath)
        label = labels_map.get(str(fpath.parts[-2]), None)
        dataset.append(audio)
        labels.append(label)

    X, X_test, y, y_test = train_test_split(dataset, labels, test_size = 0.2, random_state = SEED)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = SEED)
    windlow = 44100*2
    model = create_model(init_shape=(windlow, 1), n_class=3)
    callbacks = [LearningRateScheduler(StepDecay)]
    model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                  batch_size=64, epochs=50)#, callbacks=callbacks, verbose=1)
