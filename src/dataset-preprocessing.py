import shutil
import json
import librosa
import soundfile as sf
import pathlib
import numpy as np

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


def split_audio(audio, fs, file_path, seg=2, stride=0.25, target_dir="split_dir"):
    target_dir.mkdir(exist_ok=True, parents=True)
    # Get number of samples for 2 seconds; replace 2 by any number
    tmp_samples = seg * fs

    samples_total = len(audio)
    start_sample = int(0.2 * seg * fs)
    stride_samples = int(stride * tmp_samples)
    i = 0

    while start_sample < (samples_total - tmp_samples):

        start_sample = i * int(stride_samples)

        # check if the buffer is not exceeding total samples
        if tmp_samples > (samples_total - start_sample):
            start_sample = samples_total - tmp_samples

        end_sample = int(start_sample + tmp_samples)

        block = audio[start_sample: end_sample]
        fname = f"{file_path.stem}_block_{str(i).zfill(2)}.wav"
        sf.write(target_dir / fname, block, fs)
        i += 1


def create_dataset():
    print("Organizing data")

    DATA_PATH = 'data/nus-mix'
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
        audio, fss = librosa.core.load(fpath)
        audio = np.float64(audio)
        fpath = fpath.parent / f"{fpath.stem}_silence.wav"
        split_audio(audio, fss, fpath, stride=1, seg=2, target_dir=target_dir / "silence")


