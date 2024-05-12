import numpy as np
import os
import librosa
import config
from utils import float32_to_int16
import soundfile as sf
def main():
    fl_dataset = os.path.join(config.fl_dataset, "audio", "eval", "public")
    fl_files = os.listdir(fl_dataset)
    output_dir = os.path.join(config.fl_dataset, "audio", "eval", "resample")
    output_dict = []
    for f in fl_files:
        y, sr = librosa.load(os.path.join(fl_dataset, f), sr = config.sample_rate)
        sf.write(os.path.join(output_dir, f), y, sr)
        print(f, sr, float32_to_int16(y))
        temp_dict = {
            "audio_name": f,
            "waveform": float32_to_int16(y)
        }
        output_dict.append(temp_dict)
    npy_file = os.path.join(config.fl_dataset, "eval.npy")
    np.save(npy_file, output_dict)



if __name__ == '__main__':
    main()
