import torch
import torchaudio
import os
from torchaudio_augmentations import (
    RandomApply,
    Compose,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Reverse,
    LowPassFilter,
    HighPassFilter,
)

# Set the input directory path
input_directory = './mao'

# Set the output directory path
output_directory = './dataset_raw/mao_raw'

# Set the number of augmented files to generate for each input file
num_augmentations = 40

# Define a list of transformations to be applied to the audio file

counter = 1
# Iterate over each .wav file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        # Load the original audio file
        original_audio_path = os.path.join(input_directory, filename)
        y, sr = torchaudio.load(original_audio_path)
        transforms = Compose([
            RandomApply([PolarityInversion()], p=0.1),
            RandomApply([Gain()], p=0.3),
            RandomApply([HighPassFilter(sample_rate=sr, freq_low=200, freq_high=200)], p=0.2),
                ])
    
        os.makedirs(output_directory, exist_ok=True)
        
        # Apply the transformations to the audio file and save the resulting augmented files
        for i in range(num_augmentations):
            t_audio = transforms(y)
            t_name = f'{counter}.wav'
            t_audio_path = os.path.join(output_directory, t_name)
            torchaudio.save(t_audio_path, t_audio, sr)

            # Increment the counter
            counter += 1
