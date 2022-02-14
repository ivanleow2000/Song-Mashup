# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 07:50:51 2022
TITLE: Method 1
@author: Ivan Leow
"""

# Import packages
import os
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import soundfile as sf


### Songs ###
Vocals = "alot"
Background = "hotelcalifornia"



### Prep ###
# Set working directory
os.chdir(r"C:/Users/61435/Downloads/Others/Personal projects/Song mashup")

# Prep converters
AudioSegment.converter = os.getcwd() + "\\ffmpeg.exe"
AudioSegment.ffprobe   = os.getcwd()+ "\\ffprobe.exe"



### Extract background music ###
# Read in audio file and separate stereo to mono tracks
sound_stereo = AudioSegment.from_file(os.getcwd() + f"\\Method1_location\\{Background}.mp3", format="mp3")
sound_monoL = sound_stereo.split_to_mono()[0]
sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
n_background = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
n_background.export(os.getcwd() + f"\\Method1_location\\{Background}_bg_extracted.mp3", format="mp3")



### Extract vocals ###
y, sr = librosa.load(os.getcwd() + f"\\Method1_location\\{Vocals}.mp3", duration=230)
S_full, phase = librosa.magphase(librosa.stft(y))
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

n_vocals = librosa.istft(S_foreground*phase)

# Export extracted vocals mp3
sf.write(f"Method1_location\\{Vocals}_vc_extracted.wav", n_vocals, sr)



### Merge Background and Vocals ###
# Read in extracted vocal mp3 file as audio segment type, background is already in this type
n_vocals_mp3 = AudioSegment.from_file(os.getcwd() + f"\\Method1_location\\{Vocals}_vc_extracted.wav", format = 'mp3')

# Adjust volume of vocals (vocals are often too soft compared to background music, increase)
n_vocals_mp3 = n_vocals_mp3 + 10

# Adjust volume of background
n_background = n_background - 8

# Merge
merged_mp3 = n_vocals_mp3.overlay(n_background)

# Export final mp3 file
merged_mp3.export(os.getcwd() + f"\\Method1_location\\{Background}_{Vocals}.mp3", format="mp3")
