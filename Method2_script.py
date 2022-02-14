# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:10:46 2022
TITLE: Method 2
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
os.chdir(r"C:\\Users\\61435\\Downloads\\Others\\Personal projects\\Song mashup")

# Prep converters
AudioSegment.converter = os.getcwd() + "\\ffmpeg.exe"
AudioSegment.ffprobe   = os.getcwd()+ "\\ffprobe.exe"



### Extract background music ###
y, sr = librosa.load(os.getcwd() + f"\\Method2_location\\{Background}.mp3", duration=230)
print(f"Sampling rate of {Background}.mp3 is {sr}.")

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
#
# To avoid being biased by local continuity, we constrain similar frames to be
# separated by at least 2 seconds.
#
# This suppresses sparse/non-repetetitive deviations from the average spectrum,
# and works well to discard vocal elements.

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

n_background = librosa.istft(S_background*phase)

# Export extracted background mp3
# CHANGE SR HERE TO SPEED OR SLOW DOWN TO MATCH TEMPO OF THE TWO SONGS
sf.write(f"Method2_location\\{Background}_bg_extracted.wav", n_background, sr)



### Extract vocals ###
y, sr = librosa.load(os.getcwd() + f"\\Method2_location\\{Vocals}.mp3", duration=230)
print(f"Sampling rate of {Vocals}.mp3 is {sr}.")
S_full, phase = librosa.magphase(librosa.stft(y))
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2
mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)
mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)
S_foreground = mask_v * S_full
S_background = mask_i * S_full

n_vocals = librosa.istft(S_foreground*phase)

# Export extracted vocals mp3
# CHANGE SR HERE TO SPEED OR SLOW DOWN TO MATCH TEMPO OF THE TWO SONGS
sf.write(f"Method2_location\\{Vocals}_vc_extracted.wav", n_vocals, sr)



### Merge Background and Vocals ###
# Read in extracted background and vocal mp3 files as audio segment type
n_background_mp3 = AudioSegment.from_file(os.getcwd() + f"\\Method2_location\\{Background}_bg_extracted.wav", format = 'mp3')
n_vocals_mp3 = AudioSegment.from_file(os.getcwd() + f"\\Method2_location\\{Vocals}_vc_extracted.wav", format = 'mp3')

# Increase volume of vocals (vocals are often too soft compared to background music, ignore if fine)
n_vocals_mp3 = n_vocals_mp3 + 6

# Merge
merged_mp3 = n_vocals_mp3.overlay(n_background_mp3)

# Export final mp3 file
merged_mp3.export(os.getcwd() + f"\\Method2_location\\{Background}_{Vocals}.mp3", format="mp3")
