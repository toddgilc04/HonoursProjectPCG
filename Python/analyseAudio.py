
print("\nRunning Audio Analysis Script\n")

import librosa
import random
import os
import json
import numpy
from pathlib import Path
import sys

# set the path where audio is discovered
filePath = os.path.dirname(__file__)
audioPath = os.path.join(filePath, 'SONG_ANALYSE')

# assign where files will be looked for
files = os.listdir(audioPath)
# find any random file in folder
files = [f for f in files if os.path.isfile(os.path.join(audioPath, f))]
randomFile = random.choice(files)

# combine the file and the path 
fullAudioPath = os.path.join(audioPath, randomFile)
y, sr = librosa.load(fullAudioPath)


# beats is required, i think to ensure everything isnt stored in tempo and the data is instead split
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo = float(tempo.mean())
print('BPM: ' + format(tempo))


rms = librosa.feature.rms(y=y) # find rms of the audio (used as a loudness measurement)
rmsMin = rms.min()
rmsMax = rms.max()
rmsNormalise = (rms - rmsMin) / (rmsMax - rmsMin)    # normalise values so they are useable
# now have 100s of values per second of music
rmsMean = float(rmsNormalise.mean()) # overall loudness of track
print ('RMS Mean: ' + format(rmsMean))
# ideally find rms every x seconds


# brightness rating of audio track
specCentroid = librosa.feature.spectral_centroid(y=y, sr=sr)
specCentroidMean = float(specCentroid.mean())
print ('Spectral Centroid Mean: ' + format(specCentroidMean))
# potentially find centroid every x/y sec


# bass strength at loudest of track (found by restricting to low freq)
onsetEnv = librosa.onset.onset_strength(y=y, sr=sr, fmax = 1000,  n_mels=32)
onsetEnvMean = float(onsetEnv.mean()) 
print ('Onset Strength Mean: ' + format(onsetEnvMean))


# beat strength
onsetEnvHigherCap = librosa.onset.onset_strength(y=y, sr=sr, fmax = 8000,  n_mels=128)
beats = librosa.util.fix_frames(beats)
beatStrengths = onsetEnvHigherCap[beats]
beatStrengthsNormalise = (beatStrengths - numpy.min(beatStrengths)) / (numpy.max(beatStrengths) - numpy.min(beatStrengths))
# this gives the beat strengths from 0to 1 per onset beat, next normalise it so it can be applied to spline points in ue5
beatStrengthsList = beatStrengthsNormalise.tolist()
beatStrengthMean = float(beatStrengths.mean())
print ('Beat Strength Mean: ' + format(beatStrengthMean))


# rating of noisiness
zeroCrossingRate = librosa.feature.zero_crossing_rate(y)
zeroCrossingRateMean = float(zeroCrossingRate.mean()) 
print ('Zero Crossing Rate: ' + format(zeroCrossingRateMean))


# spectral rolloff = where the audio has most of its energy/power (either high / low frequnecy)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
rolloffMean = rolloff.mean()
print ('Rolloff Mean: ' + format(rolloffMean))


# FROM https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
# Find musical key
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
meanChroma = numpy.mean(chroma, axis=1)
chromaToKey = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Find the key by selecting the maximum chroma feature
estimatedKeyIndex = numpy.argmax(meanChroma)
estimatedKey = chromaToKey[estimatedKeyIndex]
print ('Estimated Key: ' + format(estimatedKey))

# create json structure for output
output = [
        {  
        "Name": randomFile,
        "BPM": tempo,
        "RMS Mean": rmsMean,
        "Spectral Centroid Mean": specCentroidMean,
        "Onset Strength Mean": onsetEnvMean,
        "Beat Strength Mean": beatStrengthMean,
        "Zero Crossing Rate Mean": zeroCrossingRateMean,
        "Rolloff Mean": rolloffMean,
        "Estimated Key": estimatedKey,
        "Beat Strengths Array": beatStrengthsList
        }
    ]
  

# create the json file 
outputPath = os.path.join(filePath, 'audioData.json')
with open(outputPath, "w") as outFile:
    json.dump(output, outFile, indent = 2)


#numpy.set_printoptions(threshold=sys.maxsize)