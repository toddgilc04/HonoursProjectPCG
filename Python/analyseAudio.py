
print("\nRunning Audio Analysis Script\n")

import librosa
import random
import os
import json
import numpy
from pathlib import Path
import soundfile as sf
import time


# clamp created as values have to stay within reasonable bounds
def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n
    
# normalise function!    
def normalise(n, minBound, maxBound):
    normalised = (n - minBound) / (maxBound - minBound)
    return normalised  # catch any outlying values

def finalise(n, lower, upper, meanOrMedian, printText):

    if meanOrMedian == "mean":
        n = clamp(float(n.mean()), lower, upper) 
        print (printText + ' Mean: ' + format(n))

    elif meanOrMedian == "median":
        n = clamp(float(numpy.median(n)), lower, upper)
        print (printText + ' Median: ' + format(n))
   
    n = normalise(n, lower, upper)
    print (printText + ' NM: ' + format(n))
    return n
    
BPMLower = 37
BPMHigher = 185
rmsLower = .103
rmsHigher = .763
specCentroidLower = 548.25
specCentroidHigher = 3205.83
onsetStrengthLower = 0.5475
onsetStrengthHigher = 1.1559
beatStrengthLower = 0
beatStrengthHigher = 9.7606
zeroCrossingRateLower = 0
zeroCrossingRateHigher = 0.1462
rolloffLower = 12.112
rolloffHigher = 7516.434

t0 = time.time()

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
y, sr = sf.read(fullAudioPath)
y = librosa.util.normalize(y)

t1 = time.time()

# beats is required, i think to ensure everything isnt stored in tempo and the data is instead split
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo = clamp(round(tempo.mean(), 0), BPMLower, BPMHigher)
print('BPM: ' + format(tempo))


rms = librosa.feature.rms(y=y) # find rms of the audio (used as a loudness measurement)
rmsMin = rms.min()
rmsMax = rms.max()
rmsNormalise = (rms - rmsMin) / (rmsMax - rmsMin)    # normalise array values so they are useable
# next normalise values within the IQR range
rmsMean = finalise(rmsNormalise, rmsLower, rmsHigher, "mean", "RMS")


# brightness rating of audio track
specCentroid = librosa.feature.spectral_centroid(y=y, sr=sr)
specCentroidMean = finalise(specCentroid, specCentroidLower, specCentroidHigher, "mean", "Spectral Centroid")


# bass strength at loudest of track (found by restricting to low freq)
onsetEnv = librosa.onset.onset_strength(y=y, sr=sr, fmax = 1000,  n_mels=32)
onsetEnvMedian = finalise(onsetEnv, onsetStrengthLower, onsetStrengthHigher, "median", "Onset Strength")


# beat strength
onsetEnvHigherCap = librosa.onset.onset_strength(y=y, sr=sr, fmax = 8000,  n_mels=128)
beats = librosa.util.fix_frames(beats)
beatStrengths = onsetEnvHigherCap[beats]
# normalise the array to 0 to 1
beatStrengthsNormalise = normalise(beatStrengths, numpy.min(beatStrengths), numpy.max(beatStrengths))
# this gives the beat strengths from 0 to 1 per onset beat, next normalise it so it can be applied to spline points in ue5
beatStrengthsList = beatStrengthsNormalise.tolist()
beatStrengthMedian = finalise(beatStrengthsNormalise, beatStrengthLower, beatStrengthHigher, "median", "Beat Strength")


# rating of noisiness
zeroCrossingRate = librosa.feature.zero_crossing_rate(y)
zeroCrossingRateMedian = finalise(zeroCrossingRate, zeroCrossingRateLower, zeroCrossingRateHigher, "median", "Zero Crossing Rate")


# spectral rolloff = where the audio has most of its energy/power (either high / low frequnecy)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
rolloffMedian = finalise(rolloff, rolloffLower, rolloffHigher, "median", "Spectral Rolloff")


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
        "Onset Strength Median": onsetEnvMedian,
        "Beat Strength Median": beatStrengthMedian,
        "Zero Crossing Rate Median": zeroCrossingRateMedian,
        "Rolloff Median": rolloffMedian,
        "Estimated Key": estimatedKey,
        "Beat Strengths Array": beatStrengthsList
        }
    ]
  

# create the json file 
outputPath = os.path.join(filePath, 'audioData.json')
with open(outputPath, "w") as outFile:
    json.dump(output, outFile, indent = 2)



total = t1-t0
print(randomFile + ': ')
print(total)