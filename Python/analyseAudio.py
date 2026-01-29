
print("\nRunning Audio Analysis Script\n")

import librosa
import random
import os
import json
import numpy
import sys
from pathlib import Path

# set the path where audio is discovered
filePath = os.path.dirname(__file__)
audioPath = os.path.join(filePath, 'SONG_ANALYSE')

# assign where files will be looked for
files = os.listdir(audioPath)
# find any random file in folder
files = [f for f in files if os.path.isfile(os.path.join(audioPath, f))]
random_file = random.choice(files)


# combine the file and the path 
full_path = os.path.join(audioPath, random_file)
y, sr = librosa.load(full_path)


tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
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


specCentroid = librosa.feature.spectral_centroid(y=y, sr=sr)
specCentroidMean = float(specCentroid.mean())
print ('Spectral Centroid Mean: ' + format(specCentroidMean))
# potentially find centroid every x/y sec


#numpy.set_printoptions(threshold=sys.maxsize)
#print (rmsNormalise)


onsetEnv = librosa.onset.onset_strength(y=y, sr=sr, fmax = 1000,  n_mels=32)

onsetEnvMean = float(onsetEnv.mean()) # bass strength at loudest of track (found by restricting to low freq)
print ('Onset Strength Mean: ' + format(onsetEnvMean))




# create json structure for output
output = [
        {  
        "Name": random_file,
        "BPM": tempo,
        "RMS Mean": rmsMean,
        "Spectral Centroid Mean": specCentroidMean,
        "Onset Strength Mean": onsetEnvMean
        }
    ]
  

# create the json file 
outputPath = os.path.join(filePath, 'audioData.json')
with open(outputPath, "w") as outfile:
    json.dump(output, outfile, indent = 2)
