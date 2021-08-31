from python_speech_features import mfcc
#import pitch
import scipy.io.wavfile
import numpy as np
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#path to the data
path = '/home/ubuntu/projects/voice/speaker_id/libri/dataset/LibriSpeech/train'

#path to Features
features = './train_mfcc/'
f = []

for subdir, dirs, files in os.walk(path):
        for fle in files:
            filepath = subdir + os.sep + fle
            if filepath.endswith(".wav"):
                f.append(filepath)

for i in f:
    filename = i
    try:
        #load the .wav files
        rate, signal = scipy.io.wavfile.read(filename)
        #13 MFCCs are extracted
        new_feats = mfcc(signal, rate, nfft=1024, numcep=13)
#         new_feats = scipy.stats.zscore(mfcc_feats, axis=0)

#         ptch = pitch.find_pitch(filename)
#         pitches = np.zeros((mfcc_feats.shape[0], 1))
#         for n in range(len(pitches)):
#             pitches[n] = ptch            
        #delta_feats = delta(mfcc_feats, 3)
        #all_feats = np.concatenate((mfcc_feats, delta_feats),1)
#         new_feats = np.concatenate((mfcc_feats, pitches), 1)

        wav = filename.split("train/")[1]
        foldername, fname = wav.split("/")
        fname = fname.split(".")[0]
        txt = features + foldername + os.sep + fname + ".csv"
        #save the MFCCs
        if not os.path.exists(features + foldername):
            os.makedirs(features + foldername)
        np.savetxt(txt, new_feats, delimiter=" ")
        print("Saved txt {}".format(txt))

    except Exception as e:
        print(e)
        print("Could not read {}".format(i))
