import numpy as np
import os
import random



def load_data_(_path):
    """
    Loads the data in _path
    Returns:
    The data (MFCC or Spectrogram) as raw_data
    IDs as names
    sizes of the data as sizes
    """
 
    path = _path
    sps = []
    names = []
    spects = []
    #load mfccs and labels
    for subdir, dirs, files in os.walk(path):
        for fle in files:
            filepath = subdir + os.sep + fle
            if filepath.endswith(".csv"):
                names.append(filepath.split('/')[-2])
                d = np.loadtxt(filepath)
                sps.append(d)
    ##Save the data
    for j in range(len(sps)):
        raw_data.append(sps[j])
    print('len of spects: ' + str(len(spects)))
   
   ##Save the size of the data
    sizes = []
    for i in range(len(names)):
        sizes.append(raw_data[i].shape[0])
    print('sizes of first 10: ' + str([s for s in sizes[0:10]])) 
    return raw_data, names, sizes

def prepare_data(raw_data, names, sizes):
     """
    spects: audio data  in various sizes
    names: the lables of the speakers
    sizes: the sizes of the input data

    Partition the set of input data into fixed size (26, 13) arrays 

    return: the set of data with fixed size, the lables array
    """
   new_data = []
    new_labels = []
    for i in range(len(raw_data)):
        frm = 100 #frame size defines the size of the data samples
        step = 100 #step size of the partitioning window
        
        ## If the size of the data sample is smaller than the frame size, abort the sample. 
        ## Otherwise, divide the data and save in new_data. The IDs are saved in new_labels.
        for j in range(0, sizes[i], step):
            if frm < sizes[i]:
                new_data.append(raw_data[i][j:frm, :])
                new_labels.append(names[i])
                frm += step
            else:
                continue
                        
    new_data = np.array(new_data).reshape(len(new_data), 100, 24)
    new_labels = np.array(new_labels).reshape(len(new_labels), 1)

    return new_data, new_labels
