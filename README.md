##Pipeline:
- Feature Extraction
- Data preparation
- Defining the model
- Training & evaluating the model
- Testing the model

MFCCs are extracted from raw speech signals using ```extract_mfcc.py```. 
The library for feature extraction is ```python_speech_features```. 

The path to the training data is given in the main function (data_path).

The extracted features(MFCC/Spectrogram) are loaded using ```unnorm_load_mfcc.py``` script;
- By defining the size of the parameter "frm", the size of the input data is defined. 
- Parameter "step" defines the step size of the segmentation window. If the step size < frame size --> the segments are overlapping, else: there is no overlap between data samples.
  _

```ftdnn_libri.py```: Creates FTDNN, trains and evaluates the model

  unnorm_load_mfcc.py ----> Load and prepare data
  models.py ---> Contains layers of FTDNN

![](/Users/fasounaki/Documents/Speaker_Recognition/photos/ftdnn_arch.png)

The best models during training are saved in the "checkpoints" directory.

The achieved accuracy with FTDNN is 87.2% on 1-second test segments, and 96.6% on 3-second utterances.

Trained models are saved in 'checkpoints' folder.

## Project Structure - Speaker identification with FTDNN

├──Speaker-identification-FTDNN

  ├── inputs # Datasets & features
     │   ├── audio_files 
     │   ├── MFCC

  ├── data loader
     │   ├── unnorm_load_mfcc.py

  ├── models

     │   ├── models.py
     │   ├── ftdnn_libri.py

  ├── outputs 
     │   ├── checkpoints  

  ├── README.md
  ├── requirements.txt # speaker-identification
