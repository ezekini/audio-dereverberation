# audio-dereverberation
 Disorganized code for my thesis: Speech dereverberation using DNN.

## Files.
* convolution.py: Convolve audio files contained in two different folders. Not neccesary if using the dataset provided. In case you want to use your own Room Impulse Responses to convolve with the _anechoic_ files
* functionsdef.py: Different functions used for preprocessing (scaling, FFT, loading, etc).
* dereverb-pytorch.ipynb: Notebook with the the whole pipeline. -->> [**Currently working on the cloud using Google Colab here**](https://colab.research.google.com/drive/1afmXN6R30-qaFyLhnynvBQqq3AuEp4pY)

## Dataset.
* Contained in my Google Drive in [this link](https://drive.google.com/open?id=1O-4CH0T2pt4DrAG_6dRZQJCuwJiAdB7p).
* __anechoic__ folder contains 1444 anechoic speech audiofiles.
* __reverberant__ folder contains 8664 speech audiofiles generated using the anechoic files convolved with room impulse responses (the information regarding this will be soon uploaded with the procedure, impulse files, etc.)
