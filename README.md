# Open Set Traffic Classification: Are we there yet?

This is the source code used in the research paper 'Robust open-set classification for encrypted traffic fingerprinting', published in _Computer Networks_ (2023).

## Description

This work evaluates several open-set classification methods on five network traffic fingerprinting datasets.

Datasets:
1. AWF: https://github.com/DistriNet/DLWF
2. DF: https://github.com/deep-fingerprinting/df
3. DC: https://research.csiro.au/isp/research/network-measurement-modelling/deep-bypass/
4. SETA: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9444314 (Contact authors for the dataset)
5. IOT: https://github.com/SmartHomePrivacyProject/DeepVCFingerprinting (Google Home)

Underlying CNN models:
1. AWF and DF datasets: CNN model from https://github.com/deep-fingerprinting/df
2. DC: CNN model from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8548317
3. SETA: CNN model defined in this code repo
4. IOT: CNN model from https://github.com/SmartHomePrivacyProject/DeepVCFingerprinting (sample represented with both size and direction)

Dataset splits (for datasets without explicit open set) can be found in the folder Dataset_splits

The code used for Softmax thresholding is in OpenMax/'Dataset name'/closed/open_softmax_thresh.py
  
All methods except CAC are implemented using Tensorflow/Keras, while CAC uses Pytorch (the code used is taken from the original work). 
