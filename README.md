# Open Set Traffic Classification: Are we there yet?

This is the source code used in the research paper 'Open Set Traffic Classification: Are we there yet?', currently under review for ACM SIGCOMM CCR 2022.

## Description

This work evaluates several open set classificatoin methods on five network traffic fingerprinting datasets.

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
