# Architecture optimisation

This repo contains the code supporting the results presented in the paper 'Architectural Optimisation in Deep Neural Networks. Tests of a theoretically inspired method.'.
By Sacha Cormenier, Gianluca Manzan, Paolo Branchini and Pierluigi Contucci.

# Requirements

* Python >= 3.8.
* Windows, Linux or macOS.

# Code

This repo uses the data obtained when neural networks are trained and tested using the methods introduced in https://github.com/ArchitecturalOpt/Architecture-Optimisation. In the directory 'Sets' are stocked the data obtained for different datasets and seeds.

Following the procedure introduced in the paper 'Architectural Optimisation in Deep Neural Networks. Tests of a theoretically inspired method.', neural networks initially created with one input layer of 784 neurons, three hidden layers of 64 neurons each and an output layer are modified for architectural optimisation purposes. Data containing the values from the output layers, weights and predicitions are exploited in functions used in this repo for obtaining the results presented in the paper.

----

'Functions.py' and 'Functions_OddEven.py' gather all the functions necessary for extracting the data from the different files.

'getdata.py' extract the data from the different files in the directory "Sets" using the functions from 'Functions.py' and 'Functions_OddEven.py'. It stocks the information in '.out' files stocked in "Data_saves" folder, and plot the different metrics for a given training seed and dataset.

'Statistics.py' gets back the data from the '.out' files and plot the different graphics using uncertainties.

All the images are stocked in the folder "images"