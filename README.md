# DeepLearning

This repository contains the Deep Learning code used for morphological classification of DES galaxies presented in Domínguez Sánchez et al. (2018b).

The code includes a knowdelege transfer step, consisting in loading the weights learned by a model trained and tested with SDSS data (presented in Domínguez Sánchez et al. 2018a) and then re-training the models with a small DES sample of galaxies (∼500-300).

We show that the training sample is enough for obtaining comparable results to the SDSS models trained with 5000 galaxies. This demonstrates that, once trained with a particular dataset, machines can quickly adapt to new instrument characteristics (e.g., PSF, seeing, depth), reducing by almost one order of magnitude the necessary training sample for morphological classification.

We have meade public the code for training Q1, i.e., if galaxies show a disk vs being smooth. If you are interested in the code for other classification tasks or for reading the galaxy images, contact me at helenado@sas.upenn.edu

For more details, please see (and cite) Domínguez Sánchez et al. (2018b). 

