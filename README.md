# dermosxai
Explainable methods for medical imaging datasets.

# TODO
## Usage 
Can be installed dusing pip
Dockerfile and docker-compose is provided too.

To process the datasets, see the notebooks process_DDSM, process_IAD and process_HAM10000
Code to train attribute predictors is in train_abl
Code to train linear classifiers on top of either human, resnet or human+resnet features is provided in train_classfier.py
Code to train the joint model is provided in train_joint.py
Code to train the autoencoders is in train_VAE.py

Code is provided as a guide, it will not necessarily work as is because of the data 
dependencies but documentation and implementation details should be useufl to replicate
our results in your data.

# Contact
ecobos@tuebingen.mpg.de
