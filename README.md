LatentGAN
=========

LatentGAN [1] with heteroencoder trained on ChEMBL 25 [2], which encodes SMILES strings into latent vector representations of size 512. A Wasserstein Generative Adversarial network with Gradient Penalty [3] is then trained to generate latent vectors resembling that of the training set, which are then decoded using the heteroencoder. A version of this model is also available in the MOSES platform [5]
 
The code has been confirmed to work with the environment provided in the `environment.yml` provided, but not all packages might be necessary. 

Dependencies
------------
This model uses the heteroencoder available as a package from [4]. The heteroencoder further requires [this package](https://github.com/EBjerrum/molvecgen) to run properly on the ChEMBL model. Neither of these are available from e.g. `anaconda` or `pip`, and have to be installed manually. 
IMPORTANT: The Deep Drug Coder on the main branch runs with tensorflow 2.0, and is incompatible with the pretrained models contained here. Please use the [moses branch](https://github.com/pcko1/Deep-Drug-Coder/tree/moses) of this model.

### Installation
1. Have `Git LFS` installed, available from conda.
2. Initialize Git LFS with `git lfs install`.
3.  Clone this repo using `git clone https://github.com/Dierme/latent-gan.git`
4.  Create conda environment from .yml file `conda env create --file environment.yml`
5.  Download dependencies Deep Drug Coder and molvecgen
6.  move ``Deep-Drug-Coder/ddc_pub/` and `molvecgen/molvecgen` to the root folder (to solve import references from the Deep-Drug-Coder)
~~~
mv Deep-Drug-Coder/ddc_pub/ .
mv molvecgen/molvecgen tmp/
mv tmp/ molvecgen/
~~~

- A complete file setup after cloning the repo for installing the dependencies, setting up the file structure and installing a conda environment with the name `latent_gan_env` can be easily done by using the shell script
4. Use shell script `bash install-dependencies.sh`

General Usage Instructions
--------------------------

One can use the runfile (`run.py`) for a single script that does the entire process from encoding smiles to create a training set, create and train a model, followed by sampling and decoding latent vectors back into SMILES using default hyperparameters.

~~~~
Arguments:
-sf <Input SMILES file name>
-st <Output storage directory path [DEFAULT:"storage/example/"]>
-lf <Output latent file name [DEFAULT:"encoded_smiles.latent"], this will be put in the storage directory>
-ds <Output decoded SMILES file name [DEFAULT:"decoded_smiles.csv"], this will be put in the storage directory>
--n-epochs <Number of epochs to train the model for [DEFAULT: 2000]>
--sample-n <Give how many latent vectors for the model to sample after the last epoch has finished training. Default: 30000>
--encoder <The data set the pre-trained heteroencoder has been trained on [chembl|moses] [DEFAULT:chembl]> IMPORTANT: The ChEMBL-trained heteroencoder is NOT intended for use with the MOSES SMILES, just as the moses-trained model is not intended for use with the ChEMBL based SMILES files.
~~~~

Note that tensorflow outputs a large amount of logging messages when the autoencoder is loaded, which can be overwhelming for a new user. If you want to limit these, we recommend you to use `run.py` with the following environmental variable
~~~
TF_CPP_MIN_LOG_LEVEL=2 python run.py
~~~

OR one can conduct the individual steps by using each script in succesion. This is useful to e.g. sample a saved model checkpoint using (`sample.py`).


1) Encode SMILES (`encode.py`): Gives a .latent file of latent vectors from a given SMILES file. Currently only accepts SMILES of token size smaller than 128. 


~~~~
Arguments:
-sf <Input SMILES file name>
-o <Output Smiles file name>
~~~~
 

2) Create Model (`create_model.py`): Creates blank model files generator.txt and discriminator.txt  based on an input .latent file. 

~~~~
Arguments: 
-i <.latent file> 
-o <path to directory you want to place the models in> 
~~~~

3) Train Model (`train_model.py`): Trains generator/discriminator with the specified parameters. Will also create .json logfiles of generator and discriminator losses. 
~~~~
Arguments:
-i <.latent file> 
-o model <directory path>
--n-epochs <Number of epochs to train for>
--starting-epoch <Model checkpoint epoch to start training from, if checkpoints exist> 
--batch-size <Batch size of latent vectors, Default: 64> 
--save-interval <How often to save model checkpoints> 
--sample-after-training <Give how many latent vectors for the model to sample after the last epoch has finished training. Default: 0>
--decode-mols-save-path <Give output path for SMILES file if you want your sampled latent vectors decoded> 
--n-critic-number <Number of of times discriminator will train between each generator number. Default: 5>
--lr <learning rate, Default: 2e-4> 
--b1,--b2 <ADAM optimizer constants. Default 0.5 and 0.9, respectively>
-m <Message to print into the logfile> 
~~~~

4) Sample Model (`sample.py`): Samples an already trained model for a given number of latent vectors. 
~~~~
Arguments: 
-l <input generator checkpoint file> 
-olf <path to output .latent file -n number of latent vectors to sample> 
-d <Option to also decode the latent vectors to SMILES> 
-odsf <output path to SMILES file> 
-m <message to print in logfile>
~~~~

5) Decode Model (`decode.py`) decodes a .latent file to SMILES. 
~~~~
Arguments 
-l <input .latent file> 
-o <output SMILES file path> 
-m <message to print in logfile>
~~~~


## Links

[1] [A De Novo Molecular Generation Method Using Latent Vector Based Generative Adversarial Network](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0397-9)

[2] [ChEMBL](https://www.ebi.ac.uk/chembl/)

[3] [Improved training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

[4] [Deep-Drug-Coder](https://github.com/pcko1/Deep-Drug-Coder)

[5] [Molecular Sets (MOSES): A benchmarking platform for molecular generation models](https://github.com/molecularsets/moses)
