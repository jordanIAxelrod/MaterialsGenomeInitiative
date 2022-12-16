# Materials Genome Initiative: NLP Informatics

This repo is for a joint project between [Jordan Axelrod](https://github.com/jordanIAxelrod), Defne Circi, [Logan Cooper](https://github.com/ldtcooper), Thomas Lilly, and Shota Miki. It is an attempt to apply NLP techniques to apply sentence-level classifications to materials science papers to make the data contained in them more available to the scientific community. 

## Installation

You can install the required packages by running `conda env create -n pollydarton --file env.yml`. If you don't have conda installed, you can follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html) to set it up. This also requires Gensim version 3.8.1 which conda seems to have trouble installing, so within the new conda env, also run `pip3 install gensim==3.8.1`.

**NB:** Due to permissions concerns the `/data` directory is empty. You will want to populate it with data during the installation process. It should look like this:

```
MaterialsGenomeInitiative
│
└───data
│   │
│   └───raw
│       │   action.txt
│       │   constituent.txt
│       │   null.txt
│       │   property.txt
...
|   README.md
|   split-data.py
```

The code in `split-data.py` will (deterministically) turn this raw data into a pair of TSV files (one for training, one for testing) which can be used. 

### Adding Dependencies
Dependencies can be installed using conda or vanilla python venv
<br><br>
### Conda
If you need to add any dependencies, make sure to update the `environment.yml` file so that everyone has access to them. After adding a dependency with `conda install ...` you can do this by running `conda env export --from-history>environment.yml` and committing the new file to git.
<br><br>
### Python venv
First, set the repo as your working directory. 
```
$ cd MaterialsGenomeInitiative
```
Create a python venv directory if you don't alredy have one
```
$ python -m venv env
```
Activate the venv
```
$ source ./env/bin/activate
```
Install the necessary packages. The `--no-binary` arg is required for `gensim` because the version of `gensim` required does not build with modern versions of clang.
```
$ pip install --no-binary gensim -r requirements.txt
```
Install Jupyter kernal for `venv` and select the `venv` kernel for running notebooks
```
$ ipython kernel install --user --name=venv
```

<br>
## Running Code

### Reproduction Models

### RoBERTa

Run all cells starting at "Imports" section in `src/roberta.ipynb`. Please note that this will take some time if not using a GPU or other high performance compute system.

### SciBERT, MatSciBERT, MatBERT

### Random Forest, XGBoost, LSTM