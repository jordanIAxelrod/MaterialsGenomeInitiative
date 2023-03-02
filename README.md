# Materials Genome Initiative: NLP Informatics

This repo is for a joint project between [Jordan Axelrod](https://github.com/jordanIAxelrod), [Defne Çirci](https://github.com/defnecirci), [Logan Cooper](https://github.com/ldtcooper), Thomas Lilly, and Shota Miki. It is an attempt to apply NLP techniques to apply sentence-level classifications to materials science papers to make the data contained in them more available to the scientific community.

## Installation
Due to permissions concerns the `/data` directory is empty. Please copy the provided `test.tsv` and `train.tsv` in the `data` directory as shown below. 

```
MaterialsGenomeInitiative
│
└───data
|   |   test.tsv
|   |   train.tsv
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
The `split-data.py` file can also deterministically turn raw labeled data into a pair of TSV files (one for training, one for testing) which can be used. This is not necessary for paper results reproduction.


## Adding Dependencies
Dependencies can be installed using conda or vanilla python venv. Python 3.8 was used for this project. You may experience issues with newer versions of python due to incompatibility with older packages.
### Conda
You can install the required packages by running `conda env create -n pollydarton --file env.yml`. If you don't have conda installed, you can follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html) to set it up. This also requires Gensim version 3.8.1 which conda seems to have trouble installing, so within the new conda env, also run `pip3 install gensim==3.8.1`.

If you need to add any dependencies, make sure to update the `environment.yml` file so that everyone has access to them. After adding a dependency with `conda install ...` you can do this by running `conda env export --from-history>environment.yml` and committing the new file to git.

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
Install Jupyter kernal for `env` and select the `env` kernel for running notebooks
```
$ ipython kernel install --user --name=env
```


## Running Code

### Reproduction Models
The basic reproduction can be done by running `python3 main.py` **from within the `/src` directory**. Running the code for the preprocessing, tuning, and extra data segments requires changing branches.
- Preprocessing: Run `python3 main.py` from the `preprocessing` branch [here](https://github.com/jordanIAxelrod/MaterialsGenomeInitiative/tree/preprocessing).
- Preprocessing + Hyperparameter Tuning: Run `python3 main.py` from the `tuning` branch [here](https://github.com/jordanIAxelrod/MaterialsGenomeInitiative/tree/tuning).
- Preprocessing + Extra Data: Run `python3 main.py` from the `extra-data` branch [here](https://github.com/jordanIAxelrod/MaterialsGenomeInitiative/tree/extra-data).

#### Testing the rule-based model
The code to test the rule-based model against the original data is in `prototyping.ipynb` in the section titled "Testing Rule Model". The code to run the rule-based model itself is in `new-sentence-labelling.py`.

### RoBERTa

Run all cells starting at "Imports" section in `src/roberta.ipynb`. Please note that this will take some time if not using a GPU or other high performance compute system.

### SciBERT, MatSciBERT, MatBERT

To use MatBERT, download these files into a folder and change the paths used by the model and the tokenizer in `src/Sci_Bert_Models.ipynb`

```
$ export MODEL_PATH="Your path"

$ mkdir $MODEL_PATH/matbert-base-cased $MODEL_PATH/matbert-base-uncased

$ curl -# -o $MODEL_PATH/matbert-base-cased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json
```

Run all cells in `src/Sci_Bert_Models.ipynb` for train and test results.


### Random Forest, XGBoost, LSTM
Run all cells in `src/preprocessing_additional_models.ipynb`. To avoid preprocessing, comment out cells under sections named "Preprocess".
