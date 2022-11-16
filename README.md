# Materials Genome Initiative: NLP Informatics

This repo is for a joint project between [Jordan Axelrod](https://github.com/jordanIAxelrod), Defne Circi, [Logan Cooper](https://github.com/ldtcooper), Thomas Lilly, and Shota Miki. It is an attempt to apply NLP techniques to apply sentence-level classifications to materials science papers to make the data contained in them more available to the scientific community. 

## Installation

You can install the required packages by running `conda env create -n pollydarton --file --from-history>environment.yml`.

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
