# Partially Observed Object Search with PROPHE-C
This repository contains the source code and a basic demonstration notebook for our paper: "Semantially-Driven Object Search Using Partially Observed 3D Scene Graphs".

### Prerequisites
The repo has been tested to work with Python >= 3.11, so those versions are recommended. Whether any version below that will work has not been tested.

### Installation
In a desired location on your computer, do:
```
git clone git@github.com:UW-CTRL/prophe-c.git
```

Change directories into the repo folder:
```
cd prophe-c
```

Then, set up a Python virtual environment:
```
python3 -m venv env
```

Activate the environment with:
```
source env/bin/activate
```

From there, install the required packages:
```
pip install -r requirements.txt
```

### Running an animated demonstration
The file `demo.ipynb` is a Jupyter notebook that can be used to run a couple object-search scenarios that match the ones shown in our paper's "Qualitative Results" section.  Follow the cells there, and it should all work!

### Bibtex
If you wish to use or reference this code for your own research, please use the following bibtex! :)

```
@inproceedings{
remy2023semanticallydriven,
title={Semantically-Driven Object Search Using Partially Observed 3D Scene Graphs},
author={Isaac Remy and Abhishek Gupta and Karen Leung},
booktitle={NeurIPS 2023 Foundation Models for Decision Making Workshop},
year={2023},
url={https://openreview.net/forum?id=YVZ03XsMfg}
}
```
