# embedding_map

This repository allows reproducabillity of the paper *Non-deterministic Behavior of Ranking-based
Metrics when Evaluating Embeddings* which is available in ./docs

# Reproducing Experiments
Experiments require an ubuntu 16.04 system although all Unix including OsX should work.

Install requirements for the experiments:
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python-pip
pip install --user --upgrade seaborn matplotlib numpy scipy 
```

To run experiments and produce all plots as pdfs:
```bash
git clone https://github.com/anguelos/embedding_map.git
cd embedding_map
PYTHONPATH="./src/" python ./src/experiments.py ./data/phocnet.npz 
```
The pdfs are saved in embedding_map and producing them should take less than a minute on a contemporary machine.

# Using the mAP code

File ./src/map.py is a python module that contains a fast implementation for computing *mAP* including the improovements discussed in the paper. 