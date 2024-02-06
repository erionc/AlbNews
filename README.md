# AlbNews

This repository contains Python code for reproducing the experiments with movie reviews in Albanian presented in this [paper](https://arxiv.org/abs/2306.08526). AlbNews is a topic modeling corpus of movie news headlines in Albanian, consisting of 600 topically labeled records and 2600 unlabeled records. Each labeled record includes a headline text and a label 'pol' for politics, 'cul' for culture, 'eco' for economy or 'spo' for sport. More details about the creation and the contents of AlbNews can be found [here](https://arxiv.org/abs/2306.08526).

## Data

Please download [AlbNews corpus](http://hdl.handle.net/11234/1-5411) and place its files inside the data/ folder. Afterwards, you can run the code of this repository using the following command: 

```
$ python basic_experiments.py -c <classifier>
```


## Citation

**If using the AlbNews data or the code of this repository, please cite the following paper:**

Erion Çano, Dario Lamaj. AlbNews: A Corpus of Headlines for Topic Modeling in Albanian. 
CoRR, abs/2306.08526, June 2023. URL https://arxiv.org/abs/2306.08526.

@article{DBLP:journals/corr/abs-2306-08526, \
author = {Erion {\c{C}}ano}, \
title = {AlbMoRe: A Corpus of Movie Reviews for Sentiment Analysis in Albanian}, \
journal = {CoRR}, \
volume = {abs/2306.08526}, \
year = {2023}, \
url = {https://arxiv.org/abs/2306.08526}, \
archivePrefix = {arXiv}, \
eprint = {2306.08526}, \
}