## Project Motivation

This is the code for the final project for Computational Semantics. The goal of this project is to create a system that can automatically perform semantic tagging for English and Dutch texts.

## File Descriptions
* `data` folder contains English and Dutch texts which have been divided to `train.conll`, `dev.conll` and `test.conll`. The whole data is collected from https://github.com/RikVN/DRS_parsing/tree/master/parsing/layer_data/4.0.0.
* `baseline.py` describes the model trained on `data` using Support Vector Machine(SVM) with different feature extractions.
* `multilingual_bert.py` describes the model trained on `data` using Multilingual BERT with different feature extractions.
* `rename.py` is used for converting a `.conll` file to a `.txt` file. It is supposed to be in the same folder as being converted documents.

<h2>Models</h2>
The different mBERT models that were trained for our experiments can be downloaded here: https://www.transferxl.com/download/08jGH2gXTSs0JJ


