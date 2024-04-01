# Using Logistic Classification for Pre-trained Word Embedding

Please follow the instructions below.

## Installation

For installation to work as expect please insure you are running `python=3.9` on your local machine before creating the virtual environment.
#### Create a virtual environment:

```bash
python3 -m venv venv
```

#### Activate virtual environment:

```bash
source venv/bin/activate
```

#### To deactivate:

```bash
(venv) deactivate
```


#### Install environment dependencies:

Before running the application, ensure that you have Streamlit installed. If not, you can install it using pip:

```bash
pip install -r requirements.txt
```



## Overview
This script classifies sentences into categories using logistic regression. It supports two modes:

1. Classification using pre-trained word embeddings.
2. A baseline classification using TF-IDF vectors.

## Input File Format:

The input CSV file should contain two columns:

* `Sentence`: The text of the sentence.
* `Tag`: The category label of the sentence.

Example of a valid CSV input file (classification_dataset_example.csv):

```shell bash
Sentence,Tag
"This is an example sentence.",news
"Another example sentence.",sport
"Another example sentence.",weather       
"Another example sentence.",advertisement       
"Another example sentence.",traffic      
```

## Usage Instructions
Classify Using Both Pre-trained Word Embeddings and the Baseline Classifier

```shell bash 
python classify_word_vectors.py --inFilePath <path_to_input_csv> --inModelPath <path_to_word_embedding_model> --outFileName <output_file_name> --baseLine 1
```

* `<path_to_input_csv>`: The path to the input CSV file.
* `<path_to_word_embedding_model>`: The path to the pre-trained word embedding model.
* `<output_file_name>`: The name of the file where the classification report will be saved.

## Example:

```shell bash
python classify_word_vectors.py --inFilePath English_tagged.csv --inModelPath ./word2vec_models/word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300 --outFileName classification_report.txt --baseLine 1
```
## Classify Using Only Pre-trained Word Embeddings:

```shell bash
python classify_word_vectors.py --inFilePath <path_to_input_csv> --inModelPath <path_to_word_embedding_model> --outFileName <output_file_name>
```

Use the same parameter values as described above. Omit the `--baseLine` flag to skip baseline classification.

## Example:
```shell bash
python classify_word_vectors.py --inFilePath English_tagged.csv --inModelPath ./word2vec_models/word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300 --outFileName classification_report.txt
```

This README aims to provide clear instructions on how to use the script for classifying sentences into predefined categories using logistic regression and pre-trained word embeddings.
