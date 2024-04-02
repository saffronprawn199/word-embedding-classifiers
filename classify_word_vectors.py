#!/usr/bin/env python
import os
import logging
import pandas as pd
import gensim
import numpy as np
import nltk
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import argparse
import pickle


os.system("taskset -p 0xff %d" % os.getpid())

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Download popular NLTK dataset
nltk.download("popular")

def save_sklearn_model(model, filename):
    """
        Save logistic regression model as pickle file.

        :param model: the actual sklearn model that was trained.
        :param filename: the name of the file you would like to save it as.
    """

    # save model
    # Save the model

    with open(filename + ".pkl", "wb") as file:
        pickle.dump(model, file)


def load_data(data_file, random_state):
    """
       Load and preprocess a dataset from a CSV file. This includes filtering out rows
       where the 'Sentence' column is null, selecting only 'Tag' and 'Sentence' columns,
       balancing the dataset based on the 'Tag' values (e.g. 't' and 'f'), and cleaning the text
       in the 'Sentence' column.

       :param data_file: Path to the CSV file containing the data.
       :return: A pandas DataFrame with the processed dataset, balanced by 'Tag' values and
                with cleaned 'Sentence' text.
    """

    df = pd.read_csv(data_file)
    df = df[pd.notnull(df["Sentence"])]
    df = df[["Tag", "Sentence"]]

    # Under-sampling code
    # df_t = df.loc[df['Tag'] == 't'].sample(frac = 1, random_state=random_state)
    # df_f = df.loc[df['Tag'] == 'f'].sample(frac = 1, random_state=random_state)
    #
    # df_t = df_t.iloc[:len(df_f.index)]
    # df = pd.concat([df_t, df_f], ignore_index=True)

    df["Sentence"] = df["Sentence"].apply(clean_text)

    return df

def clean_text(text):
    """
        Cleans a given text by converting it to lowercase, replacing certain punctuation
        and symbols with space, removing unwanted symbols, and excluding stopwords.

        :param text: A string containing the text to be cleaned.
        :return: A modified version of the initial string with the text cleaned as described.
    """
    # define regexes
    replace_by_space_re = re.compile("[/(){}[]|@,;]")
    bad_symbols_re = re.compile("[^0-9a-z #+_]")
    stopwords_set = set(stopwords.words("english"))
    text = text.lower()  # lowercase text
    text = replace_by_space_re.sub(
        " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = bad_symbols_re.sub(
        "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join(
        word for word in text.split() if word not in stopwords_set
    )  # delete stopwords from text
    return text


def word_averaging(wv, words):
    """
        Averages the vectors of a list of words. If a word is represented by a NumPy array,
        it's directly appended to the list of vectors to be averaged. If it's a token present
        in the word vector model's vocabulary, its corresponding vector is appended. This
        function handles words as both vectors and tokens, and computes a unit-normalized
        average vector of the input words.

        :param wv: A word vector model.
        :param words: A list of words, where each word can be a string or a NumPy array representing a vector.
        :return: A unit-normalized average of the word vectors as a NumPy array. If no words
                 can be processed, returns a zero vector of the same dimensionality as the word vectors.
    """

    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.index_to_key:
            mean.append(wv.vectors[wv.key_to_index[word]])
            all_words.add(wv.key_to_index[word])

    if not mean:
        logging.warning("cannot compute mean of following words %s", words)
        return np.zeros(
            wv.vector_size,
        )
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    """
       Applies word averaging on a list of texts. Each text is tokenized, and its words are
       averaged using the `word_averaging` function.

       :param wv: A word vector model.
       :param text_list: A list of strings, where each string is a piece of text to be processed.
       :return: A 2D NumPy array where each row represents the averaged vector of the corresponding text.
   """

    return np.vstack([word_averaging(wv, post) for post in text_list])


def w2v_tokenize_text(text):
    """
        Tokenizes a given text into words. It splits the text into sentences, tokenizes each
        sentence into words, and filters out tokens that are less than 2 characters long.

        :param text: A string containing the text to be tokenized.
        :return: A list of string tokens resulting from the tokenization of the input text.
    """

    tokens = []
    for sent in nltk.sent_tokenize(text, language="english"):
        for word in nltk.word_tokenize(sent, language="english"):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def print_report(df, y_pred, y_test, y_pred_prob, name, output_file):

    """
        Generates a classification report and appends it to a specified output file. The report
        includes the model's name, accuracy, AUC score, and a classification report detailing
        precision, recall, f1-score, and support for each class.

        :param df: Dataframe containing the dataset used for classification, required for
                   extracting unique classification tags.
        :param y_pred: Array of prediction values made by the classification model.
        :param y_test: Array of actual target values to compare against predictions.
        :param y_pred_prob: Array of prediction probabilities used to compute AUC score.
        :param name: String indicating the name of the model used for classification.
        :param output_file: Name of the file to which the report will be appended. If the file
                            does not exist, it will be created.
    """

    classifier_path = "./classification/"
    os.makedirs(classifier_path, exist_ok=True)

    file = open(classifier_path + output_file, "a")
    file.write(name + ":\n")
    file.write("accuracy %s" % accuracy_score(y_pred, y_test) + "\n")

    file.write(
        "AUC %s"
        % roc_auc_score(y_test, y_pred_prob[:, 1], multi_class="ovr", average="micro")
        + "\n"
    )
    my_tags = df["Tag"].unique()
    file.write(classification_report(y_test, y_pred, target_names=my_tags) + "\n")
    file.close()


def train_baseline(df, test_size, random_state, output_file):
    """
       Splits the dataset into training and testing subsets, creates a classification
       pipeline using a logistic regression model, fits the model to the training data,
       and then predicts on the test data. It returns the predictions, actual test values,
       and prediction probabilities.

       :param df: Pandas DataFrame containing the dataset with features and labels for
                  classification.
       :return: A tuple containing the prediction values, actual test values, and prediction
                probabilities for the test set.
    """

    x = df.Sentence
    y = df.Tag
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    logreg = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, C=1e5, multi_class="ovr")),
        ]
    )

    logreg.fit(x_train, y_train)

    # Save Logistic regression model after training.
    save_sklearn_model(logreg, "./logistic_regression_models/" + output_file.replace(".txt", ""))

    y_pred = logreg.predict(x_test)
    # Adjust ROC AUC calculation for binary classification
    y_pred_prob = logreg.predict_proba(x_test)

    return y_pred, y_test, y_pred_prob


def train_w2v_classifier(model, df, test_size, random_state, output_file):
    """
        Trains a logistic regression classifier using word vectors obtained from a pre-trained
        Word2Vec model. The function tokenizes the text, averages the word vectors, splits the
        dataset into training and test sets, fits the logistic regression model, and predicts
        on the test set.

        :param model: Path to the pre-trained Word2Vec model.
        :param df: Pandas DataFrame containing the dataset with 'Sentence' and 'Tag' columns.
        :return: A tuple containing the prediction values, actual test labels, and prediction
                 probabilities for the test set.
    """
    # Test with gensim's pretrained embeddings

    wordvec_model = Word2Vec.load(model)
    wv = wordvec_model.wv

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    test_tokenized = test.apply(
        lambda r: w2v_tokenize_text(r["Sentence"]), axis=1
    ).values
    train_tokenized = train.apply(
        lambda r: w2v_tokenize_text(r["Sentence"]), axis=1
    ).values

    x_train_word_average = word_averaging_list(wv, train_tokenized)
    x_test_word_average = word_averaging_list(wv, test_tokenized)

    logreg = LogisticRegression(n_jobs=-1, max_iter=1000, C=1e5, multi_class="ovr")
    logreg = logreg.fit(x_train_word_average, train["Tag"])

    # Save Logistic regression model after training.
    save_sklearn_model(logreg, "./logistic_regression_models/" + output_file.replace(".txt", ""))

    y_pred = logreg.predict(x_test_word_average)
    y_pred_prob = logreg.predict_proba(x_test_word_average)

    return y_pred, test.Tag, y_pred_prob


def get_w2v_classifier(model, df, output_file, test_size, random_state):
    """
        Trains a Word2Vec classifier on the provided dataset, generates a classification report,
        and saves or appends the report to a specified output file.

        :param model: Path to the pre-trained Word2Vec model to use for embedding.
        :param df: Pandas DataFrame containing the data for classification.
        :param output_file: The name of the file where the classification report will be saved
                            or appended.
    """
    y_pred, y_test, y_pred_prob = train_w2v_classifier(model, df, test_size, random_state, output_file)
    name = "Classification model using W2V"

    print_report(df, y_pred, y_test, y_pred_prob, name, output_file)
    print("Classification done")


def get_baseline(df, output_file, test_size, random_state):
    """
        Trains a baseline classification model, generates a classification report, and saves
        or appends the report to a specified output file. The baseline model is a logistic
        regression classifier with a text processing pipeline.

        :param df: Pandas DataFrame containing the data for classification.
        :param output_file: The name of the file where the classification report will be saved
                            or appended.
    """
    y_pred, y_test, y_pred_prob = train_baseline(df, test_size, random_state, output_file)
    name = "Baseline classification model"

    print_report(df, y_pred, y_test, y_pred_prob, name, output_file)
    print("Classification done")


if __name__ == "__main__":
    # Make directory to store logistic regression and embedding models.
    os.makedirs("./logistic_regression_models", exist_ok=True)
    os.makedirs("./embedding_models", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Classify pre-trained word embedding models and baseline classification model."
    )
    requiredNamed = parser.add_argument_group("Required named arguments")

    requiredNamed.add_argument(
        "--inFilePath",
        action="store",
        metavar="string",
        type=str,
        dest="inFilePath",
        help="defines the input csv file path",
        required=True,
    )

    requiredNamed.add_argument(
        "--inModelPath",
        action="store",
        metavar="string",
        type=str,
        dest="inModelPath",
        help="specifies word embedding input file name",
        required=True,
    )

    requiredNamed.add_argument(
        "--outFileName",
        action="store",
        metavar="string",
        type=str,
        dest="outFileName",
        help="specifies classification model output file name",
        required=True,
    )

    parser.add_argument(
        "--baseLine",
        metavar="bool",
        type=bool,
        default=False,
        help="specifies if the baseline classification model should be trained",
    )

    parser.add_argument(
        "--randomState",
        metavar="int",
        type=int,
        default=42,
        help="specifies random seed used to initialise logistic regression model, for reproducibility - default value is 42",
    )

    parser.add_argument(
        "--testSetSize",
        metavar="float",
        type=float,
        default=0.3,
        help="specifies split ratio between training and test set - default value is 0.3"
    )

    args = parser.parse_args()
    data = args.inFilePath
    out_file = args.outFileName
    pretrained_model = args.inModelPath
    dataframe = load_data(data, args.randomState)
    if args.baseLine == 1:
        get_baseline(dataframe, out_file, args.testSetSize, args.randomState)
        get_w2v_classifier(pretrained_model, dataframe, out_file,  args.testSetSize, args.randomState)
    else:
        get_w2v_classifier(pretrained_model, dataframe, out_file,  args.testSetSize, args.randomState)



