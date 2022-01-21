from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random

def read_conll(filename):
    words, labels= [], []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if not line.startswith(b"#") and line.strip():
                tokens = line.strip().split()
                words.append(tokens[0]) #can add some other features
                # lemma.append(tokens[2])
                # role.append(token[6])
                labels.append(tokens[3])

    return words,labels

    
def shuffle_dependent_lists(l1,l2):
    tmp = list(zip(l1,l2))
    random.Random(123).shuffle(tmp)
    return zip(*tmp)

def train_svm(X_train, Y_train):
    '''Trains a Support Vector machine with an rbf kernel on the provided
    train data with Tfidf vectors.'''
    vec = TfidfVectorizer()

    svm_classifier = Pipeline([('vec', vec), ('svc', LinearSVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier

def main():
    X_train_nl, Y_train_nl = read_conll("/content/drive/My Drive/data/nltrain.txt")
    X_test_nl, Y_test_nl = read_conll("/content/drive/My Drive/data/nltest.txt")
    X_train_en, Y_train_en = read_conll("/content/drive/My Drive/data/entrain.txt")
    X_test_en, Y_test_en = read_conll("/content/drive/My Drive/data/entest.txt")    
    # X_train = X_train_nl + X_train_en
    # Y_train = Y_train_nl + Y_train_en
    list_len = len(X_train_nl) - 1
    X_train_en = X_train_en[:list_len]
    Y_train_en = Y_train_en[:list_len]
    X_train_nl = X_train_nl[:list_len]
    Y_train_nl = Y_train_nl[:list_len]
    # downsampling English data
    X_train = X_train_nl + X_train_en
    X_test = X_test_nl
    Y_train = Y_train_nl + Y_train_en
    Y_test = Y_test_nl
    X_train, Y_train = shuffle_dependent_lists(X_train, Y_train)
    X_test, Y_test = shuffle_dependent_lists(X_test, Y_test)


    svm = train_svm(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    print("Final accuracy SVM: {}".format(accuracy_score(Y_test, Y_pred)))
    print(classification_report(Y_test, Y_pred))
    
if __name__ == "__main__":
    main()
