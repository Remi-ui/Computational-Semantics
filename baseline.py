from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import random


def read_conll(filename):
    words, labels = [], []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if not line.startswith(b"#") and line.strip():
                tokens = line.strip().split()
                words.append(tokens[0]) 
                labels.append(tokens[3])
    return words,labels
  
    
def shuffle_dependent_lists(l1,l2):
    tmp = list(zip(l1,l2))
    random.Random(123).shuffle(tmp)
    return zip(*tmp)

def train_naive_bayes(X_train, Y_train):
    '''Trains a Naive Bayes model on the training data with
    Tfidf.'''
    vec = CountVectorizer()
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)
    return naive_classifier


def train_svm(X_train, Y_train):
    '''Trains a Support Vector machine with an rbf kernel on the provided
    train data with Tfidf vectors.'''
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('svc', LinearSVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier

def train_random_forest(X_train, Y_train):
    '''Trains a random forest with Tfidf vectors.'''
    vec = TfidfVectorizer()
    r_forest = Pipeline([('vec', vec), ('svc', RandomForestClassifier())])
    r_forest = r_forest.fit(X_train, Y_train)
    return r_forest

def main():
    X_train, Y_train = read_conll("Data/train.txt")
    X_test, Y_test = read_conll("Data/test.txt")
    X_train, Y_train = shuffle_dependent_lists(X_train, Y_train)
    X_test, Y_test = shuffle_dependent_lists(X_test, Y_test)

    naive_bayes = train_naive_bayes(X_train, Y_train)
    Y_pred = naive_bayes.predict(X_test)
    print("Final accuracy Naive Bayes: {}".format(accuracy_score(Y_test, Y_pred)))

    random_forest = train_random_forest(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    print("Final accuracy Random Forest: {}".format(accuracy_score(Y_test, Y_pred)))

    svm = train_svm(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    print("Final accuracy SVM: {}".format(accuracy_score(Y_test, Y_pred)))
    
if __name__ == "__main__":
    main()
