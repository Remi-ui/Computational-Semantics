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
                words.append(tokens[0])
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

def all_in_one(X_train_big,Y_train_big,X_train_small,Y_train_small):
    # downsampling English data
    X_train_big = X_train_big[:len(X_train_small)-1]
    Y_train_big = Y_train_big[:len(Y_train_small)-1]
    X_train = X_train_big + X_train_small
    Y_train = Y_train_big + Y_train_small
    
    return X_train, Y_train

def main():
    X_train_n, Y_train_n = read_conll("/content/drive/My Drive/data/nltrain.txt")
    X_test_n, Y_test_n = read_conll("/content/drive/My Drive/data/nltest.txt")
    X_train_e, Y_train_e = read_conll("/content/drive/My Drive/data/entrain.txt")
    X_test_e, Y_test_e = read_conll("/content/drive/My Drive/data/entest.txt")    
    
    # shuffle it
    X_train_n, Y_train_n = shuffle_dependent_lists(X_train_nl, Y_train_nl)
    X_test_n, Y_test_n = shuffle_dependent_lists(X_test_nl, Y_test_nl)
    X_train_e, Y_train_e = shuffle_dependent_lists(X_train_en, Y_train_en)
    X_test_e, Y_test_e = shuffle_dependent_lists(X_test_en, Y_test_en)
    
    
    #separately
    svm1 = train_svm(X_train_e, Y_train_e)
    Y_pred_e1 = svm1.predict(X_test_e)
    
    svm2 = train_svm(X_train_n, Y_train_n)
    Y_pred_n1 = svm1.predict(X_test_n)
    
    #all-in-one
    X_train,Y_train = all_in_one(X_train_e,Y_train_e,X_train_n,Y_train_n)
    svm = train_svm(X_train, Y_train)
    Y_pred_e = svm.predict(X_test_e)
    Y_pred_n = svm.predict(X_test_n)

    #separate results
    print(f'accuracy SVM_en:{accuracy_score(Y_test_e, Y_pred_e1):.2f}')
    print("F1_score SVM_en:",f1_score(Y_test_e, Y_pred_e1,average = 'macro'))
    print(f'accuracy SVM_nl:{accuracy_score(Y_test_n, Y_pred_n1):.2f}')
    print("F1_score SVM_nl:",f1_score(Y_test_n, Y_pred_n1,average = 'macro'))
    
    #all-in-one results
    print(f'accuracy SVM_en_all:{accuracy_score(Y_test_e, Y_pred_e):.2f}')
    print("F1_score SVM_en_all:",f1_score(Y_test_e, Y_pred_e,average = 'macro'))
    print(f'accuracy SVM_nl_all:{accuracy_score(Y_test_n, Y_pred_n):.2f}')
    print("F1_score SVM_nl_all:",f1_score(Y_test_n, Y_pred_n,average = 'macro'))
          
          
if __name__ == "__main__":
    main()
