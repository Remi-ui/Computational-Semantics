from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from collections import Counter
import random

def read_conll(filename):
    words, labels = [], []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if not line.startswith(b"#") and line.strip():
                tokens = line.strip().split()
                words.append(tokens[0]) #can add some other features
                labels.append(tokens[3])
    return words,labels
  
 def shuffle_dependent_lists(l1,l2):
    tmp = list(zip(l1,l2))
    random.Random(123).shuffle(tmp)
    return zip(*tmp)
  
if __name__ == "__main__":
  X_train, Y_train = read_conll("./Data/train.txt")
  X_test, Y_test = read_conll("./Data/test.txt")
  X_train, Y_train = shuffle_dependent_lists(X_train, Y_train)
  X_test, Y_test = shuffle_dependent_lists(X_test, Y_test)
  
  vec = CountVectorizer()
  classifier = Pipeline([('vec',vec),('cls', MultinomialNB())])
  classifier.fit(X_train, Y_train)
  Y_pred = classifier.predict(X_test)
  
  acc = accuracy_score(Y_test, Y_pred)
  print("Final accuracy: {}".format(acc))
