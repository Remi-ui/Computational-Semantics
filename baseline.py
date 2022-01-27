from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,f1_score
from collections import Counter
from sklearn.svm import LinearSVC
import random

def read_conll(filename):
    features, labels = [], []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if line.startswith(b"# raw sent"):
                sentence = line[13:]
            if not line.startswith(b"#") and line.strip(): 
                tokens = line.strip().split()
                if tokens[6] == b"[]":
                    features.append(sentence)
                else:
                    features.append(tokens[6] + b" " + sentence)  
                labels.append(tokens[3])
                
    return features,labels
  
def shuffle_dependent_lists(l1,l2):
    tmp = list(zip(l1,l2))
    random.Random(123).shuffle(tmp)
    return zip(*tmp)
  
def train_svm(X_train, Y_train):
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('svc', LinearSVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier
  
  
def main():
    X_train_e, Y_train_e = read_conll("./Data/en/train.txt")
    X_test_e, Y_test_e = read_conll("./Data/en/test.txt")
    X_train_e, Y_train_e = shuffle_dependent_lists(X_train_e, Y_train_e)
    X_test_e, Y_test_e = shuffle_dependent_lists(X_test_e, Y_test_e)
    X_test_n, Y_test_n = read_conll("./Data/nl/test.txt")
    X_train_n, Y_train_n = read_conll("./Data/nl/train.txt")
    X_train_n, Y_train_n = shuffle_dependent_lists(X_train_n, Y_train_n)
    X_test_n, Y_test_n = shuffle_dependent_lists(X_test_n, Y_test_n)
    

    svm1 = train_svm(X_train_e[:len(X_train_n)], Y_train_e[:len(X_train_n)])
    Y_pred_e = svm1.predict(X_test_e)
    
    svm2 = train_svm(X_train_n, Y_train_n)
    Y_pred_n = svm2.predict(X_test_n)
  
    
    
    print(f'accuracy SVM_en:{accuracy_score(Y_test_e, Y_pred_e):.2f}')
    print("F1_score SVM_en:",f1_score(Y_test_e, Y_pred_e,average = 'macro'))
    print(f'accuracy SVM_nl:{accuracy_score(Y_test_n, Y_pred_n):.2f}')
    print("F1_score SVM_nl:",f1_score(Y_test_n, Y_pred_n,average = 'macro'))


  
if __name__ == "__main__":
    main()
