from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import random
import torch
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from sklearn import preprocessing
import numpy as np
from datasets import load_metric
tokenizer = AutoTokenizer.from_pretrained ("seongju/klue-tc-bert-base-multilingual-cased")


def read_conll(filename):
    words, labels = [], []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            if line.startswith(b"# raw sent"):
                sentence = line[13:-1]
            if not line.startswith(b"#") and line.strip():
                tokens = line.strip().split()
                if tokens[6] == b'[]':
                  words.append(tokens[0]+ b" " + sentence)
                else:
                  words.append(tokens[0] + b" " + tokens[6] + b" " + sentence) #can add some other features
                labels.append(tokens[3])
    return words,labels
  
    
def shuffle_dependent_lists(l1,l2):
    tmp = list(zip(l1,l2))
    random.Random(123).shuffle(tmp)
    return zip(*tmp)

def bert(train_dataset, test_dataset):
    model = AutoModelForSequenceClassification.from_pretrained("seongju/klue-tc-bert-base-multilingual-cased", num_labels=69, ignore_mismatched_sizes=True)

    #training_args = TrainingArguments("test_trainer")
    batch_size = 32
    args = TrainingArguments(
              f"training_with_callbacks",
              evaluation_strategy ='steps',
              eval_steps = 100, # Evaluation and Save happens every 500 steps
              save_total_limit = 5, # Only last 3 models are saved. Older ones are deleted.
              learning_rate=2e-5,
              per_device_train_batch_size=batch_size,
              per_device_eval_batch_size=batch_size,
              num_train_epochs=8, # Number of epochs
              push_to_hub=False,
              metric_for_best_model = 'f1',
              load_best_model_at_end=True)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset, callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("Trained model")
    trainer.evaluate()

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro') 

    print("----------")
    print(accuracy)
    print(f1)
    print("----------")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

class TaggingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    X_train, Y_train = read_conll("Data/train.txt")
    X_test, Y_test = read_conll("Data/test.txt")
    
    # To read the dutch files
    X_train_dutch, Y_train_dutch = read_conll("Data/train_nl.txt")

    # Adding them to the existing list with English
    #X_train = X_train + X_train_dutch
    #Y_train = Y_train + Y_train_dutch
    
    X_train, Y_train = shuffle_dependent_lists(X_train, Y_train)
    X_train_dutch, Y_train_dutch = shuffle_dependent_lists(X_train_dutch, Y_train_dutch)
    X_test, Y_test = shuffle_dependent_lists(X_test, Y_test)

    list_len = len(X_train_dutch) - 1
    X_train = X_train[:list_len]
    Y_train = Y_train[:list_len]
    X_train_dutch = X_train_dutch[:list_len]
    Y_train_dutch = Y_train_dutch[:list_len]

    X_train = X_train + X_train_dutch
    Y_train = Y_train + Y_train_dutch

    X_train = [i.decode("utf-8")  for i in X_train]
    X_test = [i.decode("utf-8") for i in X_test]
    Y_train = [i.decode("utf-8")  for i in Y_train]
    Y_test = [i.decode("utf-8") for i in Y_test]

    le = preprocessing.LabelEncoder()
    trained_le = le.fit(Y_train + Y_test)
    Y_train = trained_le.transform(Y_train)
    Y_test = trained_le.transform(Y_test)

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = TaggingDataset(train_encodings, Y_train)
    test_dataset = TaggingDataset(test_encodings, Y_test)

    bert(train_dataset, test_dataset)
    
if __name__ == "__main__":
    main()
