#anh duc code tren file nay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import string
import os
import seaborn as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
X_target0 = train[train.target == 0].sample(frac=0.9)  # Lấy mẫu ngẫu nhiên từ lớp 0
X_target1 = train[train.target == 1].sample(frac=0.9)  # Lớp 1

print("Tổng số dữ liệu trong tập train: ",train.shape[0])
print("Số câu hỏi bình thường: ", len(train[train.target == 0]))
print("Số câu hỏi toxic: ",len(train[train.target == 1]))
print("Tỉ lệ giữa 2 lớp: ",len(train[train.target == 1])/len(train[train.target == 0]))

#Hàm dự đoán sử dụng linear
tfidf = TfidfVectorizer(ngram_range=(1, 3))
def predict_linearSVC(X_train, y_train, X_test):
    tfidf.fit(X_train)
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    weights = {}
    weights[0] = len(X_target0) / 2 * len(X_train)
    weights[1] = len(X_target1) / 2 * len(X_train)
    y = [0, 1]
    class_weight = {val: weights[index] for index, val in enumerate(y)}
    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)
    train_predictions = svm.predict(X_train_tfidf)
    test_predictions = svm.predict(X_test_tfidf)
    return train_predictions, test_predictions

# Dự đoán trên tập validation
# def validate_base_model():
#     train = pd.read_csv('data/train.csv')
#     train = train.dropna(subset=['question_text'])
#     X = train.question_text
#     y = train.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     train_predictions, test_predictions= predict_linearSVC(X_train, y_train, X_test)
#     train_accuracy = accuracy_score(y_train, train_predictions)
#     test_accuracy = accuracy_score(y_test, test_predictions)
#     f1 = f1_score(y_test, test_predictions)
#     print(f"Train Accuracy: {train_accuracy}")
#     print(f"Validation Accuracy: {test_accuracy}")
#     return f1

def validate_undersampling():
    data = pd.concat([X_target0, X_target1])  # Nối hai DataFrame lại với nhau
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data_augmented = pd.concat([X_target0, X_target1, X_target1, X_target1, X_target1]) # Lặp lại dữ liệu nhãn 1 4 lần
    data_augmented = data_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    X = data_augmented.question_text
    y = data_augmented.target

    remaining_data = train.drop(data.index) # Lấy phần data chưa dùng để train từ tập train gốc để test
    remaining_data = remaining_data.sample(frac=1, random_state=42).reset_index(drop=True)
    X_test = remaining_data.question_text
    y_test = remaining_data.target

    train_predictions, test_predictions = predict_linearSVC(X, y, X_test)

    train_accuracy = accuracy_score(y, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_precision = precision_score(y, train_predictions)
    test_precision = precision_score(y_test, test_predictions)
    train_recall = recall_score(y, train_predictions)
    test_recall = recall_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)


    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
    precision1_score = tp / (tp + fp)
    recall1_score = tp / (tp + fn)
    print("Precision score", precision1_score)
    print("Recall score", recall1_score)

    print("Train Accuracy: ", train_accuracy)
    print("Validation Accuracy: ", test_accuracy)
    print("Train Precision: ", train_precision)
    print("Validation Precision: ", test_precision)
    print("Train Recall: ", train_recall)
    print("Validation Recall: ", test_recall)

    return f1

print("F1-Score với Under Sampling: ", validate_undersampling())
# print('F1-Score của base-model trên tập validation: ',validate_base_model())