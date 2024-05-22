# minh chien code tren file nay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
#from wordcloud import STOPWORDS
import seaborn as sns
# from nltk import WordNetLemmatizer
import re
import string
import os
import seaborn as sb
from sklearn.cluster import KMeans
#from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("Tổng số dữ liệu trong tập train: ",train.shape[0])
print("Số câu hỏi bình thường: ", len(train[train.target == 0]))
print("Số câu hỏi toxic: ",len(train[train.target == 1]))
print("Tỉ lệ giữa 2 lớp: ",len(train[train.target == 1])/len(train[train.target == 0]))

#Hàm dự đoán sử dụng linear
tfidf = TfidfVectorizer(ngram_range=(1, 3))
def predict_linearSVC(X_train,y_train,X_test):
    tfidf.fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    svm = LinearSVC()
    svm.fit(X_train,y_train)
    return svm.predict(X_test)
 
#Dự đoán trên tập validation
def validate_base_model():
    train = pd.read_csv('data/train.csv')
    train = train.dropna(subset=['question_text'])
    X = train.question_text
    y = train.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predict = predict_linearSVC(X_train,y_train,X_test)
    return f1_score(predict,y_test)

def validate_undersampling():
    X_target0 = train[train.target == 0].sample(frac=0.26)  # Lấy mẫu ngẫu nhiên từ lớp 0
    X_target1 = train[train.target == 1]  # Lớp 1
    data = pd.concat([X_target0, X_target1])  # Nối hai DataFrame lại với nhau
    X = data.question_text
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predict = predict_linearSVC(X_train, y_train, X_test)
    return f1_score(predict, y_test)

print("F1-Score với Under Sampling: ", validate_undersampling())

    

print('F1-Score của base-model trên tập validation: ',validate_base_model())