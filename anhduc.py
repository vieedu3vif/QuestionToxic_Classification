#anh duc code tren file nay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from eda import EDA
import time

start_time = time.time()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
X_target0 = train[train.target == 0]  # Lấy mẫu ngẫu nhiên từ lớp 0
X_target1 = train[train.target == 1] # Lớp 1

print("Tổng số dữ liệu trong tập train: ",train.shape[0])
print("Số câu hỏi bình thường: ", len(train[train.target == 0]))
print("Số câu hỏi toxic: ",len(train[train.target == 1]))
print("Tỉ lệ giữa 2 lớp: ",len(train[train.target == 1])/len(train[train.target == 0]))

#Hàm dự đoán sử dụng linear
tfidf = TfidfVectorizer(ngram_range=(1, 3))
t = EDA(random_state=1)
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

# def predict_HistGradientBoosting(X_train, y_train, X_test):
#     tfidf.fit(X_train)
#     X_train_tfidf = tfidf.transform(X_train)
#     X_test_tfidf = tfidf.transform(X_test)
#
#     # Sử dụng HistGradientBoostingClassifier
#     model = HistGradientBoostingClassifier()
#     model.fit(X_train_tfidf.toarray(), y_train)
#
#     train_predictions = model.predict(X_train_tfidf.toarray())
#     test_predictions = model.predict(X_test_tfidf.toarray())
#     return train_predictions, test_predictions
def word(text, n):
    for _ in range(n):
        res = text
        if len(res.split()) > 1:
            res = t.random_insertion(res)
        if len(res.split()) > 1:
            res = t.random_swap(res)
        if len(res.split()) > 1:
            res = t.synonym_replacement(res, top_n=10*n+n-1)
        if len(res.split()) > 1:
            res = t.random_deletion(res, p=0.1)
    return res

def validate_undersampling():
    ina = X_target1['question_text']
    x1 = ina.apply(word, args=[1])
    x1 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x1,
    'target': X_target1['target']
})
    x2 = ina.apply(word, args=[2])
    x2 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x2,
    'target': X_target1['target']
})
    x3 = ina.apply(word, args=[3])
    x3 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x3,
    'target': X_target1['target']
})
    x4 = ina.apply(word, args=[4])
    x4 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x4,
    'target': X_target1['target']
})
    x5 = ina.apply(word, args=[5])
    x5 = pd.DataFrame({
        'qid': X_target1['qid'],
        'question_text': x5,
        'target': X_target1['target']
    })
    data_augmented = pd.concat([train, x1, x2, x3, x4, x5])
    data_augmented = data_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    X = data_augmented.question_text
    y = data_augmented.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_predictions, test_predictions = predict_linearSVC(X_train, y_train, X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_precision = precision_score(y_train, train_predictions)
    test_precision = precision_score(y_test, test_predictions)
    train_recall = recall_score(y_train, train_predictions)
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

end_time = time.time()  # Kết thúc đếm thời gian
total_time = end_time - start_time
print("Tổng thời gian chạy: {:.2f} giây".format(total_time))
