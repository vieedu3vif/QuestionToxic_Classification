import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE

class MyLogisticRegression:
    def __init__(self, learning_rate=0.9, num_iterations=4000, class_weight=None, regularization_strength=0.02):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.class_weight = class_weight
        self.regularization_strength = regularization_strength

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_class_weight(self, y):
        if self.class_weight == 'balanced':
            unique_classes = np.unique(y)
            class_weight = {}
            total_samples = len(y)
            for cls in unique_classes:
                class_count = np.sum(y == cls)
                class_weight[cls] = total_samples / (len(unique_classes) * class_count)
            return class_weight
        return None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        class_weight = self.compute_class_weight(y)

        for _ in range(self.num_iterations):
            linear_model = X.dot(self.weights) + self.bias
            if isinstance(linear_model, np.matrix):
                linear_model = linear_model.A1
            y_predicted = self.sigmoid(linear_model)

            if class_weight:
                dw = np.zeros(n_features)
                db = 0
                for i in range(n_samples):
                    error = y_predicted[i] - y[i]
                    dw += error * X[i] * class_weight[y[i]]
                    db += error * class_weight[y[i]]
                dw /= n_samples
                db /= n_samples
            else:
                dw = (1 / n_samples) * X.T.dot(y_predicted - y)
                if isinstance(dw, np.matrix):
                    dw = dw.A1
                db = (1 / n_samples) * np.sum(y_predicted - y)

            dw += self.regularization_strength * self.weights / n_samples  # Add regularization term
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = X.dot(self.weights) + self.bias
        if isinstance(linear_model, np.matrix):
            linear_model = linear_model.A1
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

# Đọc dữ liệu từ file CSV
data = pd.read_csv("data/data_train/train-1.csv")

# Xử lý dữ liệu và chuyển đổi thành vector TF-IDF
vectorizer = TfidfVectorizer()
data = data.dropna(subset=["target"])

X = vectorizer.fit_transform(data["question_text"])
y = data["target"]

# K-fold cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_experiment(model, X, y, kfold):
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    all_cm = []

    for train_index, test_index in kfold.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Sử dụng SMOTE để cân bằng dữ liệu huấn luyện
        smote = SMOTE(random_state=42)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

        model.fit(x_train_resampled, y_train_resampled)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        cm = confusion_matrix(y_test, y_pred)
        all_cm.append(cm)

    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_acc, avg_f1, avg_precision, avg_recall, all_cm


models = []
metrics = []
base_models = [
    MyLogisticRegression(),
    SklearnLogisticRegression(class_weight='balanced'),  # Thêm trọng số lớp
    BernoulliNB()
]

for model in base_models:
    avg_acc, avg_f1, avg_precision, avg_recall, all_cm = run_experiment(
        model,
        X, y, kfold
    )
    metrics.append({
        'accuracy': avg_acc,
        'f1_score': avg_f1,
        'precision': avg_precision,
        'recall': avg_recall,
    })

# In ra các kết quả metrics của các mô hình
for i, model in enumerate(base_models):
    print(f"\nMetrics for {model.__class__.__name__}:")
    for metric, value in metrics[i].items():
            print(f"{metric.capitalize()}: {value:.2f}")
