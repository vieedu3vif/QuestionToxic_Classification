from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Đọc dữ liệu
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

tfidf = TfidfVectorizer(ngram_range=(1, 3))

def predict_model(model, X_train, y_train, X_test):
    tfidf.fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def validate_model(model):
    train = pd.read_csv('data/train.csv')
    train = train.dropna(subset=['question_text'])
    X = train.question_text
    y = train.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = predict_model(model, X_train, y_train, X_test)
    return f1_score(predictions, y_test)

# Tạo các mô hình
models = {
    'LinearSVC': LinearSVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'BernoulliNB': BernoulliNB(),
}

# In kết quả F1-Score của từng mô hình trên tập validation
for model_name, model in models.items():
    print(f'F1-Score of {model_name} on the validation: {validate_model(model)}')
