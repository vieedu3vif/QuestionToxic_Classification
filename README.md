# Question Toxic Classification

## Introduction
 This project is final project of Marchine Learning courses (INT3405), University of Engineering and Techonology

## Authors

- [Nguyen Viet Duc- 21020904, UET](https://github.com/vieedu3vif)
- [Nguyen Anh Duc-21020902, UET](https://github.com/FelixMeowMeow)
- [Trinh Minh Chien-21020890, UET](https://github.com/vamostmc)

## About this proeject
- We use LinearSVC to predict whether a question is insincere
- Type of the data: Text data

## How to run
```
1. Install Git LFS to work with big file in github

 git lfs install

2. Clone the repository

 git clone https://github.com/vieedu3vif/QuestionToxic_Classification.git

3.Navigate to the project directory

 cd QuestionToxic_Classification

 pip install -r requirements.txt

 python main.py
```

### Preprocessing data
- Switch back to lowercase letters 
- Remove mathematical expressions, URL links, numbers, and special characters
- Convert, shorten words and return to the original word
- Eliminate stopwords like a, an, or, of, the, do, does, ... 
- Create additional columns to evaluate the frequency of words

### Optimal
- We used textaugment
- We tried with undersampling
## Demo
[Video demo](https://www.youtube.com/watch?v=Y2ylxM-gNEg)