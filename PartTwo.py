import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np

pd.set_option("display.max_columns", None)

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv", encoding='utf-8', encoding_errors='replace')
hansard_df = pd.DataFrame(text)
hansard_df = hansard_df.replace("Labour (Co-op)", "Labour")
filtered_hansard_df = hansard_df[hansard_df["party"] != "Speaker"]
filtered_hansard_df = filtered_hansard_df[filtered_hansard_df["speech_class"] == "Speech"]
most_common_parties = filtered_hansard_df["party"].value_counts().index.tolist()[:4]
common_hansard_df = hansard_df[hansard_df["party"].isin(most_common_parties)]
final_hansard_df = common_hansard_df[common_hansard_df["speech"].str.len() >= 1000]

print(final_hansard_df.shape)

# originally copied the vectorizer code from a class project (lab 4), then modified it
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")

x_train, x_test, y_train, y_test = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

def rf_model_train_and_predict(xtrain, ytrain, xtest):
    rf_model = RandomForestClassifier(n_estimators=300)
    rf_model.fit(X=xtrain, y=ytrain)
    rf_predictions = rf_model.predict(xtest)
    return rf_predictions
    
def svm_model_train_and_predict(xtrain, ytrain, xtest):
    svm_model = svm.LinearSVC()
    svm_model.fit(X=xtrain, y=ytrain)
    svm_predictions = svm_model.predict(xtest)
    return svm_predictions

def print_all_results(rf_predictions, svm_predictions, ytest):
    f1_score_rf = f1_score(ytest, rf_predictions, average=None, zero_division=0)
    f1_score_svm = f1_score(ytest, svm_predictions, average=None, zero_division=0)
    report_rf = classification_report(ytest, rf_predictions,zero_division=0)
    report_svm = classification_report(ytest, svm_predictions, zero_division=0)

    print("RF f1")
    print(f1_score_rf)
    print("svm f1")
    print(f1_score_svm)
    print("RF classification report")
    print(report_rf)
    print("SVM classification report")
    print(report_svm)


first_rf_predictions = rf_model_train_and_predict(X_train, y_train, X_test)
first_svm_predictions = svm_model_train_and_predict(X_train, y_train, X_test)
print("Prediction 1: Original Tokenizer, question 2c")
print_all_results(first_rf_predictions, first_svm_predictions, y_test)

# now adjusting the Vectoriser parameters - 2(d)
# looked ngram parameter up in the sckkitlearn official documentation
vectorizer_second = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 3))
x_train2, x_test2, y_train2, y_test2 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
X_train2 = vectorizer_second.fit_transform(x_train2)
X_test2 = vectorizer_second.transform(x_test2)

second_rf_predictions = rf_model_train_and_predict(X_train2, y_train2, X_test2)
second_svm_predictions = svm_model_train_and_predict(X_train2, y_train2, X_test2)
print("Prediction 2: slightly improved model, question 2d")
print_all_results(second_rf_predictions, second_svm_predictions, y_test2)

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1200000000

# got my regex from https://stackoverflow.com/questions/14596884/remove-text-between-and
# I have used regex quite a few times before, but it takes me a long time to figure out, and I decided doing it myself wasn't really the point of this assignment!
def custom_preprocessor(doc):
    import re
    d = re.sub("[\(\[].*?[\)\]]", "", doc)
    return d

def custom_tokenizer_objects(doc):
    t = nlp(doc)
    tokens = []
    for i in t:
        if not i.is_stop and not i.is_punct:
            if  i.pos_ in ("NOUN", "PROPN", "ADJ", "VERB", "ADV"):
                tok = i.lemma_
                tokens.append(i.lemma_)
    return tokens

x_train5, x_test5, y_train5, y_test5 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
print("creating tokeniser...")
v_obj = CountVectorizer(max_features=5000,ngram_range=(1,3), min_df = 3, encoding="utf-8", tokenizer=custom_tokenizer_objects, preprocessor=custom_preprocessor)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
a = TfidfTransformer()

print("fitting model...")
X = v_obj.fit_transform(x_train5)

"""
# I got lots of these tests from NLP Lecture 8, and then modified them a bit

print("No. of features: ")
print({X.shape[0]})
print("Vocabulary size: ")
print({X.shape[1]})

def get_vectorizer_info(X):
    try:
        for speech in X:
            feature_names = v_obj.get_feature_names_out()
            word_count = np.asarray(speech.sum(axis=0)).ravel()
            word_freq = list(zip(feature_names, word_count))
            print("Top Words: ")
            top_words = sorted(word_freq, key = lambda x: x[1], reverse = True)[:20]
            print(top_words)
    except:
        print("could not print count of features")

"""
print("transforming fitted model...")
b = a.fit_transform(X)
print("doing features selection...")
# now attempting to improve performance by doing feature selection
# I've read through the official scikitlearn documentation again and got these ideas from there
sel = SelectKBest(score_func=chi2,k=1500)
X_train5 = sel.fit_transform(b, y_train5)
"""
try:
    #speech_index = 0
    for speech in X_train5:
        feature_names = v_obj.get_feature_names_out()
        word_count = np.asarray(speech.sum(axis=0)).ravel()
        word_freq = list(zip(feature_names, word_count))
        print("Top Words: ")
        top_words = sorted(word_freq, key = lambda x: x[1], reverse = True)[:20]
        print(top_words)
except:
    print("could not print count of features")
"""
print("fitting test data...")
X_test5a = v_obj.transform(x_test5)
X_test5b = a.transform(X_test5a)
print("transforming fitted test data")
X_test5c = sel.transform(X_test5b)
print("training Random Forest model...")
rf_predictions5 = rf_model_train_and_predict(X_train5, y_train5, X_test5c)
print("training SVM model...")
svm_predictions5 = svm_model_train_and_predict(X_train5, y_train5, X_test5c)
print("Prediction 5, custom tokenizer, question 2e")
print_all_results(rf_predictions5, svm_predictions5, y_test5)
