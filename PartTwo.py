import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

pd.set_option("display.max_columns", None)

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv", encoding='utf-8', encoding_errors='replace')
hansard_df = pd.DataFrame(text)
hansard_df = hansard_df.replace("Labour (Co-op)", "Labour")
filtered_hansard_df = hansard_df[hansard_df["party"] != "Speaker"]
most_common_parties = filtered_hansard_df["party"].value_counts().index.tolist()[:4]
common_hansard_df = hansard_df[hansard_df["party"].isin(most_common_parties)]
final_hansard_df = common_hansard_df[common_hansard_df["speech"].str.len() >= 1000]

print(final_hansard_df.shape)

# originally copied the vectorizer code from a class project (lab 4), then modified it
#t0 = time()
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

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
    report_svm = classification_report(svm_predictions,ytest, zero_division=0)

    print("RF f1")
    print(f1_score_rf)
    print("svm f1")
    print(f1_score_svm)
    print("RF classification report")
    print(report_rf)
    print("SVM classification report")
    print(report_svm)

# GET THIS BIT BACK LATER

first_rf_predictions = rf_model_train_and_predict(X_train, y_train, X_test)
first_svm_predictions = svm_model_train_and_predict(X_train, y_train, X_test)
print("Prediction 1: Original Tokenizer, question 2c")
print_all_results(first_rf_predictions, first_svm_predictions, y_test)

# now adjusting the Vectoriser parameters - 2(d)
# looked ngram parameter up in the sckkitlearn official documentation
vectorizer_second = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 3))
x_train2, x_test2, y_train2, y_test2 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
X_train2 = vectorizer_second.fit_transform(x_train2)
X_test2 = vectorizer_second.transform(x_test2)

second_rf_predictions = rf_model_train_and_predict(X_train2, y_train2, X_test2)
second_svm_predictions = svm_model_train_and_predict(X_train2, y_train2, X_test2)
print("Prediction 2: slightly improved model, question 2d")
print_all_results(second_rf_predictions, second_svm_predictions, y_test2)

"""
# first I'm going to try my tokeniser from Part 1, just to see what happens

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 20000
#nlp.max_length = 12000


vectorizer_third = TfidfVectorizer(max_features=3000, tokenizer=nlp)
x_train3, x_test3, y_train3, y_test3 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
X_train3 = vectorizer_third.fit_transform(x_train3)
X_test3 = vectorizer_third.transform(x_test3)

third_rf_predictions = rf_model_train_and_predict(X_train3, y_train3, X_test3)
third_svm_predictions = svm_model_train_and_predict(X_train3, y_train3, X_test3)
print("Prediction 3: PartOne nlp tokenizer")
print_all_results(third_rf_predictions, third_svm_predictions, y_test3)

"""
# now attempting to improve performance by doing feature selection
# I've read through the official scikitlearn documentation again and got these ideas from there
"""
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
"""

#def my_tokeniser():
"""
# I got this idea from the scikitlearn documentation - https://scikit-learn.org/stable/modules/feature_selection.html
def feature_select(X, y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    vectorizer_second = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 3))

    X_vector = vectorizer_second.fit_transform(X)
    X_new = SelectKBest(f_classif, k=3000).fit(X_vector, y)
    return X_new

x_train4, x_test4, y_train4, y_test4 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
X_selectBest = feature_select(x_train4, y_train4)
x_train4a = TfidfVectorizer(min_df = 1000, vocabulary=X_selectBest)
X_train4b = x_train4a.fit_transform(x_train4)
#vectorizer_second.fit_transform(X_train4)
X_test4 = feature_select(x_test4, y_test4)

fourth_rf_predictions = rf_model_train_and_predict(X_train4b, y_train4, X_test4)
fourth_svm_predictions = svm_model_train_and_predict(X_train4b, y_train4, X_test4)
print_all_results(fourth_rf_predictions, fourth_svm_predictions, y_test4)
"""
"""
# copied this sample code from the scikitlearn docs, then customised it: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
def custom_tokenizer(doc, features, train_or_test):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    corpus = doc
    # originally an error - seems that vocabulary needs to consist of unique values - got this idea from https://github.com/scikit-learn/scikit-learn/issues/2634
    vocabulary = list(set(features))
    pipe = Pipeline([('count', CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 3), vocabulary=vocabulary, lowercase =False)),
                    ('tfid', TfidfTransformer())]).fit(corpus)
    if train_or_test:
        pipe['count'].fit_transform(doc)
        #feature_names = pipe['count'].get_feature_names_out()
        #print(feature_names)
        #print(pipe['count2'].toarray())
    elif not train_or_test:
        pipe['count'].transform(doc)

    #df = pd.DataFrame(pipe['tfid'].toarray(), columns = pipe.get_feature_names_out())
    #print(df)
    pipe['tfid'].idf_
    #X_output = pipe.transform(corpus)
    #return X_output
"""

def custom_tokenizer(doc):
    import spacy
    from spacy.tokenizer import Tokenizer
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1200000
    t = nlp(doc)
    tokens = []
    for i in t:
        tokens.append(i.lemma_)
    return tokens


    """
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'(?u)\\b\\w\\w+\\b')
    """
    return nlp

#return tokenizer



#vectorizer_custom = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 3), tokenizer=custom_tokenizer)
x_train5, x_test5, y_train5, y_test5 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
#tk5 = custom_tokenizer(str(x_train5))
print("tokenising documents...")
v = CountVectorizer(max_features=3000, ngram_range=(1,3), encoding="utf-8", decode_error='replace', tokenizer=custom_tokenizer)
from sklearn.feature_selection import VarianceThreshold
print("fitting model...")
X = v.fit_transform(x_train5)
print("doing features selection...")
sel = VarianceThreshold(threshold=(.6 * (1 - .6))) # added feature selection, performance decreased from 0.84 to 0.78
X = sel.fit_transform(X)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


"""
try:
    #z = v.vocabulary_
    for i in X:
         print(i)
except:
        pass
        
w = v.get_feature_names_out()
"""

a = TfidfTransformer()
print("transforming fitted model...")
b = a.fit_transform(X)


"""
try:
    for i in b:
         print(i)
except:
        pass
"""

#df = pd.DataFrame(X.toarray(), columns=w)
#print(df.head())
#z = X.toarray()
print("fitting test data...")
X_train5 = b
X_test5a = v.transform(x_test5)
X_test5b = sel.transform(X_test5a)
print("transforming fitted test data")
X_test5 = a.transform(X_test5b)
print("training Random Forest model...")
rf_predictions5 = rf_model_train_and_predict(X_train5, y_train5, X_test5)
print("training SVM model...")
svm_predictions5 = svm_model_train_and_predict(X_train5, y_train5, X_test5)
print("Prediction 5, custom tokenizer, question 2e")
print_all_results(rf_predictions5, svm_predictions5, y_test5)
