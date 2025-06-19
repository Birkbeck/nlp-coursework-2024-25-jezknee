import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

pd.set_option("display.max_columns", None)

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv")
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
    f1_score_rf = f1_score(rf_predictions,ytest, average=None)
    f1_score_svm = f1_score(svm_predictions, ytest, average=None)
    report_rf = classification_report(rf_predictions,ytest,zero_division=0)
    report_svm = classification_report(svm_predictions,ytest)

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
print_all_results(first_rf_predictions, first_svm_predictions, y_test)

# now adjusting the Vectoriser parameters - 2(d)
# looked ngram parameter up in the sckkitlearn official documentation
vectorizer_second = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2, 3))
x_train2, x_test2, y_train2, y_test2 = train_test_split(final_hansard_df["speech"], final_hansard_df["party"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
X_train2 = vectorizer.fit_transform(x_train2)
X_test2 = vectorizer.transform(x_test2)

second_rf_predictions = rf_model_train_and_predict(X_train2, y_train2, X_test2)
second_svm_predictions = svm_model_train_and_predict(X_train2, y_train2, X_test2)
print_all_results(second_rf_predictions, second_svm_predictions, y_test2)
