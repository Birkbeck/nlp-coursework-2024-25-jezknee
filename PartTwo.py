import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv")
hansard_df = pd.DataFrame(text)
#print(hansard_df.head())
#print(hansard_df.count())
print(hansard_df.shape)
hansard_df = hansard_df.replace("Labour (Co-op)", "Labour")
#hansard_df = hansard_df.replace("Labour", "test")
#print(hansard_df.head())
filtered_hansard_df = hansard_df[hansard_df["party"] != "Speaker"]
# got top 5 parties because "Speaker" was number 4, and I need to remove that too
most_common_parties = filtered_hansard_df["party"].value_counts().index.tolist()[:4]
#print(most_common_parties)
common_hansard_df = hansard_df[hansard_df["party"].isin(most_common_parties)]
#print(common_hansard_df.head())
#print(common_hansard_df["party"].value_counts())
final_hansard_df = common_hansard_df[common_hansard_df["speech"].str.len() >= 1000]
#print(final_hansard_df.head())
#print(final_hansard_df["party"].value_counts())
#print(final_hansard_df["speech"].str.len())

print(final_hansard_df.shape)

# originally copied the vectorizer code from a class project, then modified it
#t0 = time()
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

x_train, x_test = train_test_split(final_hansard_df["speech"], test_size=0.3, random_state = 26, stratify=final_hansard_df["party"])
print(x_train)
print(x_test)

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)
#duration_train = time() - t0
