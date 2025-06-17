import pandas as pd
pd.set_option("display.max_columns", None)

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv")
hansard_df = pd.DataFrame(text)
print(hansard_df.head())
#print(hansard_df.count())
hansard_df = hansard_df.replace("Labour (Co-op)", "Labour")
#hansard_df = hansard_df.replace("Labour", "test")
print(hansard_df.head())
most_common_parties = hansard_df["party"].value_counts().index.tolist()[:4]
print(most_common_parties)