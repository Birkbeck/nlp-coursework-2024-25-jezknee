import pandas as pd

text = pd.read_csv(r"C:\Users\jezkn\OneDrive\Documents\Birkbeck\Work\Natural Language Processing\Coursework\nlp-coursework-2024-25-jezknee\p2-texts\hansard40000.csv")
hansard_df = pd.DataFrame(text)
print(hansard_df.head())
