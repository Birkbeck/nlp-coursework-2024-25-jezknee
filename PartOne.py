#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """

        # This function should return a dictionary mapping the title of
    # each novel to the Flesch-Kincaid reading grade level score of the text. Use the
    # NLTK library for tokenization and the CMU pronouncing dictionary for estimating syllable counts.

    # the instructions didn't say whether to include punctuation or capitalisation
    # I decided to include it, as conceivably use of punctuation & capitalisation could affect the readability of a text
    
    # couldn't find formula in lecture notes, so got it from Microsoft: https://support.microsoft.com/en-gb/office/get-your-document-s-readability-and-level-statistics-85b4969e-e80a-4777-8dd3-f7fc3c8b3fd2#__toc342546558
    #(0.39 x ASL) + (11.8 x ASW) – 15.59
    #where:
    #ASL = average sentence length (the number of words divided by the number of sentences)
    #ASW = average number of syllables per word (the number of syllables divided by the number of words)
    tok_text = nltk.word_tokenize(text)
    sent_text = nltk.sent_tokenize(text)
    total_words = len(tok_text)
    asl = total_words / len(sent_text)
    #print(asl)
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'\w+')
    #tok_text = nltk.word_tokenize(text)
    tok_text_no_punc = tk.tokenize(text)

    total_syllables = 0
    for token in tok_text_no_punc:
        word_syll_count = count_syl(token, d)
        #print(word_syll_count)
        total_syllables += word_syll_count
    #print(total_syllables)

    asw = total_syllables / total_words
    #print(asw)

    # then the calculation
    fk_result = (0.39*asl) + (11.8*asw) - 15.59
    #print(fk_result)
    return fk_result

    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters
    
    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    w = word.lower()
    try:
        syllable = d[w]
        syllables_in_word = len(syllable)
        return syllables_in_word
    except:
        #print(w + " not in the dictionary.")
        return 0

    # should also take the word out of the total words count?
    # need to split these into morphemes instead of full words
    pass

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    #1(a)(i)
    #path = Path.cwd()
    file_info = []
    texts = []
    for file_path in path.rglob("*"):
        if file_path.suffix == ".txt" and '1' in str(file_path.name):
            novel = []
            with file_path.open(mode = "r", encoding="utf-8") as fp:
                novel_text = fp.read()
                title_parts = str(Path(file_path).stem).split("-")
                for t in title_parts:
                    novel.append(t)
                novel.append(novel_text)
                texts.append(novel)
                #print(novel)
    novels_df = pd.DataFrame(texts)
    novels_df.columns = ["title", "author", "year", "text"]
    #1(a)(ii)
    # I spent hours trying to sort my dataframe with the text included. 
    # Eventually I gave up and removed the text, then joined it back later.
    novels_df.reset_index(inplace=True)
    novels_df['year'] = pd.to_numeric(novels_df['year'])
    novelsdf_without_text = novels_df.filter(["title", "author", "year"])
    novelsdf_without_text = novelsdf_without_text.sort_values(by = ['year'], ascending = True)
    pd.set_option("display.max_columns", None)
    n_df = novelsdf_without_text.merge(novels_df[["title","text"]], how = 'left', on = ['title'])
    #print(n_df)
    return n_df
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    parsed_doc_list = []
    for i in df["text"]:
        j = parse_text_with_spacy(i)
        parsed_doc_list.append(j)
    df = df.assign(parsed_doc_list)
    return df
    
    pass

def parse_text_with_spacy(text):
    from spacy.tokenizer import Tokenizer
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1200000
    tok_text = nlp(text)
    return tok_text
    #print(type(text))
    #print(type(tok_text))
    #print(text)
    #print(tok_text)
    #for token in tok_text:
        #token_text = token.lemma_
        #print(token_text)


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # This function should return a dictionary mapping the title of each
    # novel to its type-token ratio. Tokenize the text using the NLTK library only.
    # Do not include punctuation as tokens, and ignore case when counting types.

    # I used nltk's regex tokenizer, as detailed here: https://www.geeksforgeeks.org/python-nltk-tokenize-regexp/
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'\w+')
    #tok_text = nltk.word_tokenize(text)
    tok_text_no_punc = tk.tokenize(text)
    #print(tok_text)
    #print(tok_text_no_punc)
    toks = []
    for i in tok_text_no_punc:
        j = i.lower()
        toks.append(j)
    #print(toks) 

    types = set()
    for i in toks:
        types.add(i)
    #print(types)

    type_token_ratio = round(len(types) / len(toks), 3)
    #print(type_token_ratio)
    return type_token_ratio

    pass

path1 = Path.cwd()
test_df = read_novels(path1)
test_text = test_df["text"][0]
parse_text_with_spacy(test_text)
#test_text = "Here are some words. I am writing a sentence or two or three. However, the longer words make this harder to read."
#nltk_ttr(test_text)
#cmudict = nltk.corpus.cmudict.dict()
#from nltk.tokenize import RegexpTokenizer
#tk = RegexpTokenizer(r'\w+')
#tok_text = nltk.word_tokenize(text)
    
#test_t_nopunc = tk.tokenize(test_text)



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

#fk_level(test_text, cmudict)
#print(get_fks(test_df))

def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    #nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

