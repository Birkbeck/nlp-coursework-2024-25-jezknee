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

    # couldn't find formula in lecture notes, so got it from Microsoft: https://support.microsoft.com/en-gb/office/get-your-document-s-readability-and-level-statistics-85b4969e-e80a-4777-8dd3-f7fc3c8b3fd2#__toc342546558
    #(0.39 x ASL) + (11.8 x ASW) – 15.59
    #where:
    #ASL = average sentence length (the number of words divided by the number of sentences)
    #ASW = average number of syllables per word (the number of syllables divided by the number of words)
    tok_text = nltk.word_tokenize(text)
    sent_text = nltk.sent_tokenize(text)
    total_words = len(tok_text)
    asl = total_words / len(sent_text)
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'\w+')
    tok_text_no_punc = tk.tokenize(text)

    total_syllables = 0
    for token in tok_text_no_punc:
        word_syll_count = count_syl(token, d)
        total_syllables += word_syll_count

    asw = total_syllables / total_words

    # then the calculation
    fk_result = (0.39*asl) + (11.8*asw) - 15.59
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

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    #1(a)(i)
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
    novels_df = pd.DataFrame(texts)
    #1(a)(i)
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
    return n_df

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    parsed_doc_list = []
    for i in df["text"]:
        j = parse_text_with_spacy(i)
        parsed_doc_list.append(j)
    df["parsed_text"] = parsed_doc_list
    
    store_path=Path.cwd() / "pickles" / "parsed.pickle"
    df.to_pickle(store_path)
    return df

def parse_text_with_spacy(text):
    from spacy.tokenizer import Tokenizer
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1200000
    tok_text = nlp(text)
    return tok_text


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # This function should return a dictionary mapping the title of each
    # novel to its type-token ratio. Tokenize the text using the NLTK library only.
    # Do not include punctuation as tokens, and ignore case when counting types.

    # I used nltk's regex tokenizer, as detailed here: https://www.geeksforgeeks.org/python-nltk-tokenize-regexp/
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'\w+') # this just gets the letters or digits, so no punctuation
    tok_text_no_punc = tk.tokenize(text)
    toks = []
    for i in tok_text_no_punc:
        j = i.lower()
        toks.append(j)

    types = set()
    for i in toks:
        types.add(i)

    type_token_ratio = round(len(types) / len(toks), 3)
    return type_token_ratio

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

def subjects_by_verb2(doc, verb):
    # at this point I went back to the spacy documentation and read it more carefully!
    # interpreting 'syntactic subject of to hear' as the spacy-defined nsubj of the verb from the dependency parsing
    # Note - I'm not completely satisfied with this - too many pronouns rather than named entities, would have liked to extract the named entities referred to by the pronouns
    # didn't find an easy way of getting named entities from pronouns - found a long discussion here: https://explosion.ai/blog/coref
    # ultimately, I ran out of time to find the named entities from the pronouns, and the link above references an experimental library, which I thought was beyond the scope of the assignment (mentions just using spacy)
    subjects = dict()    
    for token in doc:
        if token.pos_ == "VERB":
            if token.lemma_ == verb:
                v = []
                for t in token.children:
                    if t.dep_ == "nsubj":
                        v.append(t)
                        if t.lemma_ not in subjects:
                            subjects[t.lemma_] = 1
                        elif t.lemma_ in subjects:
                            subjects[t.lemma_] += 1
    return subjects

def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    # Im going to interpret this as the PMI between "to hear" and the subject

    verb_dict = subjects_by_verb2(doc, target_verb)
    verb_count = 0
    pmi_dict = dict()
    # total words in document
    all_tokens = 0
    # number of times verb occurs in document
    for token in doc:
        all_tokens += 1
        if token.lemma_ == target_verb:
            verb_count += 1
    #print("Verb Count:" + str(verb_count))
    # number of times noun occurs in document
    for key in verb_dict:
        token_count = 0
        for token in doc:
            if token.lemma_ == key:
                token_count += 1
        # getting the number of times the verb and noun co-occur
        try:
            covalue = verb_dict[key]
        except:
            covalue = 0
        pmi_dict[key] = (covalue, verb_count, token_count)
    #print("All Tokens:" + str(all_tokens))
    #print("Noun Count:" + str(token_count))
    #print("All Values", pmi_dict)

    dict_to_return = dict()
    for j in pmi_dict:
        cooccurrence = pmi_dict[j][0]
        verb_occurrence = pmi_dict[j][1]
        noun_occurrence = pmi_dict[j][2]

        prob_verb_and_noun = (cooccurrence / all_tokens)
        #print("Pw1w2: " + str(prob_verb_and_noun))
        prob_verb_times_prob_noun = (verb_occurrence / all_tokens) * (noun_occurrence / all_tokens)
        #print("Pw1_times_Pw2: " + str(prob_verb_times_prob_noun))
        prob_to_log = prob_verb_and_noun / prob_verb_times_prob_noun
        #print("Prob to log: " + str(prob_to_log))
        import math 
        final_probability = round(math.log(prob_to_log),4)
        #print("Final probability: " + str(final_probability))
        dict_to_return[j] = final_probability
    #print(dict_to_return)

    results_top_10 = sorted(dict_to_return.items(), key=lambda item: item[1], reverse = True)[:9]
    return results_top_10

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    verb_dict = subjects_by_verb2(doc, verb)
    results_top_10 = sorted(verb_dict.items(), key=lambda item: item[1], reverse = True)[:9]
    return results_top_10

def find_dobj(doc):
    # re-reading spacy parser, identifying syntactic objects by 'dobj'
    # experimented with excluding pronouns, as they're the most common - without a good way of getting named entities from pronouns, this didn't work well, as it simply excluded many syntactic objects
    all_dobj = dict()
    for token in doc:
        token_text = token.lemma_
        if token.dep_ == "dobj":
            if token_text not in all_dobj: # and token.pos_ != "PRON":
                all_dobj[token_text] = 1
            elif token_text in all_dobj:
                all_dobj[token_text] += 1
    return all_dobj

def object_counts(doc):
    """Extracts the most common nouns in a parsed document. Returns a list of tuples."""
    row_results = find_dobj(doc)
    results_top_10 = sorted(row_results.items(), key=lambda item: item[1], reverse = True)[:9]
    return results_top_10

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    print(df.dtypes)
    nltk.download("cmudict")
    parse(df)
    #print(df.head())
    print("1(b)")
    print(get_ttrs(df))
    print("1(c)")
    print(get_fks(df))
    """
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print("1(e)")
    print(df.head())
    print(df.dtypes)

    print("1(f)(i)")
    for i, row in df.iterrows():
        print(row["title"])
        print(object_counts(row["parsed_text"]))
        print("\n")

    print("1(f)(ii)")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed_text"], "hear"))
        print("\n")

    print("1(f)(iii)")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed_text"], "hear"))
        print("\n")

