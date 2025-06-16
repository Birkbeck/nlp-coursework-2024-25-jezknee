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
    pass

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
    df["parsed_text"] = parsed_doc_list
    
    store_path=Path.cwd() / "pickles" / "parsed.pickle"
    df.to_pickle(store_path)
    return df

    pass

def parse_text_with_spacy(text):
    from spacy.tokenizer import Tokenizer
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1200000
    tok_text = nlp(text)

    # Construction via add_pipe with default model
    #parser = nlp.add_pipe("parser")
    return tok_text
    #return parser


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # This function should return a dictionary mapping the title of each
    # novel to its type-token ratio. Tokenize the text using the NLTK library only.
    # Do not include punctuation as tokens, and ignore case when counting types.

    # I used nltk's regex tokenizer, as detailed here: https://www.geeksforgeeks.org/python-nltk-tokenize-regexp/
    from nltk.tokenize import RegexpTokenizer
    tk = RegexpTokenizer(r'\w+')
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

    pass

#path1 = Path.cwd()
#test_df = read_novels(path1)
#test_text = test_df["text"][0]z
#parse_text_with_spacy(test_text)
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

def subjects_by_verb(doc, verb):
    temp_counter = -2
    token_counter = 0
    verb_list = dict()
    for token in doc:
        if token_counter == temp_counter + 1 and (token.pos_ == "NOUN" or token.pos_ == "PROPN"):
            verb_subject = token.lemma_
            if verb_subject not in verb_list:
                verb_list[verb_subject] = 1
            elif verb_subject in verb_list:
                verb_list[verb_subject] += 1
            temp_counter = -2
        elif token_counter == temp_counter + 1:
            token_counter += 1
            temp_counter += 1
        elif token.lemma_ == verb:
            temp_counter = token_counter
        token_counter += 1
    return verb_list

def object_verb_dependent(doc, verb):
    # at this point I went back to the spacy documentation and read it more carefully!
    # interpreting 'syntactic subject of to hear' as agent that does the hearing
    #
    #object_verb_list = dict()
    #for token in doc:
        #if token.lemma_ == verb:
            #root = [token for token in doc if token.head == token][0]
            #subject = list(root.lefts)[0]
            #for descendant in subject.subtree:
                #assert subject is descendant or subject.is_ancestor(descendant)
                #print(descendant.text, descendant.dep_, descendant.n_lefts, descendant.n_rights, [ancestor.text for ancestor in descendant.ancestors])
    """
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text)
    """
    # first proper draft after reading the documents
    # I'll come back to this and work out how to get the people that the pronouns refer to
    subjects = dict()    
    for token in doc:
        if token.pos_ == "VERB":
            if token.lemma_ == verb:
                v = []
                #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
                for t in token.lefts:
                    #print("lefts: ", t.lemma_, t.pos_, t.tag, t.dep_, t.shape_, t.is_alpha, t.is_stop)
                    if t.pos_ == "NOUN" or t.pos_ == "PNOUN" or t.pos_ == "PRON":
                        v.append(t)
                        if t.lemma_ not in subjects:
                            subjects[t.lemma_] = 1
                        elif t.lemma_ in subjects:
                            subjects[t.lemma_] += 1
                #for t in token.rights:
                    #print("Rights: ", t.lemma_, t.pos_, t.tag, t.dep_, t.shape_, t.is_alpha, t.is_stop)
                #print(v)
    #print(subjects)
    return subjects

"""
    for token in doc:
        verb_nlp = nlp(verb)
        for i in verb_nlp:
            if token.lemma_ == i.lemma_:
                verb_subtree = token.ancestors
                #print(verb_subtree)
                a = []
                for ancestor in verb_subtree:
                    #a = []
                    #print(ancestor)
                    #if ancestor.pos_ == "NOUN" or ancestor.pos_ == "PNOUN":
                    a.append(ancestor)
                    #print(ancestor)
                    if ancestor.lemma_ not in object_verb_list:
                        object_verb_list[ancestor.lemma_] = 1
                    elif ancestor.lemma_ in object_verb_list:
                        object_verb_list[ancestor.lemma_] += 1
                print(a)
    return object_verb_list
"""

def subjects_by_verb_pmi(doc, target_verb):
    # Im going to interpret this as the PMI between "to hear" and the subject
    verb_dict = object_verb_dependent(doc, target_verb)
    verb_count = 0
    # number of times verb occurs in document
    for token in doc:
        if token.lemma_ == target_verb:
            verb_count += 1
    print("Verb Count:" + str(verb_count))
    # number of times noun occurs in document
    for key in verb_dict:
        verb_dict[key] = []
        token_count = 0
        for token in doc:
            if token.lemma_ == key:
                token_count += 1
    print("Noun Count:" + str(token_count))
    # number of times noun occurs near verb

    

    # probability of them occurring together over probability of one times probability of the other
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    

    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    verb_dict = object_verb_dependent(doc, verb)
    results_top_10 = sorted(verb_dict.items(), key=lambda item: item[1], reverse = True)[:9]
    return results_top_10

    pass

def find_objects(doc):
    # I saw the idea of noun chunks somewhere in the documentation, thought it might be better than just getting nouns
    all_objects = dict()
    for i in doc.noun_chunks:
        if i not in all_objects:
            all_objects[i] = 1
        elif i in all_objects:
            all_objects[i] += 1
    #print(all_adjectives)
    return all_objects

def object_counts(doc):
    results = {}
    for i, row in df.iterrows():
        row_results = find_objects(row["parsed_text"])
        results_top_10 = sorted(row_results.items(), key=lambda item: item[1], reverse = True)[:9]
        results[row["title"]] = results_top_10
    return results

def find_adjectives(doc):
    # copied this from something I did in a class exercise
    all_adjectives = dict()
    for token in doc:
        token_text = token.lemma_
        if token.pos_ == "ADJ":
            if token_text not in all_adjectives:
                all_adjectives[token_text] = 1
            elif token_text in all_adjectives:
                all_adjectives[token_text] += 1
    #print(all_adjectives)
    return all_adjectives

def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    # it didn't specifically ask about this in the instructions, but the sample function was here
    # I thought I'd just create it anyway, really isn't any extra work, given that I had to do the same thing for syntactic objects
    results = {}
    for i, row in df.iterrows():
        row_results = find_adjectives(row["parsed_text"])
        results_top_10 = sorted(row_results.items(), key=lambda item: item[1], reverse = True)[:9]
        results[row["title"]] = results_top_10
    return results

    pass

def find_nouns(doc):
    # copied this from something I did in a class exercise
    # I've assumed that 'syntactic objects' means actual objects, i.e. words denoted by a noun
    # I suppose it could also have meant any token
    all_nouns = dict()
    for token in doc:
        token_text = token.lemma_
        if token.pos_ == "NOUN" or token.pos == "PROPN":
            if token_text not in all_nouns:
                all_nouns[token_text] = 1
            elif token_text in all_nouns:
                all_nouns[token_text] += 1
    return all_nouns

def noun_counts(doc):
    """Extracts the most common nouns in a parsed document. Returns a list of tuples."""
    # defining syntactic objects as nouns and proper nouns
    results = {}
    for i, row in df.iterrows():
        row_results = find_nouns(row["parsed_text"])
        results_top_10 = sorted(row_results.items(), key=lambda item: item[1], reverse = True)[:9]
        results[row["title"]] = results_top_10
    return results

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #print(df.dtypes)
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(df.head())
    print(df.dtypes)
    print(adjective_counts(df))
    print(noun_counts(df))
    #print(object_counts(df))

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed_text"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed_text"], "hear"))
        print("\n")

