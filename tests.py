from pathlib import Path

path = Path.cwd()

#file_info = []
#texts = []
#for file_path in path.rglob("*"):
    #if file_path.suffix == ".txt":
    #print(Path(file_path).name)
    #print(file_path.name)
    #file_info.append(Path(file_path).name)
        #with open(file_path,'r') as fp:
            #text = fp.readlines()
            #texts.append(text)
#print(file_info)



def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    file_info = []
    texts = []
    for file_path in path.rglob("*"):
        print(1)
        #if file_path.suffix == ".txt":
        print(Path(file_path).name)
        print(file_path.name)
        file_info.append(Path(file_path).name)
            #with open(file_path,'r') as fp:
                #text = fp.readlines()
                #texts.append(text)
    #print(file_info)


#read_novels()
#import pandas as pd
#import nltk
#cmudict = nltk.corpus.cmudict.dict()
#print(cmudict)
#print(cmudict["Chapter"])
#pd.DataFrame(cmudict)
#print(cmudict.head())
store_path=Path.cwd() / "pickles" / "parsed.pickle"
#, out_name="parsed.pickle"
print(store_path)