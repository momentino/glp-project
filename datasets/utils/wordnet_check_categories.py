import pandas as pd
from nltk.corpus import wordnet
import os


def check_categories(df):
    synsets = wordnet.synsets(verb, pos='v')
    if synsets:
        return synsets[0].pos()
    else:
        return None
def get_verb_category(verb):
    synsets = wordnet.synsets(verb, pos='v')
    print(synsets)
    if synsets:
        for s in synsets:
            print(s.lexname())
    else:
        return None

verb = 'read'
category = get_verb_category(verb)
if category:
    print(f"The category of the verb '{verb}' is {category}.")
else:
    print(f"No synsets found for the verb '{verb}'.")

if __name__ == '__main__':
    df = pd.read_csv('../Verb List/Verb Categories/verb_list.csv')
    check_categories(df)