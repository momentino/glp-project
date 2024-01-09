import os
import json

import nltk

path = "../VALSE/transitive_actant_swap.json"

def valse_foil_analysis():
    with open(path) as d:
        data=json.load(d)
        
        total_count=len(data)
        foil_count=0

        for key, value in data.items():
            foil = value['foil']
            tagged_foil = nltk.pos_tag(nltk.word_tokenize(foil))
            foil_tag_list = [t[1] for t in tagged_foil]

            if( 'IN' in foil_tag_list or 'RP' in foil_tag_list or 'TO' in foil_tag_list):
                foil_count+=1
        
        print("total number of samples: {}".format(total_count))
        print("number of wrong foil samples: {}".format(foil_count))

if __name__ == '__main__':
    valse_foil_analysis()