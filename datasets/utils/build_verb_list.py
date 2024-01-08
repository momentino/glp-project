import os
import json

import nltk

from nltk.corpus import wordnet as wn

import spacy
nlp = spacy.load('en_core_web_sm')


VALSE_ROOT = '../VALSE'
ARO_ROOT = '../ARO'

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0

def build_verb_list():
    verb_list = []
    # these are verbs that because of grammatical mistakes are into sentences that may be transitive, but after looking at the picture,
    # I can confirm they were not. However the algorithm consider the sentences with these verbs transitive because they were ambiguous or grammatically incorrect.
    # For this reason, I created this small list
    extra_intransitive = ['rot', 'pounce', 'disembark', 'camouflage', 'drop']
    verb_list_path = '../Verb List/verb_list.txt'
    if(is_file_empty(verb_list_path)):
        with open('../Verb List/verb_list.txt', 'a') as f:

            transitive_actant_swap = {}
            with open(os.path.join(VALSE_ROOT,'original_actant_swap.json')) as d:
                data = json.load(d)
                for key, value in data.items():
                    caption = value['caption']
                    tagged_caption = nltk.pos_tag(nltk.word_tokenize(caption))
                    #print(tagged_caption)
                    caption_tag_list = [t[1] for t in tagged_caption]
                    if( 'IN' not in caption_tag_list and 'RP' not in caption_tag_list and 'TO' not in caption_tag_list):
                        for t in tagged_caption:
                            if 'VB' in t[1]:
                                doc = nlp(t[0])
                                lemmatized_verb = [token.lemma_ for token in doc][0]
                                # create the actual dataset
                                transitive_actant_swap[key] = value
                                # create the verb list
                                if(lemmatized_verb not in verb_list and lemmatized_verb not in extra_intransitive):
                                    f.write(lemmatized_verb+'\n')
                                    print(caption)
                                    print(tagged_caption)
                                    verb_list.append(lemmatized_verb)
            with open(os.path.join(VALSE_ROOT,'transitive_actant_swap.json'),'w') as s:
                json.dump(transitive_actant_swap, s)
            transitive_aro = {}
            with open(os.path.join(ARO_ROOT,'original_visual_genome_relation.json')) as d:
                data = json.load(d)
                count = 0
                for elem in data:
                    caption = elem['true_caption']

                    tagged_caption = nltk.pos_tag(nltk.word_tokenize(caption))
                    caption_tag_list = [t[1] for t in tagged_caption]
                    if ('IN' not in caption_tag_list and 'RP' not in caption_tag_list and 'TO' not in caption_tag_list):
                        for t in tagged_caption:
                            if 'VBG' in t[1]:
                                doc = nlp(t[0])
                                lemmatized_verb = [token.lemma_ for token in doc][0]
                                # create the actual dataset
                                transitive_aro[count] = elem
                                count += 1
                                # create the verb list
                                if (lemmatized_verb not in verb_list and lemmatized_verb not in extra_intransitive):
                                    f.write(lemmatized_verb + '\n')
                                    print(caption)
                                    print(tagged_caption)
                                    verb_list.append(lemmatized_verb)

            with open(os.path.join(ARO_ROOT,'transitive_visual_genome_relation.json'),'w') as s:
                json.dump(transitive_aro, s)

if __name__ == '__main__':
    build_verb_list()