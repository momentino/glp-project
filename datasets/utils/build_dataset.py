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

def convert_to_passive_aro():
    verb_list_path = '../Verb List'
    with open(verb_list_path + '/verb_list_with_mistakes.txt', 'r') as f: # need to use this to match properly lemmatized verbs with their passives. The correct list is just for us.
        verbs = f.readlines()
        verbs = [v.replace('\n','') for v in verbs]  #remove newline char
    with open(verb_list_path + '/participle_verb_list.txt', 'r') as f:
        passive_verbs = f.readlines()
        passive_verbs = [v.replace('\n', '') for v in passive_verbs]  # remove newline char

    transitive_aro = {}
    with open(os.path.join(ARO_ROOT, 'transitive_visual_genome_relation.json')) as d:
        data = json.load(d)
        count = 0
        for key, value in data.items():
            caption = value['true_caption']
            print(caption)
            tagged_caption = nltk.pos_tag(nltk.word_tokenize(caption)) # tag caption
            passive_caption = []
            false_passive_caption = []
            for t in tagged_caption:
                if 'VBG' in t[1]:
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    verb_index = verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'being '+passive_verb+' by '
                    passive_caption.append(passivized_verb)
                else:
                    passive_caption.append(t[0])
            false_passive_caption = passive_caption.copy()
            """ swap object and subject """
            a = passive_caption[1]
            b = passive_caption[5]
            false_passive_caption[1] = b
            false_passive_caption[5] = a
            #print(" FALSE CAPTION ",false_passive_caption)
            #print(" TRUE CAPTION ",passive_caption)
            transitive_aro[key] = value
            transitive_aro[key]['true_passive_caption']=" ".join(passive_caption)
            transitive_aro[key]['false_passive_caption']=" ".join(false_passive_caption)
            #print(transitive_aro['true_passive_caption'], "                ", transitive_aro['false_passive_caption'])

    print(transitive_aro)
    with open(os.path.join(ARO_ROOT, 'transitive_visual_genome_relation.json'), 'w') as s:
        json.dump(transitive_aro, s)

def build_transitive_valse(verb_list):
    valse_verb_list = []
    extra_intransitive = ['rot', 'pounce', 'disembark', 'camouflage', 'drop']
    verb_list_path = '../Verb List'
    with open(verb_list_path+'/verb_list.txt', 'a') as f:
        if (is_file_empty(verb_list_path + '/valse_verb_list.txt')):
            with open(verb_list_path+'/valse_verb_list.txt', 'a') as valse_v:

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
                                    if (lemmatized_verb not in extra_intransitive and lemmatized_verb not in valse_verb_list):
                                        valse_v.write(lemmatized_verb + '\n')
                                        valse_verb_list.append(lemmatized_verb)
                                    if (lemmatized_verb not in extra_intransitive and lemmatized_verb not in verb_list):
                                        f.write(lemmatized_verb+'\n')
                                        print(caption)
                                        print(tagged_caption)
                                        verb_list.append(lemmatized_verb)
                with open(os.path.join(VALSE_ROOT,'transitive_actant_swap.json'),'w') as s:
                    json.dump(transitive_actant_swap, s)

def build_transitive_aro(verb_list):

    aro_verb_list = []
    # these are verbs that because of grammatical mistakes are into sentences that may be transitive, but after looking at the picture,
    # I can confirm they were not. However the algorithm consider the sentences with these verbs transitive because they were ambiguous or grammatically incorrect.
    # For this reason, I created this small list
    extra_intransitive = ['rot', 'pounce', 'disembark', 'camouflage', 'drop']
    verb_list_path = '../Verb List'

    transitive_aro = {}
    with open(verb_list_path + '/verb_list.txt', 'a') as f:
        with open(os.path.join(ARO_ROOT,'original_visual_genome_relation.json')) as d:
            if (is_file_empty(verb_list_path + '/aro_verb_list.txt')):
                with open(verb_list_path + '/aro_verb_list.txt', 'a') as aro_v:
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
                                    if (lemmatized_verb not in extra_intransitive and lemmatized_verb not in aro_verb_list):
                                        aro_v.write(lemmatized_verb + '\n')
                                        aro_verb_list.append(lemmatized_verb)
                                    if (lemmatized_verb not in extra_intransitive and lemmatized_verb not in verb_list):
                                        f.write(lemmatized_verb + '\n')
                                        print(caption)
                                        print(tagged_caption)
                                        verb_list.append(lemmatized_verb)

            with open(os.path.join(ARO_ROOT,'transitive_visual_genome_relation.json'),'w') as s:
                json.dump(transitive_aro, s)

if __name__ == '__main__':
    verb_list = []
    build_transitive_aro(verb_list)
    convert_to_passive_aro()