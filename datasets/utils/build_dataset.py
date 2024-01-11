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
        for key, value in data.items():
            caption = value['true_caption']
            foil = value['false_caption']
            print(caption)

            tagged_caption = nltk.pos_tag(nltk.word_tokenize(caption)) # tag caption
            tagged_foil = nltk.pos_tag(nltk.word_tokenize(foil)) # tag foil
            false_passive_caption = []
            passive_caption = []
            for t in tagged_caption:
                if 'VBG' in t[1]:
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    verb_index = verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'being '+passive_verb+' by'
                    false_passive_caption.append(passivized_verb)
                else:
                    false_passive_caption.append(t[0])
            for t in tagged_foil:
                if 'VBG' in t[1]:
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    verb_index = verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'being '+passive_verb+' by'
                    passive_caption.append(passivized_verb)
                else:
                    passive_caption.append(t[0])


            transitive_aro[key] = value


            transitive_aro[key]['true_passive_caption']=" ".join(passive_caption)
            transitive_aro[key]['false_passive_caption']=" ".join(false_passive_caption)
    print(transitive_aro)
    with open(os.path.join(ARO_ROOT, 'transitive_visual_genome_relation.json'), 'w') as s:
        json.dump(transitive_aro, s)

def build_transitive_valse(verb_list):
    valse_verb_list = []
    extra_intransitive = ['rot', 'pounce', 'disembark', 'camouflage', 'drop', 'ask', 'pee']
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
    extra_intransitive = ['rot', 'pounce', 'disembark', 'camouflage', 'drop', 'ask', 'pee']
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
                        foil = elem['false_caption']
                        # replace is with are when the sentence contains pants, jeans or trousers to keep the agreement
                        # Pay attention at the position of the element. Whether it is at the end or in the middle of the sentence

                        caption = caption.split(" ")
                        foil = foil.split(" ")
                        doc = nlp(caption[-1])
                        lemmatized_object_true = [token.lemma_ for token in doc][0]
                        # if plural
                        if(lemmatized_object_true != caption[-1] or (caption[-1] == "people") or (caption[-1] == "children")):
                            foil[foil.index("is")] = "are"
                        doc = nlp(foil[-1])
                        lemmatized_object_false= [token.lemma_ for token in doc][0]

                        if (lemmatized_object_false != foil[-1] or (foil[-1] == "people") or (foil[-1] == "children")):
                            caption[caption.index("is")] = "are"
                        caption = " ".join(caption)
                        foil = " ".join(foil)
                        elem['true_caption'] = caption
                        elem['false_caption'] = foil
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

def fix_aro_agreement():
    pass

def remove_aro_duplicates():
    transitive_aro = {}
    with open(os.path.join(ARO_ROOT, 'transitive_visual_genome_relation.json'), 'r+') as d:
        data = json.load(d)
        transitive_aro = data.copy()
        for key, values in data.items():
            for key2, values2 in data.items():
                #print(" KEY ",key, " KEY 2 ",key2)
                #print(values)
                if (key != key2 and (values['image_id'] == values2['image_id'])
                        and values['true_caption'] == values2['true_caption']
                        and values['relation_info']['object'] != values2['relation_info']['object']
                        and int(key)<int(key2)):
                    #print(" ECCO ")
                    try:
                        del transitive_aro[key2]
                    except:
                        print(" Already removed ")

    with open(os.path.join(ARO_ROOT, 'transitive_visual_genome_relation.json'), 'w') as s:
        json.dump(transitive_aro, s)

#create passive captions for VALSE
def convert_to_passive_valse():
    verb_list_path = '../Verb List'

    with open(verb_list_path + '/verb_list_with_mistakes.txt', 'r') as f:
        verbs = f.readlines()
        verbs = [v.replace('\n','') for v in verbs]  #remove newline char
    with open(verb_list_path + '/verb_list.txt', 'r') as f:
        correct_verbs = f.readlines()
        correct_verbs = [v.replace('\n','') for v in correct_verbs]  #remove newline char
    with open(verb_list_path + '/participle_verb_list.txt', 'r') as f:
        passive_verbs = f.readlines()
        passive_verbs = [v.replace('\n', '') for v in passive_verbs]  # remove newline char

    transitive_valse = {}
    with open(os.path.join(VALSE_ROOT, 'transitive_actant_swap.json')) as d:
        data = json.load(d)
        for key, value in data.items():
            
            #create false passive from true active (e.g. "the woman drives a car" --> "the woman is driven by a car")
            true_active = value['true_active']
            tagged_true_active = nltk.pos_tag(nltk.word_tokenize(true_active)) # tag caption
            foil_passive = []
            for t in tagged_true_active:
                if 'VBZ' in t[1]: # singular verb
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    if lemmatized_verb in verbs:
                        verb_index = verbs.index(lemmatized_verb)
                    elif lemmatized_verb in correct_verbs:
                        verb_index = correct_verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'is '+passive_verb+' by'
                    foil_passive.append(passivized_verb)
                elif 'VBP' in t[1]: # plural verb
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    if lemmatized_verb in verbs:
                        verb_index = verbs.index(lemmatized_verb)
                    elif lemmatized_verb in correct_verbs:
                        verb_index = correct_verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'are '+passive_verb+' by'
                    foil_passive.append(passivized_verb)
                else:
                    foil_passive.append(t[0])

            #create true passive from false active (e.g. "the car drives a woman" --> "the car is driven by a woman")
            foil_active = value['foil_active']
            tagged_foil_active = nltk.pos_tag(nltk.word_tokenize(foil_active)) # tag caption
            true_passive = []
            for t in tagged_foil_active:
                if 'VBZ' in t[1]: # singular verb
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    if lemmatized_verb in verbs:
                        verb_index = verbs.index(lemmatized_verb)
                    elif lemmatized_verb in correct_verbs:
                        verb_index = correct_verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'is '+passive_verb+' by'
                    true_passive.append(passivized_verb)
                elif 'VBP' in t[1]: # plural verb
                    doc = nlp(t[0])
                    lemmatized_verb = [token.lemma_ for token in doc][0]
                    if lemmatized_verb in verbs:
                        verb_index = verbs.index(lemmatized_verb)
                    elif lemmatized_verb in correct_verbs:
                        verb_index = correct_verbs.index(lemmatized_verb)
                    passive_verb = passive_verbs[verb_index]
                    passivized_verb = 'are '+passive_verb+' by'
                    true_passive.append(passivized_verb)
                else:
                    true_passive.append(t[0])
            transitive_valse[key] = value
            transitive_valse[key]['true_passive']=" ".join(true_passive)
            transitive_valse[key]['foil_passive']=" ".join(foil_passive)
    #print(transitive_valse)
    with open(os.path.join(VALSE_ROOT, 'transitive_actant_swap.json'), 'w') as s:
        json.dump(transitive_valse, s)


if __name__ == '__main__':
    #verb_list = []
    #build_transitive_valse(verb_list)
    convert_to_passive_valse()