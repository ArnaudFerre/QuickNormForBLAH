
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################

import spacy
from spacy.tokens import Doc

######################################################################################################################
# Method
######################################################################################################################

def hf_bb4_into_spacy(l_datasets):
    l_corpus = list()
    for dataset in l_datasets:
        for doc in dataset:
            spacy_doc = nlp(doc['text'])
            spacy_doc.user_data["document_id"] = doc["document_id"]
            l_corpus.append(spacy_doc)
    return l_corpus


def hf_bb4_into_spacy_corpus(l_datasets, l_type=None, spacyNlp=None):
    l_corpus = list()

    spacy.tokens.Span.set_extension("kb_id_", default={}, force=True)
    spacy.tokens.Span.set_extension("pred_kb_id_", default={}, force=True)

    for dataset in l_datasets:
        for doc in dataset:

            spacy_doc = spacyNlp(doc['text'])
            spacy_doc.user_data["document_id"] = doc["document_id"]
            l_corpus.append(spacy_doc)

            l_mentions = list()
            for i, mention in enumerate(doc["text_bound_annotations"]):
                if mention["type"] in l_type or l_type is None:  # if l_type is None, all entities are considered

                    start = mention["offsets"][0][0]
                    end = mention["offsets"][-1][-1]

                    mentionSpan = spacy_doc.char_span(start, end, label=mention['type'], alignment_mode="expand", span_id=mention['id'])

                    # Complete span info:
                    mentionSpan.id_ = mention["id"]
                    mentionSpan.label_ = mention["type"]
                    mentionSpan._.kb_id_ = set()
                    mentionSpan._.pred_kb_id_ = set()  # initialization of this attribute for future prediction
                    if len(doc["normalizations"]) > 0:
                        for annotation in doc["normalizations"]:
                            if annotation["ref_id"] == mention["id"]:
                                cuidBigbio = annotation["cuid"]  # sans ':' après "OBT"...
                                cuid = cuidBigbio[:3] + ":" + cuidBigbio[3:]
                                mentionSpan._.kb_id_.add(cuid)

                    l_mentions.append(mentionSpan)

            # Create recognized entities:
            spacy_doc.spans["mentions"] = l_mentions  # https://spacy.io/api/spangroup
            #spacy_doc.set_ents(l_mentions)  # No, can't consider overlapping mentions...

    return l_corpus


def create_onto_from_ontobiotope_dict(dd_obt):
    l_spacyOBT = list()
    for cui in dd_obt.keys():
        concept = nlp(dd_obt[cui]["label"])
        concept.user_data["cui"] = cui
        concept.user_data["parents"] = dd_obt[cui]["parents"]
        concept.user_data["synonyms"] = list()
        for synonym in dd_obt[cui]["tags"]:
            concept.user_data["synonyms"].append(nlp(synonym))
        l_spacyOBT.append(concept)
    return l_spacyOBT



def create_onto_from_ontobiotope_dict_v2(dd_obt, spacyNlp):
    d_spacyOBT = dict()
    for cui in dd_obt.keys():
        d_spacyOBT[cui] = spacyNlp(dd_obt[cui]["label"])
        d_spacyOBT[cui].user_data["parents"] = dd_obt[cui]["parents"]
        d_spacyOBT[cui].user_data["synonyms"] = list()
        for synonym in dd_obt[cui]["tags"]:
            d_spacyOBT[cui].user_data["synonyms"].append(spacyNlp(synonym))
    return d_spacyOBT


def create_spacy_onto_from_ontobiotope_obo_file():
    #ToDo
    return None


def spacy_tags_from_onto_to_textual_tags_list(d_spacyOnto, spanKey="synonyms", lower=True):
    l_tags = list()
    for cui in d_spacyOnto.keys():
        if lower == True:
            l_tags.append(d_spacyOnto[cui].text.lower())
        else:
            l_tags.append(d_spacyOnto[cui].text)

        for synonym in d_spacyOnto[cui].user_data[spanKey]:
            if lower == True:
                l_tags.append(synonym.text.lower())
            else:
                l_tags.append(synonym.text)
    return l_tags

def spacy_onto_to_dict(d_spacyOnto,spanKey="synonyms", lower=True):
    
    # spacy onto to simple dictionary
    d_onto = dict()
    for cui in d_spacyOnto.keys():
        label = ""
        if lower == True:
            label = d_spacyOnto[cui].text.lower()
        else:
            label = d_spacyOnto[cui].text
        d_onto[cui] = [label] 
        for synonym in d_spacyOnto[cui].user_data[spanKey]:
            synonymText = ""
            if lower == True:
                synonymText = synonym.text.lower()
            else:
                synonymText = synonym.text
            d_onto[cui].append(synonymText)
            
    return d_onto


def ncbi_onto_pubannotation():
    pass
