
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project


######################################################################################################################
# Import
######################################################################################################################

import sys
import os


import spacy
from spacy.tokens import Doc

from pronto import Ontology  # 2.5.0 ok?

from nltk.stem import WordNetLemmatizer, PorterStemmer

import tensorflow_hub as hub
import tensorflow_text as text  # same version as tensorflow
from tensorflow import math, shape, zeros, string, data, constant, not_equal, cast, reduce_sum, expand_dims, float32, where, squeeze, tensor_scatter_nd_update, scatter_nd, concat, int32, tile
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend, callbacks, activations
import numpy
from scipy.spatial.distance import cosine, euclidean, cdist
import copy

import matplotlib.pyplot as plt

import glob
import json
import re

###################################################
# Ontological loaders:
###################################################
def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath)
    for o_concept in onto:
        dd_obt[o_concept.id] = dict()
        dd_obt[o_concept.id]["label"] = o_concept.name
        dd_obt[o_concept.id]["tags"] = list()

        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.desc)

        dd_obt[o_concept.id]["parents"] = list()
        for o_parent in o_concept.parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt



def loader_medic(filePath):

    dd_medic = dict()

    with open(filePath, encoding="utf8") as file:

        requestMESH = re.compile('MESH:(.+)$')
        requestOMIM = re.compile('OMIM:(.+)$')

        for line in file:

            if line[0] != '#': #commentary lines

                l_line = line.split('\t')

                cui = l_line[1]
                mMESH = requestMESH.match(cui)
                if mMESH:
                    shortCui = mMESH.group(1)
                else:  # OMIM
                    shortCui = cui

                dd_medic[shortCui]=dict()

                dd_medic[shortCui]["label"] = l_line[0]
                dd_medic[shortCui]["parents"] = list()

                if len(l_line[2]) > 0:
                    dd_medic[shortCui]["alt_cui"] = list()
                    l_altCuis = l_line[2].split('|')
                    for altCui in l_altCuis:
                        mMESH = requestMESH.match(altCui)
                        if mMESH:
                            shortAltCui = mMESH.group(1)
                            dd_medic[shortCui]["alt_cui"].append(shortAltCui)
                        else: #OMIM
                            dd_medic[shortCui]["alt_cui"].append(altCui)

                if len(l_line[7]) > 1:  # no tag
                    dd_medic[shortCui]["tags"] = l_line[7].rstrip().split('|')
                else:
                    dd_medic[shortCui]["tags"] = []

                if len(l_line[4]) > 0:
                    l_parents = l_line[4].split('|')
                    for parentCui in l_parents:
                        mMESH = requestMESH.match(parentCui)
                        if mMESH:
                            shortParentCui = mMESH.group(1)
                            dd_medic[shortCui]["parents"].append(shortParentCui)
                        else: #OMIM
                            dd_medic[shortCui]["alt_cui"].append(parentCui)


    return dd_medic


def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath)
    for concept_id in onto:
        dd_obt[concept_id] = dict()
        dd_obt[concept_id]["label"] = onto[concept_id].name
        dd_obt[concept_id]["tags"] = list()

        for tag in onto[concept_id].synonyms:
            dd_obt[concept_id]["tags"].append(tag.description)

        dd_obt[concept_id]["parents"] = list()
        for parent in onto[concept_id].superclasses(distance=1, with_self=False):
            dd_obt[concept_id]["parents"].append(parent.id)

    return dd_obt



###################################################
# PubAnnotation writers:
###################################################
def getListOfDocFromBatch(corpusName="", batchName=""):
    l_docUrl = list()

    end = False
    i = 0
    while end is not True:
        i += 1

        if i == 1:
            r = requests.get('https://pubannotation.org/projects/' + batchName + '/docs/sourcedb/' + corpusName + '.json')
        else:
            r = requests.get('https://pubannotation.org/projects/' + batchName + '/docs/sourcedb/' + corpusName + '.json?page=' + str(i))

        if len(r.json()) > 0:
            for j in range(len(r.json())):
                l_docUrl.append(r.json()[j]["sourceid"])
        else:
            end = True

    return l_docUrl

######

def getJSON_annotationFromfile(corpusName="", batchName="", docId=None):
    url = "https://pubannotation.org/projects/"+batchName+"/docs/sourcedb/"+corpusName+"/sourceid/" + docId + "/annotations.json"
    return requests.get(url).json()

def writeJSON_doc(path="", corpusName="", batchName="", docId=None):
    jsonContent = getJSON_annotationFromfile(corpusName=corpusName, batchName=batchName, docId=docId)
    with open(path+docId+"_annotations.json", 'w') as f:
        json.dump(jsonContent, f)

######

def writeBatchOfJSON_doc(path="", corpusName="", batchName="", l_docId=None):
    for docId in l_docId:
        writeJSON_doc(path=path, corpusName=corpusName, batchName=batchName, docId=docId)

###################################################
# PubAnnotation loaders:
###################################################

def load_pubannotation_corpus(folder):
    
    corpus = list()
    
    files = glob.glob(folder+"/*.json")
    for file in files:
        f = open(file)
        doc = json.load(f)
        f.close()
        corpus.append(doc)
        
    return corpus


def pubannotation_to_spacy_corpus(folder, l_type=None, spacyNlp=None):
    
    puba_corpus = load_pubannotation_corpus(folder)
    spacy_corpus = list()
    
    spacy.tokens.Span.set_extension("kb_id_", default={}, force=True)
    spacy.tokens.Span.set_extension("kb_name_", default="", force=True)
    spacy.tokens.Span.set_extension("pred_kb_id_", default={}, force=True)
    
    for doc in puba_corpus:
        
        # Due to a format problem in JSON file...
        spacy_doc = spacyNlp(doc['text'].replace("\n\n", " "))  # Problem in creating the JSON...

        spacy_doc.user_data["document_id"] = doc["sourceid"]
        
        l_mentions = list()
        disc_mentions = dict()
        for i, mention in enumerate(doc["denotations"]):
            
            match = re.search("^(.+?)-(\d)$", mention["id"])
            if (match is not None):
                mention_id = match.group(1)
                index = match.group(2)
                mention_type = mention["obj"]
                if (not mention_id in disc_mentions):
                    disc_mentions[mention_id] = dict()
                    disc_mentions[mention_id]['end'] = mention["span"]["end"]
                elif (disc_mentions[mention_id]['end'] < mention["span"]["end"]):
                    disc_mentions[mention_id]['end'] = mention["span"]["end"]
                if (mention_type != "_FRAGMENT"):
                    disc_mentions[mention_id]['type'] = mention_type
                if (index == "0"):
                    disc_mentions[mention_id]['start'] = mention["span"]["begin"]
            
            else:
                if (mention["obj"] in l_type or l_type is None):
                    start = int(mention["span"]["begin"])
                    end = int(mention["span"]["end"])

                    mentionSpan = spacy_doc.char_span(start, end, label=mention['obj'], alignment_mode="expand", span_id=mention['id'])

                    if mentionSpan is None:
                        print(spacy_doc.user_data["document_id"], mention['id'], start, end, spacy_doc.text[start:end], len(spacy_doc.text))

                    # Complete span info:
                    mentionSpan.id_ = mention["id"]
                    mentionSpan.label_ = mention["obj"]
                    mentionSpan._.kb_id_ = set()
                    mentionSpan._.pred_kb_id_ = set()  # initialization of this attribute for future prediction
                    l_mentions.append(mentionSpan)
        
        for disc in disc_mentions:
            if (disc_mentions[disc]["type"] in l_type or l_type is None):
                start = disc_mentions[disc]["start"]
                end = disc_mentions[disc]["end"]

                mentionSpan = spacy_doc.char_span(start, end, label=disc_mentions[disc]['type'], alignment_mode="expand", span_id=disc)

                # Complete span info:
                mentionSpan.id_ = disc
                mentionSpan.label_ = disc_mentions[disc]['type']
                mentionSpan._.kb_id_ = set()
                mentionSpan._.pred_kb_id_ = set()  # initialization of this attribute for future prediction
                l_mentions.append(mentionSpan)
                          
        spacy_doc.spans["mentions"] = l_mentions
        
        # attributes (normalization)
        if ("attributes" in doc):
            for att in doc["attributes"]:
                mention_id = att['subj']
                match = re.search("^(.+?)-(\d)$", mention_id)
                if (match is not None):
                    mention_id = match.group(1)
         
                for mention in spacy_doc.spans["mentions"]:
                    if (mention.id_ == mention_id):
                        mention._.kb_id_.add(att['obj'])
                        mention._.kb_name_ = att['pred']
                
        spacy_corpus.append(spacy_doc)
        
    return spacy_corpus
    


######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    """
    # Load url of doc:
    BB4_name = "BB-norm@ldeleger"
    l_trainDoc = getListOfDocFromBatch(corpusName=BB4_name, batchName="bionlp-ost-19-BB-norm-train")
    l_devDoc = getListOfDocFromBatch(corpusName=BB4_name, batchName="bionlp-ost-19-BB-norm-dev")
    l_testDoc = getListOfDocFromBatch(corpusName=BB4_name, batchName="bionlp-ost-19-BB-norm-test")
    print("Number of doc in train, dev, test:", len(l_trainDoc), len(l_devDoc), len(l_testDoc))

    # Write the batches (one file per doc):
    trainBatchName = "bionlp-ost-19-BB-norm-train"
    writeBatchOfJSON_doc(path="./datasets/BB4/"+trainBatchName+"/", corpusName=BB4_name, batchName=trainBatchName, l_docId=l_trainDoc)

    devBatchName = "bionlp-ost-19-BB-norm-dev"
    writeBatchOfJSON_doc(path="./datasets/BB4/" + devBatchName + "/", corpusName=BB4_name, batchName=devBatchName, l_docId=l_devDoc)

    testBatchName = "bionlp-ost-19-BB-norm-test"
    writeBatchOfJSON_doc(path="./datasets/BB4/" + testBatchName + "/", corpusName=BB4_name, batchName=testBatchName, l_docId=l_testDoc)
    """