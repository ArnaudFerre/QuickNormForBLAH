
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project


######################################################################################################################
# Import
######################################################################################################################

import sys
import os

from datasets import load_dataset_builder, load_dataset, get_dataset_config_names

import spacy
from spacy.tokens import Doc

from pronto import Ontology

from nltk.stem import WordNetLemmatizer, PorterStemmer

import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import math, shape, zeros, string, data, constant, not_equal, cast, reduce_sum, expand_dims, float32, where, squeeze, tensor_scatter_nd_update, scatter_nd, concat, int32, tile
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend, callbacks, activations
import numpy
from scipy.spatial.distance import cosine, euclidean, cdist
import copy

import matplotlib.pyplot as plt


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

                dd_medic[shortCui]["tags"] = l_line[7].rstrip().split('|')

                if len(l_line[4]) > 0:
                    l_parents = l_line[4].split('|')
                    dd_medic[shortCui]["parents"] = list()
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