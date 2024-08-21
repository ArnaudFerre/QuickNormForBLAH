
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################

import copy

from tensorflow import math, zeros, convert_to_tensor, int32, gather
import numpy

from converters import spacy_tags_from_onto_to_textual_tags_list

######################################################################################################################
# Method
######################################################################################################################

def is_desc(dd_ref, cui, cuiParent):
    """
    Description: A function to get if a concept is a descendant of another concept.
    Here, only used to select a clean subpart of an existing ontology (see select_subpart_hierarchy method).
    """
    result = False
    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:
            if cuiParent in dd_ref[cui]["parents"]:  # Not working if infinite is_a loop (normally never the case!)
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result:
                        break
    return result



def select_subpart_hierarchy(dd_ref, newRootCui):
    """
    Description: By picking a single concept in an ontology, create a new sub ontology with this concept as root.
    Here, only used to select the habitat subpart of the Ontobiotope ontology.
    """
    dd_subpart = dict()
    dd_subpart[newRootCui] = copy.deepcopy(dd_ref[newRootCui])
    dd_subpart[newRootCui]["parents"] = []

    for cui in dd_ref.keys():
        if is_desc(dd_ref, cui, newRootCui):
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

    # Clear concept-parents which are not in the descendants of the new root:
    for cui in dd_subpart.keys():
        dd_subpart[cui]["parents"] = list()
        for parentCui in dd_ref[cui]["parents"]:
            if is_desc(dd_ref, parentCui, newRootCui) or parentCui == newRootCui:
                dd_subpart[cui]["parents"].append(parentCui)

    return dd_subpart


def copying_spacy_onto(d_spacyOnto):
    d_copiedOnto = dict()

    for cui in d_spacyOnto.keys():
        d_copiedOnto[cui] = nlp(d_spacyOnto[cui].text)
        d_copiedOnto[cui].user_data["parents"] = d_spacyOnto[cui].user_data["parents"]
        d_copiedOnto[cui].user_data["synonyms"] = list()
        for synonym in d_spacyOnto[cui].user_data["synonyms"]:
            d_copiedOnto[cui].user_data["synonyms"].append(nlp(synonym.text))

    return d_copiedOnto

#########################################################

def get_hf_bb4_nbMentions(dataset, l_type=[]):
    nbMentions = 0
    for doc in dataset:
        for mention in doc["text_bound_annotations"]:
            if mention["type"] in l_type:
                nbMentions += 1
    return nbMentions


# WARNING: BigBio format doesn't contain test of BB4 norm...
def get_hf_bigbio_bb4_nbMentions(dataset, l_type=[]):
    nbMentions = 0
    for doc in dataset:
        for mention in doc["entities"]:
            if mention["type"] in l_type:
                nbMentions += 1
    return nbMentions


def get_hf_bb4_nbAmbiguous(l_datasets, l_type=[]):
    d_surfaceForms = dict()
    for dataset in l_datasets:
        for doc in dataset:
            for i, mention in enumerate(doc["text_bound_annotations"]):
                if mention["type"] in l_type:
                    mentionSurfaceForm = " ".join(mention["text"])

                    if mentionSurfaceForm not in d_surfaceForms.keys():
                        d_surfaceForms[mentionSurfaceForm] = dict()
                        d_surfaceForms[mentionSurfaceForm]["conceptSet"] = set()
                        for annotation in doc["normalizations"]:
                            if annotation["ref_id"] == mention["id"]:
                                d_surfaceForms[mentionSurfaceForm]["conceptSet"].add(annotation["cuid"])
                        d_surfaceForms[mentionSurfaceForm]["ambiguous"] = 0

                    if len(doc["normalizations"]) > 0:
                        for annotation in doc["normalizations"]:
                            if annotation["ref_id"] == mention["id"]:
                                if annotation["cuid"] not in d_surfaceForms[mentionSurfaceForm]["conceptSet"]:
                                    d_surfaceForms[mentionSurfaceForm]["ambiguous"] = True

    nbAmbiguous = 0
    for dataset in l_datasets:
        for doc in dataset:
            for i, mention in enumerate(doc["text_bound_annotations"]):
                if mention["type"] in l_type:
                    mentionSurfaceForm = " ".join(mention["text"])
                    if d_surfaceForms[mentionSurfaceForm]["ambiguous"] == True:
                        nbAmbiguous += 1

    return nbAmbiguous


def get_hf_bb4_nbMultiNorm(l_datasets, l_type=[]):
    nbMultiNorm = 0
    for dataset in l_datasets:
        for doc in dataset:
            for i, mention in enumerate(doc["text_bound_annotations"]):
                if mention["type"] in l_type:
                    l_concepts = list()
                    for annotation in doc["normalizations"]:
                        if annotation["ref_id"] == mention["id"]:
                            l_concepts.append(annotation["ref_id"])
                    if len(l_concepts) > 1:
                        nbMultiNorm += 1
    return nbMultiNorm


def get_list_of_tag_vectors(d_spacyOnto, TFhubPreprocessModel, TFhubModel, spanKey="synonyms", lower=True, mode="sequence_output"):

    dim = TFhubModel.variables[0].shape[1]

    # Preprocess the input mentions:
    l_lowercasedTags = spacy_tags_from_onto_to_textual_tags_list(d_spacyOnto, spanKey="synonyms", lower=lower)
    inputs = TFhubPreprocessModel(l_lowercasedTags)
    del l_lowercasedTags

    # Compute the mention representations at the output of BERT:
    outputs = TFhubModel(inputs)

    k = 0  # because there are labels more tags (it moves only with labels)
    dd_conceptVectors = dict()
    for cui in d_spacyOnto.keys():
        dd_conceptVectors[cui] = dict()
    for cui in d_spacyOnto.keys():
        if lower == True:
            label = d_spacyOnto[cui].text.lower()
        else:
            label = d_spacyOnto[cui].text
        if mode == "sequence_output":
            dd_conceptVectors[cui][label] = get_vector_from_spacy_mention(k, inputs, outputs, dim)
        elif mode == "pooled_output":
            dd_conceptVectors[cui][label] = get_vector_from_spacy_mention_pooled(k, outputs)
        k += 1  # Add +1 for the current label

        for synonym in d_spacyOnto[cui].user_data[spanKey]:
            if lower == True:
                synonymText = synonym.text.lower()
            else:
                synonymText = synonym.text
            if mode == "sequence_output":
                dd_conceptVectors[cui][synonymText] = get_vector_from_spacy_mention(k, inputs, outputs, dim)
            elif mode == "pooled_output":
                dd_conceptVectors[cui][synonymText] = get_vector_from_spacy_mention_pooled(k, outputs)
            k += 1  # Add +1 for the current synonym

    return dd_conceptVectors, k

######

def get_vector_from_spacy_mention(mentionIndex, associatedInputs, associatedOutputs, dim):
    """
    Description: Optimized version of these calculation (see get_vector_from_spacy_mention_old() for comparison).
    """
    last_index = numpy.argmax(associatedInputs["input_mask"][mentionIndex] <= 0)
    mention_mask = associatedInputs["input_mask"][mentionIndex]  # e.g. [1 1 1 1 0 ... 0], shape=(128,), dtype=int32)
    sequence_output = associatedOutputs["sequence_output"][mentionIndex]
    valid_indices = numpy.where(numpy.logical_and(mention_mask > 0, numpy.logical_and(0 < numpy.arange(len(mention_mask)), numpy.arange(len(mention_mask)) < last_index - 1)))[0]
    # e.g. [1 2]
    valid_indices_tensor = convert_to_tensor(valid_indices, dtype=int32)
    mentionVector = math.reduce_sum(gather(sequence_output, valid_indices_tensor), axis=0)
    mentionVector = math.scalar_mul(1 / numpy.linalg.norm(mentionVector), mentionVector)

    return mentionVector


def get_vector_from_spacy_mention_old(mentionIndex, associatedInputs, associatedOutputs, dim):
    """
    Description: deprecated. Vectors of mentions are now calculated internally by the language model.
    """
    mentionVector = zeros([dim])  # Initialization of a vector with the size of the output vector of BERT

    last_index = 0
    for j, tok in enumerate(associatedInputs["input_mask"][mentionIndex]):  # "input_mask" gives info on padding (tok=0 means it's a padding token)
        if tok > 0:  # if the current token is not null
            pass
        else:
            last_index = j  # now, it's just padding token
            break
    for j, tok in enumerate(associatedInputs["input_mask"][mentionIndex]):
        if tok > 0 and (j == 0 or j == last_index - 1):  # Does not take into account CLS and SEP
            pass
        elif tok > 0 and (0 < j < last_index - 1):  # only no-CLS, no-SEP, no-padding tokens
            mentionVector = math.add(mentionVector, associatedOutputs["sequence_output"][mentionIndex][j])  # Sum all token of the mention
        else:
            break

    # unit-normalization vector:
    inverse_norm = 1 / numpy.linalg.norm(mentionVector)
    mentionVector = math.scalar_mul(inverse_norm, mentionVector)

    return mentionVector

######

def get_vectors(l_labels, TFhubPreprocessModel, TFhubModel, mode="sequence_output"):

    dim = TFhubModel.variables[0].shape[1]

    # Preprocess the input mentions:
    inputs = TFhubPreprocessModel(l_labels)
    # Compute the mention representations at the output of BERT:
    outputs = TFhubModel(inputs)

    d_label2vec = dict()
    if mode == "sequence_output":
        for k, label in enumerate(l_labels):
            if label not in d_label2vec.keys():
                d_label2vec[label] = get_vector_from_spacy_mention(k, inputs, outputs, dim)
    elif mode == "pooled_output":
        for k, label in enumerate(l_labels):
            if label not in d_label2vec.keys():
                d_label2vec[label] = get_vector_from_spacy_mention_pooled(k, outputs)

    l_vectors = list()
    for label in l_labels:
        l_vectors.append(d_label2vec[label])
            
    return l_vectors


def get_vectors_old(labels, TFhubPreprocessModel, TFhubModel, mode="sequence_output"):
    # print("Create Label vectors...")

    dim = TFhubModel.variables[0].shape[1]

    # Preprocess the input mentions:
    inputs = TFhubPreprocessModel(labels)

    # Compute the mention representations at the output of BERT:
    outputs = TFhubModel(inputs)

    l_vectors = []
    for k, label in enumerate(labels):
        if mode == "sequence_output":
            l_vectors.append(get_vector_from_spacy_mention(k, inputs, outputs, dim))
        elif mode == "pooled_output":
            l_vectors.append(get_vector_from_spacy_mention_pooled(k, outputs))

    return l_vectors

######

def get_vectors_as_dict(onto, TFhubPreprocessModel, TFhubModel, mode="sequence_output"):
    
    print("Create Label vectors...")

    dim = TFhubModel.variables[0].shape[1]
         
    labels = []
    for cui in onto:
        for tag in onto[cui]:
            labels.append(tag)
    print(str(len(labels))+" labels")

    # Preprocess the input mentions:
    inputs = TFhubPreprocessModel(labels)

    # Compute the mention representations at the output of BERT:
    outputs = TFhubModel(inputs)
    
    vectors = dict()
    k = 0
    print("\nTEST")
    for cui in onto:
        vectors[cui] = dict()
        for tag in onto[cui]:
            if mode == "sequence_output":
                vectors[cui][tag] = get_vector_from_spacy_mention(k, inputs, outputs, dim)
            elif mode == "pooled_output":
                vectors[cui][tag] = get_vector_from_spacy_mention_pooled(k, outputs)

            k += 1
            
    return vectors, k
