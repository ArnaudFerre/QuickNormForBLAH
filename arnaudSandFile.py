# Author: Louise Deléger & Arnaud Ferré
# Description: QuickNorm method developed during BLAH and after.
# Work fil for the main to optimize the code.

######################################################################################################################
# Import
######################################################################################################################
print("Importing dependencies...")

print("Importing TF...")
from tensorflow import string, config
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, \
    backend, callbacks, activations
print("TF imported")

print("Importing tensorflow_text...")
import tensorflow_text as text  # mandatory to avoid error (https://stackoverflow.com/questions/75576980/tensorflow-2-x-error-op-type-not-registered-casefoldutf8-in-binary-running-o)
print("tensorflow_text imported.")

import tensorflow_hub as hub

print("\n\nGPU TF info...")
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print("get_available_devices():", get_available_devices())
print("device_lib.list_local_devices():", device_lib.list_local_devices())
print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))


import os
import glob
from os import listdir
import random
import sys
import pathlib
import time
import argparse

import numpy
from scipy.spatial.distance import cdist

from loaders import loader_ontobiotope, pubannotation_to_python_corpus
from utils import select_subpart_hierarchy, get_vectors_as_dict, get_vectors
from evaluation import accuracy

######################################################################################################################
# Method
######################################################################################################################

def get_label_from_cui(cui, dd_onto, verbose=0):
    for cui_b in dd_onto.keys():
        if cui == cui_b:
            return dd_onto[cui]["label"]
    if verbose == 1:
        print("ERROR: CUI not in ontology...", "(CUI="+cui+")")

def get_labels_from_mention(d_mention, dd_onto, gold=True):
    if gold == True:
        for cui in list(d_mention["cui"]):  # only the first prediction
            return get_label_from_cui(cui, dd_onto), cui
    else:
        for cui in list(d_mention["pred_cui"]):  # only one prediction
            return get_label_from_cui(cui, dd_onto), cui

def print_mention_info(d_mention, dd_onto, tabs=""):
    print(tabs+"MENTION:", d_mention["surface"])
    print(tabs+"pred:", str(get_labels_from_mention(d_mention, dd_onto, gold=False)))
    print(tabs+"gold:", str(get_labels_from_mention(d_mention, dd_onto, gold=True)))


def get_nearest_concept_in_batch(ddd_docWithMentions, predMatrix, dd_labelsVectors, dd_predictions,
                                 maxScores):

    dim = predMatrix.shape[1]
    numberOfTags = sum(len(v) if isinstance(v, dict) else 1 for v in dd_labelsVectors.values())
    labtagsVectorMatrix = numpy.zeros((numberOfTags, dim))

    i = 0
    for cui in dd_labelsVectors.keys():
        for labtag in dd_labelsVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_labelsVectors[cui][labtag]  # 6846:0.34 6847:0.35 6848:0.29 6849:0.3 6850:0.17 6851:0.19 6852:0.18 6853:nan 6854:nan 6855:nan 6856:nan 6857:nan 6858:nan 6859:nan 6860:nan 6861:nan 6862:nan 6863:nan 6864:nan 6865:nan 6866:nan 6867:nan 6868:nan
            i += 1

    scoreMatrix = cdist(predMatrix, labtagsVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    # Just to obtain the true cosine value:
    i = 0
    for doc_id in ddd_docWithMentions.keys():
        for mention_id in ddd_docWithMentions[doc_id].keys():
            j = 0
            for cui in dd_labelsVectors.keys():
                for labtag in dd_labelsVectors[cui].keys():
                    scoreMatrix[i][j] = 1 - scoreMatrix[i][j]
                    j += 1
            i += 1
    print("\t\tDone.")

    # update maxScores and predictions
    for i in range(len(dd_predictions)):
        j_max_score = numpy.argmax(scoreMatrix[i])
        if (scoreMatrix[i][j_max_score] > maxScores[i]):
            j = -1
            stopSearch = False
            for cui in dd_labelsVectors.keys():
                if stopSearch == True:
                    break
                for labtag in dd_labelsVectors[cui].keys():
                    j += 1
                    if (j == j_max_score):
                        dd_predictions[i] = cui
                        maxScores[i] = scoreMatrix[i][j_max_score]
                        stopSearch = True
                        break

    return dd_predictions, maxScores


def get_nearest_concept(ddd_docWithMentions, numberOfMentions, predMatrix, dd_onto, TFhubPreprocessModel, TFhubModel, mode="sequence_output"):
    batch_size = 1024  # as BioSyn does

    maxScores = numpy.zeros(numberOfMentions)
    dd_predictions = []
    for i in range(numberOfMentions):
        dd_predictions.append("")

    current_size = 0
    dd_concepts = dict()

    for cui in dd_onto:
        if (current_size == batch_size):
            # create concept embeddings for the current batch
            dd_conceptVectors, numberOfTags = get_vectors_as_dict(dd_concepts, TFhubPreprocessModel, TFhubModel, mode=mode)
            # compute nearest neighbors for the current batch and keep track of max scores
            dd_predictions, maxScores = get_nearest_concept_in_batch(ddd_docWithMentions, predMatrix,
                                                                     dd_conceptVectors, dd_predictions, maxScores)
            # reinitialize
            current_size = 0
            dd_concepts = dict()

        # store current concept
        l_tags = list()
        l_tags.append(dd_onto[cui]["label"].lower())
        for tag in dd_onto[cui]["tags"]:
            l_tags.append(tag.lower())
        dd_concepts[cui] = l_tags
        current_size += 1

    # process the remainder
    if (dd_concepts):
        dd_conceptVectors, numberOfTags = get_vectors_as_dict(dd_concepts, TFhubPreprocessModel, TFhubModel, mode=mode)
        dd_predictions, maxScores = get_nearest_concept_in_batch(ddd_docWithMentions, predMatrix,
                                                                 dd_conceptVectors, dd_predictions, maxScores)

    i = 0
    nbUnknown = 0
    for doc_id in ddd_docWithMentions.keys():

        print("\n", doc_id)

        for mention_id in list(ddd_docWithMentions[doc_id].keys()):
            if len(ddd_docWithMentions[doc_id][mention_id]["pred_cui"]) == 0:
                ddd_docWithMentions[doc_id][mention_id]["pred_cui"].add(dd_predictions[i])
                #print_mention_info(mention, dd_onto, tabs=" ")

            i += 1
    print("Failed predictions:", nbUnknown)


    return ddd_docWithMentions


def write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath):
    # Add some random examples to complete the last batch:
    # numberInBatch has stopped with last example
    if numberInBatch < batchSize - 1:  # If the last batch is not perfectly full
        print("Completing batch " + str(batchNb))
        for i in range(numberInBatch + 1, batchSize):
            randomBatchNumber = random.choice(list(range(batchNb - 1)))
            X_filename = "batch" + str(randomBatchNumber) + "_X.npy"
            random_X = numpy.loadtxt(batchFilePath + X_filename, delimiter="\t", dtype="str")
            randomExampleNumber = random.choice(list(range(batchSize - 1)))
            l_mentions.append(str(random_X[randomExampleNumber]))
            Y_filename = "batch" + str(randomBatchNumber) + "_Y.npy"
            random_Y = numpy.load(batchFilePath + Y_filename)
            concept_vectors.append(random_Y[randomExampleNumber])

    #print("Writing batch " + str(batchNb))
    filename = "batch" + str(batchNb) + "_X.npy"
    file_object = open(batchFilePath + filename, "wb")
    numpy.savetxt(file_object, l_mentions, fmt="%s", encoding='utf8')
    file_object.close()
    filename = "batch" + str(batchNb) + "_Y.npy"
    file_object = open(batchFilePath + filename, "wb")
    numpy.save(file_object, numpy.array(concept_vectors))
    file_object.close()


def prepare_training_data_in_batches(ddd_train_corpus, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=True, addMentions=True, mode="pooled_output"):

    # create folder if does not exist
    pathlib.Path(batchFilePath).mkdir(parents=True, exist_ok=True)

    # empty batch folder
    files = glob.glob(batchFilePath + '/batch*')
    for f in files:
        os.remove(f)

    # build batches
    batchNb = 0
    nbMentions = 0

    print(time.strftime("%H:%M:%S", time.localtime()), "(beginning for mentions...)")
    if addMentions == True:
        l_mentions = []
        l_concepts = []
        count = 0
        numberInBatch = 0
        for doc_id in ddd_train_corpus.keys():
            nbMentions += len(ddd_train_corpus[doc_id].keys())
            for mention_id in ddd_train_corpus[doc_id].keys():
                l_tags = list()
                #print(doc_id, mention_id, ddd_train_corpus[doc_id][mention_id])
                for cui in ddd_train_corpus[doc_id][mention_id]["cui"]:
                    l_tags.append(dd_onto[cui]["label"].lower())
                    for tag in dd_onto[cui]["tags"]:
                        l_tags.append(tag.lower())
                for tag in l_tags:
                    count += 1
                    numberInBatch = count % batchSize
                    l_mentions.append(" ".join(ddd_train_corpus[doc_id][mention_id]["surface"].lower().split()))
                    l_concepts.append(tag)

                    if numberInBatch == (batchSize - 1):

                        # get_vectors is relatively expensive (ToDo: optimization?):
                        concept_vectors = get_vectors(l_concepts, preprocessor, bert_encoder, mode=mode)

                        # Writing is relatively fast:
                        write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)
                        # re-initialize:
                        batchNb += 1
                        l_mentions = []
                        l_concepts = []

        # Deal with the last batch (if there at least one another mention)
        if (len(l_mentions) > 0):
            concept_vectors = get_vectors(l_concepts, preprocessor, bert_encoder, mode=mode)
            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)

    if addLabels == True:
        print(time.strftime("%H:%M:%S", time.localtime()), "(beginning for labels...)")
        # Add labels/synonyms to training data: (aim: stabilizing the output vectors)
        # Number of examples before: 1945 1945
        print("Prepare labels for training...")
        # create vectors in batches
        l_mentions = []
        count = 0
        numberInBatch = 0
        for cui in dd_onto:
            l_tags = list()
            l_tags.append(dd_onto[cui]["label"].lower())
            for tag in dd_onto[cui]["tags"]:
                l_tags.append(tag.lower())
            for tag in l_tags:
                count += 1
                numberInBatch = count % batchSize
                l_mentions.append(tag)
                if numberInBatch == (batchSize - 1):
                    # get concept vectors
                    concept_vectors = get_vectors(l_mentions, preprocessor, bert_encoder, mode=mode)
                    # write batch
                    write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)
                    # increment and re-initialize
                    batchNb += 1  # batchNb is not reinitialized for the labels (so, the next print is for all mentions and labels).
                    l_mentions = []

        # Deal with the last batch
        if (len(l_mentions) > 0):
            concept_vectors = get_vectors(l_mentions, preprocessor, bert_encoder, mode=mode)
            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)

    if addLabels == False and addMentions == False:
        print("ERROR: need at least to train on labels or on traning mentions")
        sys.exit(0)


def prepare_training_data_in_batches_v2(ddd_train_corpus, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=True, addMentions=True, mode="pooled_output"):

    # Create one-hot vector for each concept:
    dd_labelsVectors = dict()
    for i, cui in enumerate(dd_onto.keys()):
        dd_labelsVectors[cui] = numpy.zeros(len(dd_onto.keys()))
        dd_labelsVectors[cui][i] = 1

    # create folder if does not exist
    pathlib.Path(batchFilePath).mkdir(parents=True, exist_ok=True)

    # empty batch folder
    files = glob.glob(batchFilePath + '/batch*')
    for f in files:
        os.remove(f)

    # build batches
    batchNb = 0
    nbMentions = 0

    print(time.strftime("%H:%M:%S", time.localtime()), "(beginning for mentions...)")
    if addMentions == True:
        l_mentions = []
        l_concepts = []
        count = 0
        numberInBatch = 0
        for doc_id in ddd_train_corpus.keys():
            nbMentions += len(ddd_train_corpus[doc_id].keys())
            for mention_id in ddd_train_corpus[doc_id].keys():

                l_mentions.append(" ".join(ddd_train_corpus[doc_id][mention_id]["surface"].lower().split()))

                first_cui = list(ddd_train_corpus[doc_id][mention_id]["cui"])[0]  # ToDo: include all gold CUIs?
                l_concepts.append(dd_labelsVectors[first_cui])

                count += 1
                numberInBatch = count % batchSize

                if numberInBatch == (batchSize - 1):

                    # Writing is relatively fast:
                    write_batch(l_mentions, l_concepts, numberInBatch, batchNb, batchSize, batchFilePath)

                    # re-initialize:
                    batchNb += 1
                    l_mentions = []
                    l_concepts = []

        # Deal with the last batch (if there at least one another mention)
        if (len(l_mentions) > 0):
            first_cui = list(ddd_train_corpus[doc_id][mention_id]["cui"])[0]
            l_concepts.append(dd_labelsVectors[first_cui])
            write_batch(l_mentions, l_concepts, numberInBatch, batchNb, batchSize, batchFilePath)

    if addLabels == True:
        print(time.strftime("%H:%M:%S", time.localtime()), "(beginning for labels...)")
        # Add labels/synonyms to training data: (aim: stabilizing the output vectors)
        # Number of examples before: 1945 1945
        print("Prepare labels for training...")
        # create vectors in batches
        l_mentions = []
        concept_vectors = []
        count = 0
        numberInBatch = 0
        for cui in dd_onto:
            l_tags = list()
            l_tags.append(dd_onto[cui]["label"].lower())
            for tag in dd_onto[cui]["tags"]:
                l_tags.append(tag.lower())
            for tag in l_tags:
                count += 1
                numberInBatch = count % batchSize
                l_mentions.append(tag)
                concept_vectors.append(dd_labelsVectors[cui])
                if numberInBatch == (batchSize - 1):
                    # write batch
                    write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)
                    # increment and re-initialize
                    batchNb += 1  # batchNb is not reinitialized for the labels (so, the next print is for all mentions and labels).
                    l_mentions = []
                    concept_vectors = []

        # Deal with the last batch
        if (len(l_mentions) > 0):
            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)

    if addLabels == False and addMentions == False:
        print("ERROR: need at least to train on labels or on traning mentions")
        sys.exit(0)



def generator1input(datasetPath, BatchNb):
    batchValue = 0
    while True:
        if batchValue < BatchNb:
            X_batch = numpy.loadtxt(datasetPath + "batch" + str(batchValue) + "_X.npy", dtype='str', delimiter="\t")
            Y_batch = numpy.load(datasetPath + "batch" + str(batchValue) + "_Y.npy")

            yield X_batch, Y_batch

            if batchValue < BatchNb - 1:
                batchValue += 1
            else:
                batchValue = 0
        else:
            print("ERROR: Batch size error...")
            sys.exit(0)


def submodel_with_batches(preprocessor, bert_encoder, trainingDataPath, epoch=10, patience=10, delta=0.0001, verbose=0,
                          shuffle=False):

    dim = bert_encoder.variables[0].shape[1]

    numberOfBatches = int(len(listdir(trainingDataPath)) / 2)
    my_generator = generator1input(trainingDataPath, numberOfBatches)

    text_input = Input(shape=(), dtype=string, name='inputs')
    encoder_inputs = preprocessor(text_input)
    bert_outputs = bert_encoder(encoder_inputs)

    pooled_output = bert_outputs["pooled_output"]  # [batch_size, 128].
    unitNormLayer = layers.UnitNormalization()(pooled_output)  # less good without...
    dense_layer = layers.Dense(dim, activation=None, kernel_initializer=initializers.Identity())(unitNormLayer)
    TFmodel = Model(inputs=text_input, outputs=dense_layer)
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=['cosine_similarity', 'logcosh'])  # losses.CosineSimilarity()
    TFmodel.summary() if verbose == True else None

    callback = callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=delta)
    history = TFmodel.fit(my_generator, epochs=epoch, callbacks=[callback], verbose=verbose, steps_per_epoch=numberOfBatches)

    return preprocessor, bert_encoder, TFmodel


def submodel_with_batches_v2(preprocessor, bert_encoder, trainingDataPath, epoch=10, patience=10, delta=0.0001, verbose=0,
                          shuffle=False, dim=None):
    """
    # Description: modification with classifcation output
    """

    #dim = bert_encoder.variables[0].shape[1]  # If output is the size of embeddings

    numberOfBatches = int(len(listdir(trainingDataPath)) / 2)
    my_generator = generator1input(trainingDataPath, numberOfBatches)

    text_input = Input(shape=(), dtype=string, name='inputs')
    encoder_inputs = preprocessor(text_input)
    bert_outputs = bert_encoder(encoder_inputs)

    pooled_output = bert_outputs["pooled_output"]  # [batch_size, 128].
    unitNormLayer = layers.UnitNormalization()(pooled_output)  # less good without...
    #dense_layer = layers.Dense(dim, activation=None, kernel_initializer=initializers.Identity())(unitNormLayer)
    classification_output = layers.Dense(dim, activation=activations.softmax)(unitNormLayer)

    TFmodel = Model(inputs=text_input, outputs=classification_output)
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.categorical_crossentropy, metrics=['accuracy'])  # losses.CosineSimilarity()
    TFmodel.summary() if verbose == True else None

    callback = callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=delta)
    history = TFmodel.fit(my_generator, epochs=epoch, callbacks=[callback], verbose=verbose, steps_per_epoch=numberOfBatches)

    return preprocessor, bert_encoder, TFmodel


def twoStep_finetuned_quicknorm_train(ddd_trainDoc, dd_onto, PREPROCESS_MODEL, TF_BERT_MODEL, batchFilePath,
                                      batchSize, verbose=0, mode="pooled_output"):
    ######
    # Loading TF Hub model
    ######
    preprocessor = hub.KerasLayer(PREPROCESS_MODEL, name="preprocessor")
    bert_encoder = hub.KerasLayer(TF_BERT_MODEL, trainable=True, name="bert_encoder")

    ######
    # Target finetuning:
    ######
    labelVectorMode = "sequence_output"  # "sequence_output" / "pooled_output" (stangely work less)

    print("Prepating training data...")
    # addLabels=True pour le vrai quicknorm
    prepare_training_data_in_batches(ddd_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder, addLabels=True, addMentions=True, mode=labelVectorMode)
    print(time.strftime("%H:%M:%S", time.localtime()), "training data prepared.")

    # For BB4, 10 epochs. But too long for NCBI-DC, so reduce to 4
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=10, patience=30, delta=0.0001, verbose=1, shuffle=False)
    # del TFmodel

    print("\n\n")

    weights = bert_encoder.get_weights()

    prepare_training_data_in_batches(ddd_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=False, addMentions=True, mode=labelVectorMode)
    # For BB4, 20 epochs. But too long for NCBI-DC, so reduce to 10
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=20,
                                                                patience=30, delta=0.0001, verbose=1, shuffle=False)

    return preprocessor, weights, bert_encoder, TFmodel


def twoStep_finetuned_quicknorm_train_v2(ddd_trainDoc, dd_onto, PREPROCESS_MODEL, TF_BERT_MODEL, batchFilePath,
                                      batchSize, verbose=0, mode="pooled_output"):
    ######
    # Loading TF Hub model
    ######
    preprocessor = hub.KerasLayer(PREPROCESS_MODEL, name="preprocessor")
    bert_encoder = hub.KerasLayer(TF_BERT_MODEL, trainable=True, name="bert_encoder")

    ######
    # Target finetuning:
    ######
    labelVectorMode = "sequence_output"  # "sequence_output" / "pooled_output" (stangely work less)

    print("Prepating training data...")
    weights = bert_encoder.get_weights()
    prepare_training_data_in_batches_v2(ddd_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder, addLabels=True, addMentions=True, mode=labelVectorMode)
    print(time.strftime("%H:%M:%S", time.localtime()), "training data prepared.")

    # For BB4, 10 epochs. But too long for NCBI-DC, so reduce to 4
    preprocessor, bert_encoder, TFmodel = submodel_with_batches_v2(preprocessor, bert_encoder, batchFilePath, epoch=50, patience=30, delta=0.0001, verbose=1, shuffle=False, dim=len(dd_onto.keys()))
    # del TFmodel

    return preprocessor, weights, bert_encoder, TFmodel


def twoStep_finetuned_quicknorm_predict(ddd_valDoc, dd_onto, preprocessor, bert_encoder, weights, TFmodel, verbose=0,
                                        mode="pooled_output"):
    ######
    # Regression prediction:
    ######
    labelVectorMode = "sequence_output"
    print("\tRegression prediction...")
    nbMentionsInPred = 0
    for doc_id in ddd_valDoc.keys():
        nbMentionsInPred += len(ddd_valDoc[doc_id].keys())
    print("\tNumber of mentions for prediction:", nbMentionsInPred)
    dim = bert_encoder.variables[0].shape[1]

    Y_pred = numpy.zeros((nbMentionsInPred, dim))  # Could be huge? Todo: prediction for a limited batch of mention?

    # Todo: optimize with predicting a full batch a the same time (not so helpful here because quick calculation with a dense layer)
    i = 0
    for doc_id in ddd_valDoc.keys():
        for mention_id in list(ddd_valDoc[doc_id].keys()):
            x_test = [ddd_valDoc[doc_id][mention_id]["surface"].lower()]
            Y_pred[i] = TFmodel.predict(x_test, verbose=0)[0]  # result of the regression for the i-th mention.

            i += 1
    print("\t\tDone.")

    ######
    # Nearest neighbours calculation for each mention and CUI attribution:
    ######
    bert_encoder.set_weights(weights)  # ToDo: kesako ?
    ddd_valDoc = get_nearest_concept(ddd_valDoc, nbMentionsInPred, Y_pred, dd_onto, preprocessor, bert_encoder, mode=labelVectorMode)

    return ddd_valDoc


# a similar method for the prediction as twoStep_finetuned_quicknorm_predict, but in classification mode
def twoStep_finetuned_quicknorm_predict_v2(ddd_valDoc, dd_onto, preprocessor, bert_encoder, weights, TFmodel, verbose=0,
                                        mode="pooled_output"):

    nbMentionsInPred = 0
    for doc_id in ddd_valDoc.keys():
        nbMentionsInPred += len(ddd_valDoc[doc_id].keys())
    print("\tNumber of mentions for prediction:", nbMentionsInPred)

    Y_pred = numpy.zeros( (nbMentionsInPred, len(dd_onto.keys())) )

    i = 0
    for doc_id in ddd_valDoc.keys():
        for mention_id in list(ddd_valDoc[doc_id].keys()):
            x_test = [ddd_valDoc[doc_id][mention_id]["surface"].lower()]
            Y_pred[i] = TFmodel.predict(x_test, verbose=0)[0]

            ddd_valDoc[doc_id][mention_id]["pred_cui"].add(list(dd_onto.keys())[numpy.argmax(Y_pred[i])])

            i += 1
    print("\t\tDone.")

    return ddd_valDoc


######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':

    def main(option):

        start = time.time()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)

        if option == 'bb4':

            print("\nQuickNorm on BB4:")

            bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
            bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
            bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
            bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"

            print("Loading OntoBiotope...")
            dd_obt = loader_ontobiotope("./datasets/BB4/OntoBiotope_BioNLP-OST-2019.obo")
            dd_obt_hab = select_subpart_hierarchy(dd_obt, "OBT:000001")
            print("OntoBiotope loaded with", len(dd_obt_hab.keys()), "habitat concepts.")

            print("Loading BB4 Habitats corpus...")
            ddd_BB4_hab_train = pubannotation_to_python_corpus(bb4_norm_train_folder, l_type=["Habitat"])
            ddd_BB4_hab_dev = pubannotation_to_python_corpus(bb4_norm_dev_folder, l_type=["Habitat"])
            print("\n\tNb of doc in train:", len(ddd_BB4_hab_train))
            print("\n\tNb of doc in dev:", len(ddd_BB4_hab_dev))
            print("\n\tNb of mentions in train:", sum(len(sub_dict.keys()) for sub_dict in ddd_BB4_hab_train.values()))
            print("\n\tNb of mentions in dev:", sum(len(sub_dict.keys()) for sub_dict in ddd_BB4_hab_dev.values()))
            print("Done.")

            print("\nTraining...")
            batchFilePath = "./tmp/"
            PREPROCESS_MODEL = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"  # 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'  #
            TF_BERT_model = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-2-h-128-a-2/2"  # 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'  #
            batchSize = 64  # 256  # 64
            preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(ddd_BB4_hab_train,
                                                                                             dd_obt_hab, PREPROCESS_MODEL,
                                                                                             TF_BERT_model, batchFilePath,
                                                                                             batchSize, verbose=1,
                                                                                             mode="pooled_output")
            print("training done.")

            print("\nPrediction...")
            ddd_normalized_BB4_hab_dev = twoStep_finetuned_quicknorm_predict(ddd_BB4_hab_dev, dd_obt_hab,
                                                                                         preprocessor, bert_encoder, weights,
                                                                                         TFmodel, verbose=0,
                                                                                         mode="pooled_output")
            print("Prediction done.")

            end = time.time()
            print(end - start, "sec de temps d'execution.")
            print("Accuracy:", accuracy(ddd_normalized_BB4_hab_dev))


        elif option == 'ncbi':

            print("\nQuickNorm on NCBI-DC:")

            ncbi_norm_test_folder = "datasets/NCBI-DC/pubannotation-test-clean/"
            ncbi_norm_dev_folder = "datasets/NCBI-DC/pubannotation-dev-clean/"
            ncbi_norm_train_folder = "datasets/NCBI-DC/pubannotation-train-clean/"

            print("Loading Medic ontology...")
            from loaders import loader_medic
            dd_medic = loader_medic("datasets/NCBI-DC/CTD_diseases_dnorm.tsv")
            print("MEDIC loaded with", len(dd_medic.keys()), "concepts.")

            print("Loading NCBI-Disease...")
            ddd_NCBI_train = pubannotation_to_python_corpus(ncbi_norm_train_folder)
            ddd_NCBI_dev = pubannotation_to_python_corpus(ncbi_norm_dev_folder)
            print("\n\tNb of doc in train:", len(ddd_NCBI_train))
            print("\n\tNb of doc in dev:", len(ddd_NCBI_dev))
            print("\n\tNb of mentions in train:", sum(len(sub_dict.keys()) for sub_dict in ddd_NCBI_train.values()))
            print("\n\tNb of mentions in dev:", sum(len(sub_dict.keys()) for sub_dict in ddd_NCBI_dev.values()))

            print("\nTraining...")
            batchFilePath = "./tmp/"
            PREPROCESS_MODEL = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"  #
            TF_BERT_model = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-2-h-128-a-2/2"  #
            batchSize = 64  # 256  # 64
            preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(ddd_NCBI_train,
                                                                                             dd_medic, PREPROCESS_MODEL,
                                                                                             TF_BERT_model, batchFilePath,
                                                                                             batchSize, verbose=1,
                                                                                             mode="pooled_output")
            print("training done.")

            print("\nPrediction...")
            ddd_normalized_NCBI_dev = twoStep_finetuned_quicknorm_predict(ddd_NCBI_dev, dd_medic,
                                                                                         preprocessor, bert_encoder, weights,
                                                                                         TFmodel, verbose=0,
                                                                                         mode="pooled_output")
            print("Prediction done.")

            end = time.time()
            print(end - start, "sec de temps d'execution.")
            print("Accuracy:", accuracy(ddd_normalized_NCBI_dev))


        elif option == 'classif':

            print("\nQuickNorm_v2 on BB4:")

            bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
            bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
            bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
            bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"

            print("Loading OntoBiotope...")
            dd_obt = loader_ontobiotope("./datasets/BB4/OntoBiotope_BioNLP-OST-2019.obo")
            dd_obt_hab = select_subpart_hierarchy(dd_obt, "OBT:000001")
            print("OntoBiotope loaded with", len(dd_obt_hab.keys()), "habitat concepts.")

            print("Loading BB4 Habitats corpus...")
            ddd_BB4_hab_train = pubannotation_to_python_corpus(bb4_norm_train_folder, l_type=["Habitat"])
            ddd_BB4_hab_dev = pubannotation_to_python_corpus(bb4_norm_dev_folder, l_type=["Habitat"])
            print("\n\tNb of doc in train:", len(ddd_BB4_hab_train))
            print("\n\tNb of doc in dev:", len(ddd_BB4_hab_dev))
            print("\n\tNb of mentions in train:", sum(len(sub_dict.keys()) for sub_dict in ddd_BB4_hab_train.values()))
            print("\n\tNb of mentions in dev:", sum(len(sub_dict.keys()) for sub_dict in ddd_BB4_hab_dev.values()))
            print("Done.")

            print("\nTraining...")
            batchFilePath = "./tmp/"
            PREPROCESS_MODEL = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"  # 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'  #
            TF_BERT_model = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-2-h-128-a-2/2"  # 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'  #
            batchSize = 64  # 256  # 64
            preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train_v2(ddd_BB4_hab_train,
                                                                                             dd_obt_hab, PREPROCESS_MODEL,
                                                                                             TF_BERT_model, batchFilePath,
                                                                                             batchSize, verbose=1,
                                                                                             mode="pooled_output")
            print("training done.")

            print("\nPrediction...")
            ddd_normalized_BB4_hab_dev = twoStep_finetuned_quicknorm_predict_v2(ddd_BB4_hab_dev, dd_obt_hab,
                                                                                         preprocessor, bert_encoder, weights,
                                                                                         TFmodel, verbose=0,
                                                                                         mode="pooled_output")
            print("Prediction done.")

            end = time.time()
            print(end - start, "sec de temps d'execution.")
            print("Accuracy:", accuracy(ddd_normalized_BB4_hab_dev))


        else:
            print(f"Option inconnue : {option}")


    ##################################################################################################################

    print("\n\n\n")

    parser = argparse.ArgumentParser(description="Experimental script to test QuickNorm on BB4-Habitat and NCBI-Disease Corpus.")
    parser.add_argument('--option', type=str, help='Choose between "bb4" and "ncbi".', required=True)
    args = parser.parse_args()

    main(args.option)












