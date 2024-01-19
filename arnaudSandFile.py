# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################

import sys
import os
import glob
from os import listdir

import random

from datasets import load_dataset_builder, load_dataset, get_dataset_config_names

import spacy
from spacy.tokens import Doc

from pronto import Ontology

from nltk.stem import WordNetLemmatizer, PorterStemmer

import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import math, shape, zeros, string, data, constant, not_equal, cast, reduce_sum, expand_dims, float32, \
    where, squeeze, tensor_scatter_nd_update, scatter_nd, concat, int32, tile
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, \
    backend, callbacks, activations
import numpy
from scipy.spatial.distance import cosine, euclidean, cdist

import matplotlib.pyplot as plt

from loaders import loader_ontobiotope, pubannotation_to_spacy_corpus
from printers import print_bb4_hf_mentions, print_pubannotation
from converters import hf_bb4_into_spacy_corpus, create_onto_from_ontobiotope_dict_v2, spacy_onto_to_dict
from utils import select_subpart_hierarchy, get_list_of_tag_vectors, get_vectors_as_dict, get_vectors
from evaluation import accuracy


######################################################################################################################
# Method
######################################################################################################################

def lowercase_tokens(doc):
    new_tokens = [token.lower_ for token in doc]
    return spacy.tokens.Doc(doc.vocab, words=new_tokens)


def get_nearest_cui(l_docWithMentions, numberOfTags, predMatrix, dd_labelsVectors):
    """
    :param predMatrix: ambiguous name... It's the list of the predictions for each mention from l_docWithMentions.
    :return: The list of documents with predicted concepts for each mention.
    """
    dim = predMatrix.shape[1]
    labtagsVectorMatrix = numpy.zeros((numberOfTags, dim))
    i = 0
    for cui in dd_labelsVectors.keys():
        for labtag in dd_labelsVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_labelsVectors[cui][labtag]
            i += 1

    print('\tMatrix of distance calculation... (time consuming)')
    scoreMatrix = cdist(predMatrix, labtagsVectorMatrix,
                        'cosine')  # cdist() is an optimized algo to distance calculation.
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    # Just to obtain the true cosine value
    i = 0
    for doc in l_docWithMentions:
        for mention in list(doc.spans["mentions"]):
            j = 0
            for cui in dd_labelsVectors.keys():
                for labtag in dd_labelsVectors[cui].keys():
                    scoreMatrix[i][j] = 1 - scoreMatrix[i][j]
                    j += 1
            i += 1
    print("\t\tDone.")

    # For each mention, find back the nearest label/tag vector, then attribute the associated concept:
    i = 0
    for doc in l_docWithMentions:
        for mention in list(doc.spans["mentions"]):
            if len(mention._.pred_kb_id_) == 0:  # exact match + by heart doesn't predict
                j_max_score = numpy.argmax(scoreMatrix[i])
                j = -1
                stopSearch = False
                for cui in dd_labelsVectors.keys():
                    if stopSearch == True:
                        break
                    for labtag in dd_labelsVectors[cui].keys():
                        j += 1
                        if j == j_max_score:
                            mention._.pred_kb_id_.add(cui)
                            stopSearch = True
                            break
            i += 1

    return l_docWithMentions


def get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix, dd_labelsVectors, dd_predictions,
                                 maxScores):
    dim = predMatrix.shape[1]
    labtagsVectorMatrix = numpy.zeros((numberOfTags, dim))
    i = 0
    for cui in dd_labelsVectors.keys():
        for labtag in dd_labelsVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_labelsVectors[cui][labtag]
            i += 1

    print('\tMatrix of distance calculation...')
    scoreMatrix = cdist(predMatrix, labtagsVectorMatrix,
                        'cosine')  # cdist() is an optimized algo to distance calculation.
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    # Just to obtain the true cosine value
    i = 0
    for doc in l_docWithMentions:
        for mention in list(doc.spans["mentions"]):
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


def get_nearest_concept(l_docWithMentions, numberOfMentions, predMatrix, dd_onto, TFhubPreprocessModel, TFhubModel,
                        mode="sequence_output"):
    batch_size = 1000

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
            dd_predictions, maxScores = get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix,
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
        dd_predictions, maxScores = get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix,
                                                                 dd_conceptVectors, dd_predictions, maxScores)

    # store predictions in spacy docs
    i = 0
    for doc in l_docWithMentions:
        for mention in list(doc.spans["mentions"]):
            if len(mention._.pred_kb_id_) == 0:  # exact match + by heart doesn't predict
                mention._.pred_kb_id_.add(dd_predictions[i])
            i += 1

    return l_docWithMentions


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


def prepare_training_data_in_batches(l_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=True, addMentions=True, mode="pooled_output"):
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
        for doc in l_trainDoc:
            nbMentions += len(doc.spans["mentions"])
            for mention in doc.spans["mentions"]:
                l_tags = list()
                for cui in list(mention._.kb_id_):
                    l_tags.append(dd_onto[cui]["label"].lower())
                    for tag in dd_onto[cui]["tags"]:
                        l_tags.append(tag.lower())
                for tag in l_tags:
                    count += 1
                    numberInBatch = count % batchSize
                    l_mentions.append(" ".join(mention.text.lower().split()))
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

    print(time.strftime("%H:%M:%S", time.localtime()), "(beginning for labels...)")
    if addLabels == True:
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
                    batchNb += 1
                    l_mentions = []

        # Deal with the last batch
        if (len(l_mentions) > 0):
            concept_vectors = get_vectors(l_mentions, preprocessor, bert_encoder, mode=mode)
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


def submodel(l_mentions, l_conceptVectors, preprocessor, bert_encoder, d_spacyOnto, epoch=10, patience=10, delta=0.0001,
             verbose=0, evaluate=False, mentionsVal=None, conceptsVectors=None, shuffle=False):
    print("\tNumber of examples:", len(l_conceptVectors), len(l_mentions))
    dim = bert_encoder.variables[0].shape[1]
    print("\tSize of embeddings:", dim)

    train_dataset = data.Dataset.from_tensor_slices((l_mentions, l_conceptVectors))
    if shuffle == True:
        train_dataset = train_dataset.shuffle(train_dataset.cardinality())
    train_dataset = train_dataset.batch(64)

    text_input = Input(shape=(), dtype=string, name='inputs')
    encoder_inputs = preprocessor(text_input)
    bert_outputs = bert_encoder(encoder_inputs)

    pooled_output = bert_outputs["pooled_output"]  # [batch_size, 128].
    unitNormLayer = layers.UnitNormalization()(pooled_output)  # less good without...
    dense_layer = layers.Dense(dim, activation=None, kernel_initializer=initializers.Identity())(unitNormLayer)
    TFmodel = Model(inputs=text_input, outputs=dense_layer)
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(),
                    metrics=['cosine_similarity', 'logcosh'])  # losses.CosineSimilarity()
    TFmodel.summary() if verbose == True else None

    if evaluate == True:  # doesn't work...

        validation_dataset = data.Dataset.from_tensor_slices((mentionsVal, conceptsVectors))  # lists
        validation_dataset = validation_dataset.batch(64)

        num_epochs = epoch
        train_losses = []
        val_losses = []

        for i_epoch in range(num_epochs):
            history = TFmodel.fit(train_dataset, epochs=1, verbose=1)

            # Evaluate the loss on the validation dataset
            val_loss = TFmodel.evaluate(validation_dataset, verbose=0)

            train_losses.append(history.history['loss'][0])
            val_losses.append(val_loss)

            # print(f"Epoch {i_epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

        # Plot the training and validation losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)


    elif evaluate == False:

        callback = callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=delta)
        history = TFmodel.fit(train_dataset, epochs=epoch, callbacks=[callback], verbose=verbose)

    return preprocessor, bert_encoder, TFmodel


def submodel_with_batches(preprocessor, bert_encoder, trainingDataPath, epoch=10, patience=10, delta=0.0001, verbose=0,
                          shuffle=False):
    #    print("\tNumber of examples:", len(l_conceptVectors), len(l_mentions))
    dim = bert_encoder.variables[0].shape[1]
    print("\tSize of embeddings:", dim)

    numberOfBatches = int(len(listdir(trainingDataPath)) / 2)
    my_generator = generator1input(trainingDataPath, numberOfBatches)

    text_input = Input(shape=(), dtype=string, name='inputs')
    encoder_inputs = preprocessor(text_input)
    bert_outputs = bert_encoder(encoder_inputs)

    pooled_output = bert_outputs["pooled_output"]  # [batch_size, 128].
    unitNormLayer = layers.UnitNormalization()(pooled_output)  # less good without...
    dense_layer = layers.Dense(dim, activation=None, kernel_initializer=initializers.Identity())(unitNormLayer)
    TFmodel = Model(inputs=text_input, outputs=dense_layer)
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(),
                    metrics=['cosine_similarity', 'logcosh'])  # losses.CosineSimilarity()
    TFmodel.summary() if verbose == True else None

    callback = callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=delta)
    history = TFmodel.fit(my_generator, epochs=epoch, callbacks=[callback], verbose=verbose,
                          steps_per_epoch=numberOfBatches)

    return preprocessor, bert_encoder, TFmodel


def twoStep_finetuned_quicknorm_train(l_trainDoc, dd_onto, PREPROCESS_MODEL, TF_BERT_MODEL, batchFilePath,
                                      batchSize, verbose=0, mode="pooled_output"):
    # spacy onto to simple dictionary
    lowerCase = True
    syno = "synonyms"

    ######
    # Loading TF Hub model
    ######
    preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
    bert_encoder = hub.KerasLayer(TF_BERT_MODEL, trainable=True)

    ######
    # Target finetuning:
    ######
    labelVectorMode = "sequence_output"  # "sequence_output" / "pooled_output" (stangely work less)

    print("Prepating training data...")
    prepare_training_data_in_batches(l_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=True, addMentions=True, mode=labelVectorMode)
    print(time.strftime("%H:%M:%S", time.localtime()), "training data prepared.")

    # For BB4, 10 epochs. But too long for NCBI-DC, so reduce to 4
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=3,
                                                                patience=30, delta=0.0001, verbose=1, shuffle=False)
    # del TFmodel

    print("\n\n")

    weights = bert_encoder.get_weights()

    prepare_training_data_in_batches(l_trainDoc, dd_onto, batchFilePath, batchSize, preprocessor, bert_encoder,
                                     addLabels=False, addMentions=True, mode=labelVectorMode)
    # For BB4, 20 epochs. But too long for NCBI-DC, so reduce to 10
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=10,
                                                                patience=30, delta=0.0001, verbose=1, shuffle=False)

    return preprocessor, weights, bert_encoder, TFmodel


def twoStep_finetuned_quicknorm_predict(l_valDoc, dd_onto, preprocessor, bert_encoder, weights, TFmodel, verbose=0,
                                        mode="pooled_output"):

    ######
    # Regression prediction:
    ######
    labelVectorMode = "sequence_output"
    print("\tRegression prediction...")
    nbMentionsInPred = 0
    for doc in l_valDoc:
        nbMentionsInPred += len(doc.spans["mentions"])
    print("\tNumber of mentions for prediction:", nbMentionsInPred)
    dim = bert_encoder.variables[0].shape[1]
    Y_pred = numpy.zeros((nbMentionsInPred, dim))
    # Todo: optimize with predicting a full batch a the same time (not so helpful here because quick calculation with a dense layer)
    i = 0
    for doc in l_valDoc:
        for mention in list(doc.spans["mentions"]):
            x_test = [mention.text.lower()]
            Y_pred[i] = TFmodel.predict(x_test, verbose=0)[0]  # result of the regression for the i-th mention.
            i += 1
    print("\t\tDone.")

    ######
    # Nearest neighbours calculation for each mention and CUI attribution:
    ######
    bert_encoder.set_weights(weights)
    l_valDoc = get_nearest_concept(l_valDoc, nbMentionsInPred, Y_pred, dd_onto, preprocessor, bert_encoder,
                                   mode=labelVectorMode)

    return l_valDoc


######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':

    print("\nQuickNorm on NCBI-DC:")

    import time
    start = time.time()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    ncbi_norm_traindev_folder = "datasets/NCBI-DC/pubannotation-traindev-2/"
    ncbi_norm_test_folder = "datasets/NCBI-DC/pubannotation-test-2/"
    ncbi_norm_dev_folder = "datasets/NCBI-DC/pubannotation-dev-2/"
    ncbi_norm_train_folder = "datasets/NCBI-DC/pubannotation-train-2/"

    nlp = spacy.load("en_core_web_sm")
    l_NCBI_entity_types = ["SpecificDisease", "DiseaseClass", "Modifier", "CompositeMention"]

    print("Create an ontology in SpaCy (a list of concepts, each one as a SpaCy doc):")
    from loaders import loader_medic
    dd_medic = loader_medic("datasets/NCBI-DC/CTD_diseases_dnorm.tsv")
    print("MEDIC loaded.", len(dd_medic.keys()))

    l_NCBI_entity_types = ["SpecificDisease", "DiseaseClass", "Modifier", "CompositeMention"]
    l_spacy_NCBI_train = pubannotation_to_spacy_corpus(ncbi_norm_train_folder, l_type=l_NCBI_entity_types, spacyNlp=nlp)
    print("\nNb of doc in train:", len(l_spacy_NCBI_train))

    print("\nTraining...")
    batchFilePath = "./tmp/"
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    batchSize = 256  # 64

    preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(l_spacy_NCBI_train,
                                                                                     dd_medic, PREPROCESS_MODEL,
                                                                                     TF_BERT_model, batchFilePath,
                                                                                     batchSize, verbose=1,
                                                                                     mode="pooled_output")
    print("training done.")

    print("\nPrediction...")
    l_NCBI_entity_types = ["SpecificDisease", "DiseaseClass", "Modifier", "CompositeMention"]
    l_spacy_NCBI_dev = pubannotation_to_spacy_corpus(ncbi_norm_dev_folder, l_type=l_NCBI_entity_types, spacyNlp=nlp)
    print("\nNb of doc in train:", len(l_spacy_NCBI_train))
    l_spacyNormalizedNCBI_by_quicknorm_dev = twoStep_finetuned_quicknorm_predict(l_spacy_NCBI_dev, dd_medic,
                                                                                 preprocessor, bert_encoder, weights,
                                                                                 TFmodel, verbose=0,
                                                                                 mode="pooled_output")
    print("Prediction done.")


    end = time.time()
    print(end - start, "sec de temps d'execution.")

    print("Accuracy:", accuracy(l_spacyNormalizedNCBI_by_quicknorm_dev))

    """
    print("\nQuickNorm on BB4:")
    import time
    start = time.time()

    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"

    print("Create an ontology in SpaCy (a list of concepts, each one as a SpaCy doc):")
    dd_obt = loader_ontobiotope("./datasets/BB4/OntoBiotope_BioNLP-OST-2019.obo")
    dd_obt_hab = select_subpart_hierarchy(dd_obt, "OBT:000001")

    nlp = spacy.load("en_core_web_sm")
    d_spacy_hab_OBT = create_onto_from_ontobiotope_dict_v2(dd_obt_hab, nlp)

    l_spacy_BB4_hab_train = pubannotation_to_spacy_corpus(bb4_norm_train_folder, l_type=["Habitat"], spacyNlp=nlp)
    l_spacy_BB4_hab_dev = pubannotation_to_spacy_corpus(bb4_norm_dev_folder, l_type=["Habitat"], spacyNlp=nlp)

    print("\nTraining...")
    batchFilePath = "./tmp/"
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    batchSize = 64
    preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(l_spacy_BB4_hab_traindev,
                                                                                     d_spacy_hab_OBT, PREPROCESS_MODEL,
                                                                                     TF_BERT_model, batchFilePath,
                                                                                     batchSize, verbose=1,
                                                                                     mode="pooled_output")
    print("training done.")

    print("\nPrediction...")
    l_spacyNormalizedBB4_by_quicknorm_test = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_test, d_spacy_hab_OBT,
                                                                                 preprocessor, bert_encoder, weights,
                                                                                 TFmodel, verbose=0,
                                                                                 mode="pooled_output")
    print("Prediction done.")
    
    from printers import spacy_into_a2
    save_path = "datasets/BB4/predictions/"
    BB4_dict = {"Habitat": "OntoBiotope", "Microorganism": "NCBI_Taxonomy","Phenotype": "OntoBiotope"}
    spacy_into_a2(l_spacyNormalizedBB4_by_quicknorm_test, save_file=True, save_path=save_path, pred=True,
                  align_type_onto=BB4_dict)
    print("BB4 test predictions saved (a2 format) in", save_path)


    end = time.time()
    print(end - start, "sec de temps d'execution.")
    """
    """
    for doc in l_spacyNormalizedBB4_by_quicknorm_test:
        print(doc.user_data["document_id"])
        for mention in doc.spans["mentions"]:
            for cuiPred in list(mention._.pred_kb_id_):
                if cuiPred not in mention._.kb_id_:
                    print(mention.text, "\t", d_spacyOBT[list(mention._.pred_kb_id_)[0]].text)
    """




    """
    
    # Correct ontology:
    l_wrong_cuis = list()
    with open("./datasets/NCBI-DC/wrongCUIList.txt") as my_file:
        for line in my_file:
            l_wrong_cuis.append(line[:-1])

    for doc in l_spacy_NCBI_test:
        for mention in doc.spans["mentions"]:
            for cui in mention._.kb_id_:
                (mention._.pred_kb_id_).add(cui)
    for doc in l_spacy_NCBI_test:
        for mention in doc.spans["mentions"]:
            for cui in mention._.kb_id_:
                if cui in l_wrong_cuis:
                    for correct_cui in dd_medic.keys():
                        if "alt_cui" in dd_medic[correct_cui].keys():
                            for altCui in dd_medic[correct_cui]["alt_cui"]:
                                if altCui == cui:
                                    (mention._.pred_kb_id_).remove(cui)
                                    (mention._.pred_kb_id_).add(correct_cui)
                                    break
    for doc in l_spacy_NCBI_test:
        for mention in doc.spans["mentions"]:
            for cui in mention._.kb_id_:
                mention._.kb_id_ = mention._.pred_kb_id_
                mention._.pred_kb_id_ = set()

    print_pubannotation(l_spacy_NCBI_test,
                        "datasets/NCBI-DC/pubannotation-test-2/",
                        "http://pubannotation.org/docs/sourcedb/PubMed/",
                        "NCBI-Disease-train",
                        "PubMed",
                        "MEDIC")

    """


