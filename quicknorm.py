
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
from tensorflow import math, shape, zeros, string, data, constant, not_equal, cast, reduce_sum, expand_dims, float32, where, squeeze, tensor_scatter_nd_update, scatter_nd, concat, int32, tile
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend, callbacks, activations
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
    scoreMatrix = cdist(predMatrix, labtagsVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
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

def get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix, dd_labelsVectors, dd_predictions, maxScores):
    
    dim = predMatrix.shape[1]
    labtagsVectorMatrix = numpy.zeros((numberOfTags, dim))
    i = 0
    for cui in dd_labelsVectors.keys():
        for labtag in dd_labelsVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_labelsVectors[cui][labtag]
            i += 1

    print('\tMatrix of distance calculation...')
    scoreMatrix = cdist(predMatrix, labtagsVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
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

def get_nearest_concept(l_docWithMentions, numberOfMentions, predMatrix, d_onto, TFhubPreprocessModel, TFhubModel, mode="sequence_output"):
    
    batch_size = 1000
    
    maxScores = numpy.zeros(numberOfMentions)
    dd_predictions = []
    for i in range(numberOfMentions):
        dd_predictions.append("")
    
    current_size = 0
    dd_concepts = dict()
   
    for cui in d_onto:
        if (current_size == batch_size):
            # create concept embeddings for the current batch
            dd_conceptVectors, numberOfTags = get_vectors_as_dict(dd_concepts, TFhubPreprocessModel, TFhubModel, mode=mode)
            # compute nearest neighbors for the current batch and keep track of max scores
            dd_predictions, maxScores = get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix, dd_conceptVectors, dd_predictions, maxScores)
            # reinitialize
            current_size = 0
            dd_concepts = dict()
            
        # store current concept
        dd_concepts[cui] = d_onto[cui]
        current_size+=1
        
    # process the remainder
    if (dd_concepts):
        dd_conceptVectors, numberOfTags = get_vectors_as_dict(dd_concepts, TFhubPreprocessModel, TFhubModel, mode=mode)
        dd_predictions, maxScores = get_nearest_concept_in_batch(l_docWithMentions, numberOfTags, predMatrix, dd_conceptVectors, dd_predictions, maxScores)
        
    # store predictions in spacy docs
    i = 0
    for doc in l_docWithMentions:
        for mention in list(doc.spans["mentions"]):
            if len(mention._.pred_kb_id_) == 0:  # exact match + by heart doesn't predict
                mention._.pred_kb_id_.add(dd_predictions[i])
            i+=1
                 
    return l_docWithMentions

def prepare_training_data(l_trainDoc, dd_conceptVectors, d_spacyOnto, addLabels=True, addMentions=True):

    l_mentions = list()
    l_conceptVectors = list()

    if addMentions == True:
        nbMentions = 0
        for doc in l_trainDoc:
            nbMentions += len(doc.spans["mentions"])
            for mention in doc.spans["mentions"]:
                for cui in list(mention._.kb_id_):
                    for tag in dd_conceptVectors[cui].keys():
                        l_conceptVectors.append(dd_conceptVectors[cui][tag])
                        l_mentions.append(mention.text.lower())
        nbConcepts = len(d_spacyOnto.keys())
        print("\tNumber of training mentions:", nbMentions, "- number of concepts:", nbConcepts)

    if addLabels == True:
        # Add labels/synonyms to training data: (aim: stabilizing the output vectors)
        # Number of examples before: 1945 1945
        for cui in d_spacyOnto.keys():
            for tag in dd_conceptVectors[cui].keys():
                l_conceptVectors.append(dd_conceptVectors[cui][tag])
                l_mentions.append(tag)

    if addLabels == False and addMentions == False:
        print("ERROR: need at least to train on labels or on traning mentions")
        sys.exit(0)

    return l_mentions, l_conceptVectors

def write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath):
    
    # Add some random examples to complete the last batch:
    # numberInBatch has stopped with last example
    if numberInBatch < batchSize-1:  # If the last batch is not perfectly full
        print("Completing batch "+str(batchNb))
        for i in range(numberInBatch+1, batchSize):
            randomBatchNumber = random.choice(list(range(batchNb-1)))
            X_filename = "batch"+str(randomBatchNumber)+"_X.npy"
            random_X = numpy.loadtxt(batchFilePath+X_filename, delimiter="\t", dtype="str")
            randomExampleNumber = random.choice(list(range(batchSize-1)))
            l_mentions.append(str(random_X[randomExampleNumber]))
            Y_filename = "batch"+str(randomBatchNumber)+"_Y.npy"
            random_Y = numpy.load(batchFilePath+Y_filename)
            concept_vectors.append(random_Y[randomExampleNumber])
    
    print("Writing batch "+str(batchNb))
    filename = "batch"+str(batchNb)+"_X.npy"
    file_object = open(batchFilePath+filename, "wb")
    numpy.savetxt(file_object,l_mentions,fmt="%s",encoding='utf8')
    file_object.close()
    filename = "batch"+str(batchNb)+"_Y.npy"
    file_object = open(batchFilePath+filename, "wb")
    numpy.save(file_object, numpy.array(concept_vectors))
    file_object.close()

def prepare_training_data_in_batches(l_trainDoc, d_onto, batchFilePath, batchSize, preprocessor, bert_encoder, addLabels=True, addMentions=True, mode="pooled_output"):
    
    # empty batch folder
    files = glob.glob(batchFilePath+'/batch*')
    for f in files:
        os.remove(f)
    
    # build batches
    batchNb = 0
    nbMentions = 0
    
    if addMentions == True:
        l_mentions = []
        l_concepts = []
        count = 0
        numberInBatch = 0
        for doc in l_trainDoc:
            nbMentions += len(doc.spans["mentions"])
            for mention in doc.spans["mentions"]:
                for cui in list(mention._.kb_id_):
                    for tag in d_onto[cui]:
                        count += 1
                        numberInBatch = count % batchSize
                        l_mentions.append(" ".join(mention.text.lower().split()))
                        l_concepts.append(tag)
                        
                        if numberInBatch == (batchSize - 1):
                            # get concept vectors
                            concept_vectors = get_vectors(l_concepts, preprocessor, bert_encoder, mode=mode)
                            # write batch
                            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)
                            # re-initialize
                            batchNb += 1
                            l_mentions = []
                            l_concepts = []
                
        # Deal with the last batch
        if(len(l_mentions) > 0):
            concept_vectors = get_vectors(l_concepts, preprocessor, bert_encoder, mode=mode)
            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)
         
    if addLabels == True:
        # Add labels/synonyms to training data: (aim: stabilizing the output vectors)
        # Number of examples before: 1945 1945
        print("Prepare labels for training...")
        # create vectors in batches
        l_mentions = []
        count = 0
        numberInBatch = 0
        for cui in d_onto:
            for tag in d_onto[cui]:
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
        if(len(l_mentions) > 0):
            concept_vectors = get_vectors(l_mentions, preprocessor, bert_encoder, mode=mode)
            write_batch(l_mentions, concept_vectors, numberInBatch, batchNb, batchSize, batchFilePath)

    if addLabels == False and addMentions == False:
        print("ERROR: need at least to train on labels or on traning mentions")
        sys.exit(0)

def generator1input(datasetPath, BatchNb):
    batchValue = 0
    while True:
        if batchValue < BatchNb:
            X_batch = numpy.loadtxt(datasetPath+"batch"+str(batchValue)+"_X.npy", dtype='str', delimiter = "\t")
            Y_batch = numpy.load(datasetPath+"batch"+str(batchValue)+"_Y.npy")

            yield X_batch, Y_batch

            if batchValue < BatchNb-1:
                batchValue += 1
            else:
                batchValue = 0
        else:
            print("ERROR: Batch size error...")
            sys.exit(0)

def submodel(l_mentions, l_conceptVectors, preprocessor, bert_encoder, d_spacyOnto, epoch=10, patience=10, delta=0.0001, verbose=0, evaluate=False, mentionsVal=None, conceptsVectors=None, shuffle=False):

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
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=['cosine_similarity', 'logcosh'])  # losses.CosineSimilarity()
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

            #print(f"Epoch {i_epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

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

def submodel_with_batches(preprocessor, bert_encoder, trainingDataPath, epoch=10, patience=10, delta=0.0001, verbose=0, shuffle=False):

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
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=['cosine_similarity', 'logcosh'])  # losses.CosineSimilarity()
    TFmodel.summary() if verbose == True else None

    callback = callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=delta)
    history = TFmodel.fit(my_generator, epochs=epoch, callbacks=[callback], verbose=verbose, steps_per_epoch=numberOfBatches)

    return preprocessor, bert_encoder, TFmodel

def twoStep_finetuned_quicknorm(l_trainDoc, l_valDoc, d_spacyOnto, PREPROCESS_MODEL, TF_BERT_MODEL, spacyNlp=None, verbose=0, mode="pooled_output"):
    """
    ToDo: keep the trained dense layer ; try ONLY labels first (here, labels and mentions)
    ToDO: train without the easy prediction from an exact match (to focus on hard case and gain speed)
    ToDo: estimate the gain/loss in spped and accyracy with first beginning with exact match

    # < 0,600 with samples+labels for the two steps (thus more than 1 pt less, and moreover, more than 3min to calculate)
    400 sec for Accuracy: 0.619 with 10 epochs with labels+mentions then 20 epochs with training samples (logcosh: 3.7070e-04) - 6min40sec
    403 sec for Accuracy: 0.612                                                                 (logcosh: 3.6927e-04)
    426 sec for Accuracy: 0.603                                                                 (logcosh: 3.8254e-04)
    362 sec for Accuracy: 0.605                                                                 (logcosh: 3.5360e-04)
    381 sec for Accuracy: 0.615                                                                 (logcosh: 3.5936e-04)
    402 sec for Accuracy: 0.618                                                                 (logcosh: 4.0720e-04)

    412 sec for Accuracy: 0.621 with 10 epochs with shuffled labels+mentions then 20 epochs with training samples (logcosh: 3.3993e-04)
    370 sec for Accuracy: 0.609
    384 sec for Accuracy: 0.597
    411 sec for Accuracy: 0.585
    383 sec for Accuracy: 0.620
    418 sec for Accuracy: 0.600
    419 sec for Accuracy: 0.617

    420 sec for Accuracy: 0.619 with 10 epochs with shuffled labels+mentions then 10 with shuffled training samples then 10 epochs with training samples (logcosh: 1.7926e-04)
    439 sec for Accuracy: 0.595
    439 sec for Accuracy: 0.613
    432 sec for Accuracy: 0.613

    497 sec for Accuracy: 0.600 with 10 epochs with shuffled labels+mentions then 10 with shuffled training samples then 20 epochs with training samples (logcosh: 1.9844e-04)
    500 sec for Accuracy: 0.595

    457 sec for Accuracy: 0.613 with 10 epochs with shuffled labels+mentions then 10 with shuffled training samples then 15 epochs with training samples (logcosh: 1.9844e-04)
    457 sec for Accuracy: 0.601

    365 sec for Accuracy: 0.606 with 8 epochs with labels then 20 epochs with training samples (logcosh: 4.1877e-04)
    380 sec for Accuracy: 0.603 with 8 epochs with shuffled labels+mentions then 20 epochs with training samples (logcosh: 4.1877e-04)
    370 sec for Accuracy: 0.606

    422 sec for Accuracy: 0.602 with 20 epochs with shuffled labels+mentions
    423 sec for Accuracy: 0.611


    # Round 1 only on onto, then round 2 only on training mentions:
    333 sec for Accuracy: 0.52? with 10 epochs with labels then 20 epochs with training samples (logcosh: 4.9995e-04)
    393 sec for Accuracy: 0.553 with 10 epochs with labels then 30 epochs with training samples (logcosh: 4.4657e-04)
    405 sec for Accuracy: 0.576 with 10 epochs with labels then 10, 20 epochs with training samples (logcosh: 2.6160e-04)
    417 sec for Accuracy: 0.538 with 10, 5 epochs with labels then 20 epochs with training samples (logcosh: 3.3862e-04)
    360 sec for Accuracy: 0.567 with 10 epochs with labels then 10, 10 epochs with training samples (logcosh: 2.6751e-04)
    432 sec for Accuracy: 0.551 with 10 epochs with labels then 5, 5, 20 epochs with training samples (logcosh: 1.1226e-04)

    343 sec for Accuracy: 0.597 with 10 epochs with labels then 10 epochs with training samples

    476 sec for Accuracy: 0.59 with 10 epochs with shuffled labels+mentions then 30 epochs with training samples (logcosh: 2.9728e-04)
    470 sec for Accuracy: 0.583

    323 sec for Accuracy: 0.587 with 5 epochs with shuffled labels+mentions then 20 epochs with training samples (logcosh: 3.6258e-04)
    314 sec for Accuracy: 0.572

    424 sec for Accuracy: 0.588 with 10 epochs with shuffled labels+mentions then 20 epochs with shuffled training samples (logcosh: 3.3050e-04)
    425 sec for Accuracy: 0.598

    507 sec for Accuracy: 0.608 with 20 epochs with shuffled labels+mentions, then 10 epochs with training samples ()
    548 sec for Accuracy: 0.596
    520 sec for Accuracy: 0.592

    401 sec for Accuracy: 0.57? with 10 epochs with labels+mentions then 20 epochs with shuffled training samples (logcosh: 3.4910e-04)
    411 sec for Accuracy: 0.588

    453 sec for Accuracy: 0.611 with 10 epochs with labels then 30 epochs with training samples (logcosh: 3.5865e-04)
    569 sec for Accuracy: 0.600 with 20 epochs with labels then 20 epochs with training samples (logcosh: 4.1428e-04)

    501 sec for Accuracy: 0.601 with 15 epochs with labels then 20 epochs with training samples (logcosh: 3.8690e-04)
    438 sec for Accuracy: 0.588 with 10 epochs with labels then 25 epochs with training samples (logcosh: 3.6593e-04)
    362 sec for Accuracy: 0.595 with 10 epochs with labels then 15 epochs with training samples (logcosh: 3.7738e-04)
    1051 ? for Accuracy: 0.616 with 10 then 5 epochs with labels then 15(20?) epochs with training samples (logcosh: 1.1240e-04)
    584 sec for Accuracy: 0.594 with 5 then 5 epochs with labels then 15(20?) epochs with training samples (logcosh: 1.9872e-04)
    ??? sec for Accuracy: 0.604 with 10 then 5 epochs with labels then 20 epochs with training samples (logcosh: 1.0310e-04)
    10, then training samples 5, 20 : fail

    574 sec for Accuracy: 0.603 with 30 epochs with shuffled labels+mentions
    777 sec for Accuracy: 0.600 with 40 epochs with shuffled labels+mentions

    NB: C-Norm on BB4-Hab = 0.633
    1644 sec for Accuracy: 0.579 for a bigger Small BERT  with 10 epochs with labels+mentions then 20 epochs with training samples (logcosh: 3.0542e-04)
    """

    ######
    # Loading TF Hub model
    ######
    preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
    bert_encoder = hub.KerasLayer(TF_BERT_MODEL, trainable=True)

    ######
    # Target finetuning:
    ######
    labelVectorMode = "sequence_output"  # "sequence_output" / "pooled_output" (stangely work less)

    dd_conceptVectors, nbTags = get_list_of_tag_vectors(d_spacyOnto, preprocessor, bert_encoder, spanKey="synonyms", lower=True, mode=labelVectorMode)
    print("\t", nbTags, "tags embeddings loaded.")
    l_mentions, l_conceptVectors = prepare_training_data(l_trainDoc, dd_conceptVectors, d_spacyOnto, addLabels=True, addMentions=True)
    print("Number of training samples:", len(l_mentions))
    preprocessor, bert_encoder, TFmodel = submodel(l_mentions, l_conceptVectors, preprocessor, bert_encoder, d_spacyOnto, epoch=10, patience=30, delta=0.0001, verbose=1, shuffle=False)
    # del TFmodel
    # del dd_conceptVectors

    print("\n\n")

    dd_conceptVectors, nbTags = get_list_of_tag_vectors(d_spacyOnto, preprocessor, bert_encoder, spanKey="synonyms", lower=True, mode=labelVectorMode)
    l_mentions, l_conceptVectors = prepare_training_data(l_trainDoc, dd_conceptVectors, d_spacyOnto, addLabels=False, addMentions=True)
    print("Number of training samples:", len(l_mentions))
    preprocessor, bert_encoder, TFmodel = submodel(l_mentions, l_conceptVectors, preprocessor, bert_encoder, d_spacyOnto, epoch=20, patience=30, delta=0.0001, verbose=1, shuffle=False)

    ######
    # Regression prediction:
    ######
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
    l_valDoc = get_nearest_cui(l_valDoc, nbTags, Y_pred, dd_conceptVectors)
    del dd_conceptVectors

    return l_valDoc

def twoStep_finetuned_quicknorm_train(l_trainDoc, d_spacyOnto, PREPROCESS_MODEL, TF_BERT_MODEL, batchFilePath, batchSize, verbose=0, mode="pooled_output"):

    # spacy onto to simple dictionary
    lowerCase = True
    syno = "synonyms"
    d_onto = spacy_onto_to_dict(d_spacyOnto,spanKey=syno, lower=lowerCase)
    
    ######
    # Loading TF Hub model
    ######
    preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
    bert_encoder = hub.KerasLayer(TF_BERT_MODEL, trainable=True)

    ######
    # Target finetuning:
    ######
    labelVectorMode = "sequence_output"  # "sequence_output" / "pooled_output" (stangely work less)

    prepare_training_data_in_batches(l_trainDoc, d_onto, batchFilePath, batchSize, preprocessor, bert_encoder, addLabels=True, addMentions=True, mode=labelVectorMode)
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=10, patience=30, delta=0.0001, verbose=1, shuffle=False)
    # del TFmodel

    print("\n\n")
    
    weights = bert_encoder.get_weights()

    prepare_training_data_in_batches(l_trainDoc, d_onto, batchFilePath, batchSize, preprocessor, bert_encoder, addLabels=False, addMentions=True, mode=labelVectorMode)
    preprocessor, bert_encoder, TFmodel = submodel_with_batches(preprocessor, bert_encoder, batchFilePath, epoch=20, patience=30, delta=0.0001, verbose=1, shuffle=False)

    return preprocessor, weights, bert_encoder, TFmodel

def twoStep_finetuned_quicknorm_predict(l_valDoc, d_spacyOnto, preprocessor, bert_encoder, weights, TFmodel, verbose=0, mode="pooled_output"):
    
    # spacy onto to simple dictionary
    lowerCase = True
    syno = "synonyms"
    d_onto = spacy_onto_to_dict(d_spacyOnto,spanKey=syno, lower=lowerCase)
    
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
    l_valDoc = get_nearest_concept(l_valDoc, nbMentionsInPred, Y_pred, d_onto, preprocessor, bert_encoder, mode=labelVectorMode)

    return l_valDoc

######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    
    output_folder_dev = "output/dev"
    output_folder_test = "output/test"
    
    batchFilePath = "./tmp/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath) 

    #bb4_norm = load_dataset(path="bigbio/bionlp_st_2019_bb", name="bionlp_st_2019_bb_norm_source")
    # print_hf_doc(bb4_norm['train'][0])  # A doc in BB4-norm train
    #print("An example in the doc bb4_norm['train'][0]:")
    #print_bb4_hf_mentions(bb4_norm['train'][0], nbExamples=1)

    #print("Extract normalization information and transfer into SpaCy format...")
    nlp = spacy.load("en_core_web_sm")
    #l_spacy_BB4 = hf_bb4_into_spacy_corpus([bb4_norm['validation']], l_type=["Habitat"], spacyNlp=nlp)

    print("Create an ontology in SpaCy (a list of concepts, each one as a SpaCy doc):")
    dd_obt = loader_ontobiotope("./datasets/BB4/OntoBiotope_BioNLP-OST-2019.obo")
    dd_obt_hab = select_subpart_hierarchy(dd_obt, "OBT:000001")
    d_spacyOBT = create_onto_from_ontobiotope_dict_v2(dd_obt_hab, nlp)

    print("\nQuickNorm:")

    import time
    start = time.time()

    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    #nlp = spacy.load("en_core_web_sm")
   # l_spacy_BB4_hab_train = hf_bb4_into_spacy_corpus([bb4_norm['train']], l_type=["Habitat"], spacyNlp=nlp)
   # l_spacy_BB4_hab_val = hf_bb4_into_spacy_corpus([bb4_norm['validation']], l_type=["Habitat"], spacyNlp=nlp)
    l_spacy_BB4_hab_train = pubannotation_to_spacy_corpus(bb4_norm_train_folder, l_type=["Habitat"], spacyNlp=nlp)
    l_spacy_BB4_hab_val = pubannotation_to_spacy_corpus(bb4_norm_dev_folder, l_type=["Habitat"], spacyNlp=nlp)
    l_spacy_BB4_hab_test = pubannotation_to_spacy_corpus(bb4_norm_test_folder, l_type=["Habitat"], spacyNlp=nlp)

   # l_spacyNormalizedBB4_by_quicknorm = twoStep_finetuned_quicknorm(l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, spacyNlp=nlp, verbose=1, mode="pooled_output")
    preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(l_spacy_BB4_hab_train, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, batchFilePath, batchSize, verbose=1, mode="pooled_output")
    l_spacyNormalizedBB4_by_quicknorm_val = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_val, d_spacyOBT, preprocessor, bert_encoder, weights, TFmodel, verbose=0, mode="pooled_output")
    l_spacyNormalizedBB4_by_quicknorm_test = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_test, d_spacyOBT, preprocessor, bert_encoder, weights, TFmodel, verbose=0, mode="pooled_output")
    
    print_pubannotation(l_spacyNormalizedBB4_by_quicknorm_val, output_folder_dev, "http://pubannotation.org/docs/sourcedb/BB-norm@ldeleger", "bionlp-ost-19-BB-norm-dev", "BB-norm@ldeleger", "OntoBiotope")
    
    print_pubannotation(l_spacyNormalizedBB4_by_quicknorm_val, output_folder_test, "http://pubannotation.org/docs/sourcedb/BB-norm@ldeleger", "bionlp-ost-19-BB-norm-test", "BB-norm@ldeleger", "OntoBiotope")
    
    end = time.time()

    """
    for doc in l_spacyNormalizedBB4_by_quicknorm:
        for mention in doc.spans["mentions"]:
            for cuiPred in list(mention._.pred_kb_id_):
                if cuiPred not in mention._.kb_id_:
                    print(mention.text, "   pred:", d_spacyOBT[list(mention._.pred_kb_id_)[0]].text, " --- gold:", d_spacyOBT[list(mention._.kb_id_)[0]].text)
    """

    print(end - start, "sec de temps d'execution.")

    print("Accuracy:", accuracy(l_spacyNormalizedBB4_by_quicknorm_val))
    





