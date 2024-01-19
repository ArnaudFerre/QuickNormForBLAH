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
from quicknorm import twoStep_finetuned_quicknorm_train, twoStep_finetuned_quicknorm_predict

#from memory_profiler import memory_usage
#from time import sleep
import time
import tracemalloc


# importing libraries
import os
import psutil
 
# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
 
        return result
    return wrapper

@profile
def train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path):
    
    preprocessor, weights, bert_encoder, TFmodel = twoStep_finetuned_quicknorm_train(l_spacy_BB4_hab_train, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, batchFilePath, batchSize, verbose=1, mode="pooled_output")
    
    TFmodel.save(model_path)
    filename = weights_path
    file_object = open(filename, "wb")
    numpy.savez(file_object, *weights)
    file_object.close()
    
    return preprocessor, weights, bert_encoder, TFmodel

@profile
def dev_test_quicknorm_after_training(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path):

#    l_spacyNormalizedBB4_by_quicknorm_val = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")
    l_spacyNormalizedBB4_by_quicknorm_test = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")

    from printers import spacy_into_a2
    spacy_into_a2(l_spacyNormalizedBB4_by_quicknorm_test, save_file=True, save_path=save_path, pred=True, align_type_onto={"Habitat": "OntoBiotope", "Microorganism": "NCBI_Taxonomy", "Phenotype": "OntoBiotope"})
    print("BB4 test predictions saved (a2 format) in", save_path)
    
@profile
def dev_quicknorm_after_training(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path):
    
    l_spacyNormalizedBB4_by_quicknorm_val = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")
    #l_spacyNormalizedBB4_by_quicknorm_test = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")

    from printers import spacy_into_a2
    spacy_into_a2(l_spacyNormalizedBB4_by_quicknorm_val, save_file=True, save_path=save_path, pred=True, align_type_onto={"Habitat": "OntoBiotope", "Microorganism": "NCBI_Taxonomy", "Phenotype": "OntoBiotope"})
    print("BB4 test predictions saved (a2 format) in", save_path)

@profile
def dev_test_quicknorm(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights_path, TFmodel_path, save_path):
    
    TFmodel = tf.keras.models.load_model(TFmodel_path)
    
    weights_npz = numpy.load(weights_path)
    weights = [w1[k] for k in weights_npz]
    
    l_spacyNormalizedBB4_by_quicknorm_val = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")
    l_spacyNormalizedBB4_by_quicknorm_test = twoStep_finetuned_quicknorm_predict(l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, verbose=0, mode="pooled_output")

    from printers import spacy_into_a2
    spacy_into_a2(l_spacyNormalizedBB4_by_quicknorm_test, save_file=True, save_path=save_path, pred=True, align_type_onto={"Habitat": "OntoBiotope", "Microorganism": "NCBI_Taxonomy", "Phenotype": "OntoBiotope"})
    print("BB4 test predictions saved (a2 format) in", save_path)

# instantiation of decorator function
@profile
def load_BB4_dataset(bb4_norm_train_folder, bb4_norm_dev_folder, bb4_norm_test_folder, entity_types, onto_subpart_id):
    
    nlp = spacy.load("en_core_web_sm")
    #l_spacy_BB4 = hf_bb4_into_spacy_corpus([bb4_norm['validation']], l_type=["Habitat"], spacyNlp=nlp)

    print("Create an ontology in SpaCy (a list of concepts, each one as a SpaCy doc):")
    dd_obt = loader_ontobiotope("./datasets/BB4/OntoBiotope_BioNLP-OST-2019.obo")
    dd_obt_hab = select_subpart_hierarchy(dd_obt, onto_subpart_id)
    d_spacyOBT = create_onto_from_ontobiotope_dict_v2(dd_obt_hab, nlp)
    
    l_spacy_BB4_hab_train = pubannotation_to_spacy_corpus(bb4_norm_train_folder, l_type=entity_types, spacyNlp=nlp)
    l_spacy_BB4_hab_val = pubannotation_to_spacy_corpus(bb4_norm_dev_folder, l_type=entity_types, spacyNlp=nlp)
    l_spacy_BB4_hab_test = pubannotation_to_spacy_corpus(bb4_norm_test_folder, l_type=entity_types, spacyNlp=nlp)
    
    return d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test
    
    
def BB4_habitat_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
    
    output_folder_dev = "output/dev"
    output_folder_test = "output/test"
    save_path = "datasets/BB4/predictions/"
    
    model_path = 'models/BB4traindev_habitat_model.keras'
    weights_path = "models/BB4traindev_habitat_encoder_weights.npz"
    
    batchFilePath = "./tmp/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
        
    # starting the monitoring
  #  tracemalloc.start()
        
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_traindev_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Habitat"], "OBT:000001")
    
    '''
     # displaying the memory
    memory_loading = tracemalloc.get_traced_memory()
 
    # stopping the library
    tracemalloc.stop()
    
    tracemalloc.start()
    '''
    
    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)
    #memory_training = tracemalloc.get_traced_memory()
    #tracemalloc.stop()
    
    #tracemalloc.start()
    dev_test_quicknorm_after_training(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)
    #memory_predicting = tracemalloc.get_traced_memory()
    #tracemalloc.stop()
    '''
    print("Dataset loading: ")
    print(memory_loading)
    print("Training: ")
    print(memory_training)
    print("Predicting: ")
    print(memory_predicting)
    '''
    
def BB4_habitat_dev_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    
    output_folder_dev = "output/dev"
    output_folder_test = "output/test"
    save_path = "datasets/BB4/predictions-dev/"
    
    model_path = 'models/BB4train_habitat_model.keras'
    weights_path = "models/BB4train_habitat_encoder_weights.npz"
    
    batchFilePath = "./tmp/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
        
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_train_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Habitat"], "OBT:000001")

    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)

    dev_quicknorm_after_training(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)
    
    
def BB4_habitat_time_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
    
    output_folder_dev = "output/dev"
    output_folder_test = "output/test"
    save_path = "datasets/BB4/predictions/"
    
    model_path = 'models/BB4traindev_habitat_model.keras'
    weights_path = "models/BB4traindev_habitat_encoder_weights.npz"
    
    batchFilePath = "./tmp/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
        
    # starting the monitoring
    start = time.time()
        
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_traindev_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Habitat"], "OBT:000001")
 
    # stop
    end = time.time()
    
    # get time
    time_loading = end - start
    
    start = time.time()
    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)
    memory_training = tracemalloc.get_traced_memory()
    end = time.time()
    
    time_training = end - start
    
    start = time.time()
    dev_test_quicknorm_after_training(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)

    end = time.time()
    
    time_predicting = end - start
    
    print("Dataset loading: ")
    print(time_loading)
    print("Training: ")
    print(time_training)
    print("Predicting: ")
    print(time_predicting)
    
    
def BB4_phenotype_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
    
    output_folder_dev = "output/dev-pheno"
    output_folder_test = "output/test-pheno"
    save_path = "datasets/BB4/predictions-pheno/"
    
    model_path = 'models/BB4train_pheno_model.keras'
    weights_path = "models/BB4train_pheno_encoder_weights.npz"
    
    batchFilePath = "./tmp-pheno/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
        
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_traindev_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Phenotype"], "OBT:000002")
    
    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)

    dev_test_quicknorm_after_training(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)
    
def BB4_phenotype_time_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    bb4_norm_traindev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-traindev/"
    
    output_folder_dev = "output/dev-pheno"
    output_folder_test = "output/test-pheno"
    save_path = "datasets/BB4/predictions-pheno/"
    
    model_path = 'models/BB4train_pheno_model.keras'
    weights_path = "models/BB4train_pheno_encoder_weights.npz"
    
    batchFilePath = "./tmp-pheno/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
       
    start = time.time()
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_traindev_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Phenotype"], "OBT:000002")
    end = time.time()
    
    # get time
    time_loading = end - start
    
    start = time.time()
    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)
    end = time.time()
    time_training = end - start
    
    start = time.time()
    dev_test_quicknorm_after_training(l_spacy_BB4_hab_val, l_spacy_BB4_hab_test, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)

    end = time.time()
    
    time_predicting = end - start
    
    print("Dataset loading: ")
    print(time_loading)
    print("Training: ")
    print(time_training)
    print("Predicting: ")
    print(time_predicting)
    
def BB4_phenotype_dev_experiment():
    
    PREPROCESS_MODEL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    TF_BERT_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2'
    
    batchSize = 64
    
    bb4_norm_train_folder = "datasets/BB4/bionlp-ost-19-BB-norm-train/"
    bb4_norm_dev_folder = "datasets/BB4/bionlp-ost-19-BB-norm-dev/"
    bb4_norm_test_folder = "datasets/BB4/bionlp-ost-19-BB-norm-test/"
    
    output_folder_dev = "output/dev-pheno"
    output_folder_test = "output/test-pheno"
    save_path = "datasets/BB4/predictions-pheno-dev/"
    
    model_path = 'models/BB4train_pheno_model.keras'
    weights_path = "models/BB4train_pheno_encoder_weights.npz"
    
    batchFilePath = "./tmp-pheno/"
  
    if not os.path.exists(output_folder_dev): 
        os.makedirs(output_folder_dev)
        
    if not os.path.exists(output_folder_test): 
        os.makedirs(output_folder_test)
        
    if not os.path.exists(batchFilePath): 
        os.makedirs(batchFilePath)
        
    d_spacyOBT, l_spacy_BB4_hab_train, l_spacy_BB4_hab_val, l_spacy_BB4_hab_test = load_BB4_dataset(bb4_norm_train_folder, bb4_norm_dev_folder, bb4_norm_test_folder, ["Phenotype"], "OBT:000002")
    
    preprocessor, weights, bert_encoder, TFmodel = train_quicknorm(l_spacy_BB4_hab_train, d_spacyOBT, batchFilePath, batchSize, PREPROCESS_MODEL, TF_BERT_model, model_path, weights_path)

    dev_quicknorm_after_training(l_spacy_BB4_hab_val, d_spacyOBT, PREPROCESS_MODEL, TF_BERT_model, weights, TFmodel, save_path)


if __name__ == '__main__':
    
    #BB4_habitat_experiment()
    #BB4_habitat_dev_experiment()
    #BB4_phenotype_experiment()
    #BB4_phenotype_dev_experiment()
    
    BB4_habitat_time_experiment()
    #BB4_phenotype_time_experiment()
        
    
    
