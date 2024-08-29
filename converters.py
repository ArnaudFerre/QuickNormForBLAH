
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################

import re
import os
import json

######################################################################################################################
# Method
######################################################################################################################
def ncbi_onto_pubannotation():
    pass

def ncbi_disease_to_pubannotation(ncbi_d_filename, puba_folder, project_uri, project_name, sourcedb, onto_name):
    
    if not os.path.exists(puba_folder): 
        os.makedirs(puba_folder) 
    
    with open(ncbi_d_filename, encoding="utf8") as file:
        
        nbdoc = 0
        puba_doc = dict()
        count_norm = 1
        count_mention = 1
        for line in file:
            
            if (re.search('^\d+\|t\|', line) is not None):
                
                if (nbdoc > 0): # print previous json doc
                    with open(puba_folder+"/"+ puba_doc['sourceid'] + '.json', 'w') as fp:
                        json.dump(puba_doc, fp)
                
                docid, flag, text = line.split("|")
                print(docid)
                # initialize document with current info
                puba_doc = dict()
                puba_doc['target'] = project_uri + "/sourceid/" + docid
                puba_doc['sourcedb'] = sourcedb
                puba_doc['sourceid'] = docid
                puba_doc['text'] = text
                puba_doc['project'] = project_name
                puba_doc['denotations'] = []
                puba_doc['attributes'] = []
                count_norm = 1
                count_mention = 1
                nbdoc += 1
                
            if (re.search('^\d+\|a\|', line) is not None):
                docid, flag, text = line.split("|")
                puba_doc['text'] = puba_doc['text'] + "\n" + text
                
            if (re.search('^\d+\t\d', line) is not None):
                docid, start, end, mention_text, label, cuis = line.split("\t")
                cuis = cuis.rstrip()
                mention_id = "T" + str(count_mention)
                mention_dict = dict()
                mention_dict['id'] = mention_id
                mention_dict['obj'] = label
                mention_dict['span'] = dict()
                mention_dict['span']['begin'] = start
                mention_dict['span']['end'] = end
                puba_doc['denotations'].append(mention_dict)
                count_mention += 1
                
                l_cuis = re.split('\||\+', cuis)
                for cui in l_cuis:
                    att_dict = dict()
                    att_dict['pred'] = onto_name
                    att_dict['id'] = "A" + str(count_norm)
                    att_dict['subj'] = mention_id
                    if (cui.strip() == "MESH:C535662"):
                        cui = "C535662"
                    att_dict['obj'] = cui.strip()
                    puba_doc['attributes'].append(att_dict)
                    count_norm += 1


def extract_data(ddd_data, l_type=[]):
    dd_data = dict()
    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = copy.deepcopy(ddd_data[fileName][id])
    return dd_data

def loader_one_ncbi_fold(l_foldPath):
    ddd_data = dict()

    i = 0
    for foldPath in l_foldPath:

        with open(foldPath, encoding="utf8") as file:

            fileNameWithoutExt, ext = splitext(basename(foldPath))
            ddd_data[fileNameWithoutExt] = dict()

            notInDoc = True
            nextAreMentions = False
            for line in file:

                if line == '\n' and nextAreMentions == True and notInDoc == False:
                    notInDoc = True
                    nextAreMentions = False


                if nextAreMentions == True and notInDoc == False:
                    l_line = line.split('\t')

                    exampleId = "ncbi_" + "{number:06}".format(number=i)
                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[3]
                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[4]

                    #Parfois une liste de CUIs (des '|' ou des '+'):
                    cuis = l_line[5].rstrip()
                    request11 = re.compile('.*\|.*')
                    request12 = re.compile('.*\+.*')
                    if request11.match(cuis):
                        #Composite mentions:
                        if ddd_data[fileNameWithoutExt][exampleId]["type"] == "CompositeMention":
                            l_cuis = cuis.split('|')
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = l_cuis
                        else: # Composite mention which are not typed as CompositeMention?
                            l_cuis = cuis.split('|')
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = l_cuis

                    elif request12.match(cuis):
                        l_cuis = cuis.split('+') #multi-normalization
                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = l_cuis
                    else:
                        if cuis.strip()=="MESH:C535662": # FORMATING ERROR in the initial testfold file of the NCBI Disease Corpus...
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = ["C535662"]
                        else:
                            ddd_data[fileNameWithoutExt][exampleId]["cui"] = [cuis.strip()]

                    i+=1


                request2 = re.compile('^\d*\|a\|')
                if nextAreMentions==False and request2.match(line) is not None:
                    nextAreMentions = True

                request3 = re.compile('^\d*\|t\|')
                if notInDoc==True and request3.match(line) is not None:
                    notInDoc = False


    return ddd_data


######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    pass

    #ncbi_disease_to_pubannotation("datasets/NCBI-DC/NCBItrainset_corpus.txt", "datasets/NCBI-DC/pubannotation-train", "http://pubannotation.org/docs/sourcedb/PubMed", "NCBI-Disease-train", "PubMed", "MEDIC")
    #ncbi_disease_to_pubannotation("datasets/NCBI-DC/NCBIdevelopset_corpus.txt", "datasets/NCBI-DC/pubannotation-dev", "http://pubannotation.org/docs/sourcedb/PubMed", "NCBI-Disease-dev", "PubMed", "MEDIC")
    #ncbi_disease_to_pubannotation("datasets/NCBI-DC/NCBItestset_corpus.txt", "datasets/NCBI-DC/pubannotation-test", "http://pubannotation.org/docs/sourcedb/PubMed", "NCBI-Disease-test", "PubMed", "MEDIC")
    # ddd_dataTrain = loader_one_ncbi_fold(["../NCBI/Voff/NCBItrainset_corpus.txt"])
    # dd_Train = extract_data(ddd_dataTrain, l_type=['CompositeMention', 'Modifier', 'SpecificDisease', 'DiseaseClass'])  # All entity types.
    # print("loaded.(Nb of mentions in train corpus =", len(dd_Train.keys()), ")")