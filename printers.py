
# Author: Louise Deléger & Arnaud Ferré
# Description: QuickNorm

import json

def print_hf_doc(hf_doc):
    for key in hf_doc.keys():
        print(key, ":", hf_doc[key])


def print_bb4_hf_mentions(hf_doc, nbExamples=None):
    if len(hf_doc["text_bound_annotations"]) > 0:  # avoid void doc
        for i, mention in enumerate(hf_doc["text_bound_annotations"]):
            if nbExamples is not None:
                if i >= nbExamples:
                    break
            if len(hf_doc["normalizations"]) > 0:
                for annotation in hf_doc["normalizations"]:
                    if annotation["ref_id"] == mention["id"]:
                        print(mention, "\nGold:", hf_doc["normalizations"][i], "\n")
            else:
                print(mention)


def print_hf_bigbio_mentions(hf_bigbio_doc, nbExamples=None):
    if len(hf_bigbio_doc["entities"]) > 0:
        for i, mention in enumerate(hf_bigbio_doc["entities"]):
            if nbExamples is not None:
                if i >= nbExamples:
                    break
            if len(mention["normalized"]) > 0:
                print(mention, "\nGold:", mention["normalized"], "\n")
            else:
                print(mention)

                
def print_pubannotation(spacy_norm_corpus, output_folder, project_uri, project_name, sourcedb, onto_name):
    
    for doc in spacy_norm_corpus:
        puba_doc = dict()
        puba_doc['target'] = project_uri + "/sourceid/" + doc.user_data["document_id"]
        puba_doc['sourcedb'] = sourcedb
        puba_doc['sourceid'] = doc.user_data["document_id"]
        puba_doc['text'] = doc.text
        puba_doc['project'] = project_name
        puba_doc['denotations'] = []
        puba_doc['attributes'] = []
        count_norm = 1
        for mention in doc.spans["mentions"]:
            mention_dict = dict()
            mention_dict['id'] = mention.id_
            mention_dict['obj'] = mention.label_
            mention_dict['span'] = dict()
            mention_dict['span']['begin'] = mention.start_char
            mention_dict['span']['end'] = mention.end_char
            puba_doc['denotations'].append(mention_dict)
            if (len(mention._.pred_kb_id_) > 0):
                for predCui in list(mention._.pred_kb_id_):
                    pred_dict = dict()
                    pred_dict['pred'] = onto_name
                    pred_dict['id'] = "A" + str(count_norm)
                    pred_dict['subj'] = mention.id_
                    pred_dict['obj'] = predCui
                    puba_doc['attributes'].append(pred_dict)
                    count_norm += 1
            elif (len(mention._.kb_id_) > 0):
                for predCui in list(mention._.kb_id_):
                    pred_dict = dict()
                    pred_dict['pred'] = onto_name
                    pred_dict['id'] = "A" + str(count_norm)
                    pred_dict['subj'] = mention.id_
                    pred_dict['obj'] = predCui
                    puba_doc['attributes'].append(pred_dict)
                    count_norm += 1

        with open(output_folder+"/"+doc.user_data["document_id"]+'.json', 'w') as fp:
            json.dump(puba_doc, fp)
        