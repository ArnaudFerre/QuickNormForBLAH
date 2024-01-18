
# Author: Louise Deléger & Arnaud Ferré
# Description: QuickNorm

import json
import sys


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


######
# BB4 printers from PubAnnotation
######
def getA1File(JsonDocName):

    with open(JsonDocName, "r") as f:
        d_doc = json.load(f)

        # Only for discontinuous mentions:
        dd_discontinuous_mention_begin = dict()
        dd_discontinuous_mention_end = dict()
        dl_begin_and_end = dict()
        for item in d_doc["denotations"]:
            Tid = item["id"]
            if Tid.find('-') > 0:  # If it is a discontinuous mention
                T_number = int(Tid[1:Tid.find('-')])
                if T_number not in dd_discontinuous_mention_begin.keys():
                    dd_discontinuous_mention_begin[T_number] = dict()
                if T_number not in dd_discontinuous_mention_end.keys():
                    dd_discontinuous_mention_end[T_number] = dict()
                T_fragment_number = int(Tid[Tid.find('-') + 1:])
                dd_discontinuous_mention_begin[T_number][T_fragment_number] = item["span"]["begin"]
                dd_discontinuous_mention_end[T_number][T_fragment_number] = item["span"]["end"]
                dl_begin_and_end[T_number] = list()
        for T_nb in dd_discontinuous_mention_begin.keys():
            nb_fragments = len(dd_discontinuous_mention_begin.keys())
            for i in range(nb_fragments+1):
                dl_begin_and_end[T_number].append([str(dd_discontinuous_mention_begin[T_nb][i]), str(dd_discontinuous_mention_end[T_nb][i])])

        d_a1_lines = dict()
        for item in d_doc["denotations"]:
            Tid = item["id"]

            if Tid.find('-') > 0:  # If it is a discontinuous mention
                if item["obj"] == "_FRAGMENT":
                    pass
                else:
                    T_number = int(Tid[1:Tid.find('-')])
                    l_offset = list()
                    l_fragments = list()
                    for elt in dl_begin_and_end[T_number]:
                        l_offset.append(" ".join(elt))
                        l_fragments.append(d_doc["text"][int(elt[0]):int(elt[1])])
                    offset = ";".join(l_offset)
                    mention = " ".join(l_fragments)
                    d_a1_lines[T_number] = "T"+str(T_number)+"\t"+item["obj"]+" "+offset+"\t"+mention+"\n"

            else:  # If it is a continuous mention (no '-', so = -1)
                T_number = int(Tid[1:])
                begin = item["span"]["begin"]
                end = item["span"]["end"]
                d_a1_lines[T_number] = "T" + str(T_number) + "\t" + item["obj"] + " " +str(begin)+" "+str(end)+"\t" + d_doc["text"][begin:end]+"\n"

        a1Content = ""
        for i in range(1, len(d_a1_lines.keys())+1):
            a1Content += d_a1_lines[i]

    return a1Content


def writeA1file(a1Filename, content):
    with open(a1Filename, "w") as f:
        f.write(content)


def writeA1batch():
    pass



def getA2File(JsonDocName):
    with open(JsonDocName, "r") as f:
        d_doc = json.load(f)

        d_a2_lines = dict()
        for annotation in d_doc["attributes"]:
            N_number = int(annotation["id"][1:])
            if annotation["subj"].find('-') > 0:
                T_number = annotation["subj"][0:annotation["subj"].find("-")]
            else:
                T_number = annotation["subj"]
            d_a2_lines[N_number] = "N"+str(N_number)+"\t"+annotation["pred"]+" Annotation:"+T_number+" Referent:"+annotation["obj"]+"\n"

        a2Content = ""
        for i in range(1, len(d_a2_lines.keys())+1):
            a2Content += d_a2_lines[i]

    return a2Content

def writeA2file(a2Filename, content):
    with open(a2Filename, "w") as f:
        f.write(content)


def spacy_into_a2(l_spacy_corpus, save_file=False, save_path=None, pred=True, align_type_onto=None):

    nb_mentions = 0
    for doc in l_spacy_corpus:
        docName = doc.user_data["document_id"]

        d_mentions = dict()
        for mention in doc.spans["mentions"]:
            T_id = mention.id_
            T_number = int(T_id[1:])
            d_mentions[T_number] = dict()
            #d_mentions[T_number]["surface"] = mention.text
            if pred == True:
                d_mentions[T_number]["kb_id"] = mention._.pred_kb_id_
            else:
                d_mentions[T_number]["kb_id"] = mention._.kb_id_
            d_mentions[T_number]["kb_name"] = align_type_onto[mention.label_]

        tried_keys = sorted(d_mentions.keys())
        a2content = ""
        N_number = 1
        for T_number in tried_keys:
            s_cui = d_mentions[T_number]["kb_id"]
            for cui in s_cui:
                a2content += "N"+str(N_number)+"\t"+d_mentions[T_number]["kb_name"]+" Annotation:T"+str(T_number)+" Referent:"+cui+"\n"
                N_number += 1
                nb_mentions += 1


        if save_file == True:
            if save_path is None:
                save_path = "./"
            with open(save_path+docName+".a2", 'w') as fp:
                fp.write(a2content)
                
    print("\nNumber of mentions:", nb_mentions)




######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    """
    from loaders import pubannotation_to_spacy_corpus
    import spacy

    nlp = spacy.load("en_core_web_sm")
    l_spacy_BB4_hab_train = pubannotation_to_spacy_corpus("datasets/BB4/bionlp-ost-19-BB-norm-train/", l_type=["Habitat", "Microorganism", "Phenotype"], spacyNlp=nlp)

    spacy_into_a2(l_spacy_BB4_hab_train, save_file=True, save_path="datasets/BB4/predictions/", pred=False, align_type_onto={"Habitat": "OntoBiotope", "Microorganism": "NCBI_Taxonomy", "Phenotype": "OntoBiotope"})
    """
    # ToDo: add at the end of QuickNorm



    sys.exit(0)



    content = getA2File("datasets/BB4/bionlp-ost-19-BB-norm-train/BB-norm-448557_annotations.json")
    writeA2file("datasets/BB4/predictions/BB-norm-448557_annotations.a2", content)

    content = getA1File("datasets/BB4/bionlp-ost-19-BB-norm-train/BB-norm-448557_annotations.json")
    writeA1file("datasets/BB4/predictions/BB-norm-448557_annotations.a1", content)

        