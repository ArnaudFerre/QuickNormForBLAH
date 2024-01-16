
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

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
