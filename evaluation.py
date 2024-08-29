
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################



######################################################################################################################
# Method
######################################################################################################################

def accuracy(ddd_normalized_corpus):
    totalScore = 0.0
    nbMentions = 0

    for doc_id in ddd_normalized_corpus.keys():  # [doc["sourceid"]][mention["id"]]["cui"]:  #  l_spacyNormalizedCorpus:
        for mention_id in ddd_normalized_corpus[doc_id]:
            score = 0.0
            nbMentions+=1
            if len(ddd_normalized_corpus[doc_id][mention_id]["pred_cui"]) > 0:
                for predCui in list(ddd_normalized_corpus[doc_id][mention_id]["pred_cui"]):
                    if predCui in ddd_normalized_corpus[doc_id][mention_id]["cui"]:
                        score += 1
                score = score / max(len(ddd_normalized_corpus[doc_id][mention_id]["cui"]), len(ddd_normalized_corpus[doc_id][mention_id]["pred_cui"]))

            totalScore += score  # Must be incremented even if no prediction

    totalScore = totalScore / nbMentions

    return totalScore

######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    pass
