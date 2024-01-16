
# Author: Louise Deléger & Arnaud Ferré
# Description: TRUE project

######################################################################################################################
# Import
######################################################################################################################



######################################################################################################################
# Method
######################################################################################################################

def accuracy(l_spacyNormalizedCorpus):
    totalScore = 0.0
    nbMentions = 0

    for doc in l_spacyNormalizedCorpus:
        for mention in doc.spans["mentions"]:
            score = 0.0
            nbMentions+=1
            if len(mention._.pred_kb_id_) > 0:
                for predCui in list(mention._.pred_kb_id_):
                    if predCui in mention._.kb_id_:
                        score += 1
                score = score / max(len(mention._.kb_id_), len(mention._.pred_kb_id_))

            totalScore += score  # Must be incremented even if no prediction

    totalScore = totalScore / nbMentions

    return totalScore

######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    pass
