# QuickNormForBLAH
# Fast training and prediction for biomedical entity linking: integrating QuickNorm, a lightweight LM-based method, into PubAnnotation

## Motivation
In recent years, Language Models (LMs) have emerged as a transformative force in NLP, pushing the boundaries of what machines can accomplish in most NLP tasks, such as information extraction. Large Language Models (LLMs), with their now billions of parameters, have transcended the capabilities of their predecessors, empowering them to comprehend and generate human-like text with remarkable fluency and context-awareness. However, this exponential growth in model size has not come without its share of challenges and concerns. First and foremost, the scale of LLM introduces significant computational costs. Training and fine-tuning these massive models require extensive computing resources, including specialized hardware like powerful GPUs. This substantial infrastructure investment can be a barrier to entry for many researchers and organizations. Furthermore, the environmental impact of training and running large-scale LLM is a growing concern. The energy consumption associated with training deep learning models at this scale can be substantial, contributing to carbon emissions and environmental degradation. As such, researchers and developers are under increasing pressure to explore energy-efficient alternatives in the development of NLP methods. During this 8th BLAH edition, we choose to address this concern with a focus on the biomedical entity linking (BEL) task (also commonly known as concept/entity normalization). BEL plays a key role in biomedical text-mining by ensuring that various biomedical mentions in unstructured text are linked to their standardized entities from domain-specific knowledge bases. This task is pivotal in extracting structured knowledge from the vast and growing corpus of biomedical literature, enabling researchers and healthcare professionals to access relevant information quickly and accurately.

## Objectives
We propose to perfect and finalize QuickNorm, a lightweight LM-based method designed to streamline the training and prediction process for BEL, which is not based on LLM, but rather on SLM (for Small Language Model). QuickNorm leverages the accuracy of LM while mitigating their computational demands. We propose to integrate QuickNorm into PubAnnotation, a popular platform for collaborative biomedical text annotation and curation. By combining the agility of QuickNorm with the collaborative capabilities of the PubAnnotation platform, we aim to provide a practical and accessible tool for BEL, advancing the field of biomedical informatics and facilitating breakthroughs in biomedicine. We also propose to integrate new BEL gold standards into PubAnnotation to enable better evaluation of BEL methods: the NCBI Disease Corpus (Doğan et al., 2014) and the BioCreative V BC5CDR corpus (Li et al., 2016). This will allow us to (quickly) experiment and evaluate the robustness of QuickNorm on these well-known datasets, as well as on the already integrated Bacteria Biotope 2019 dataset (Bossy et al., 2019). And if time permits, we will conduct new experiments, e.g. by using LLMs instead of an SLM, and initiate a comparative study, particularly regarding computation times and RAM consumption, with a few other state-of-the-art methods (e.g. BioSyn, Sung et al., 2020).

## The QuickNorm method
QuickNorm is designed to be fast and easy-to-use, while retaining good performance. A version of the QuickNorm method already exists and is based on existing techniques, bringing them together in a simple and efficient way. It relies on optimization tricks, on deep learning and preprocessing techniques and on the use of a SLM. An originality of the method is to address the BEL task as a regression problem instead of a classification one. The optimization tricks include for instance using a C language function to perform resource-intensive distance calculations, and a deep learning technique only to unit-normalize the input. Figure 1a and 1b outline the main steps of the method for both training and prediction phases. QuickNorm has already been tested on a gold standard dataset (Bacteria Biotope), which is readily available on PubAnnotation. It achieved results that were less than 3 points in accuracy below the state-of-the-art method (C-Norm), positioning it in second place ahead of more complex methods. Furthermore, it exhibits remarkable efficiency by running on a standard laptop CPU in less than 6 minutes and with a relatively lower RAM consumption. In comparison, C-Norm requires several hours of computation on a professional CPU server and consumes several tens of GB of RAM. To address the readiness for integration into PubAnnotation, we are committed to preparing QuickNorm for seamless inclusion. This will involve creating clear documentation, cleaning up the code and ensuring user-friendliness. Our team is dedicated to making these code improvements even before the hackathon, ensuring that QuickNorm is well-prepared for seamless integration into PubAnnotation.

<figure>
    <img src="figures/figure1a.emf" alt="figure 1a: Training steps of the QuickNorm method">
    <figcaption>Figure 1a: Training steps of the QuickNorm method. Initially, all concept label vectors are calculated using the initial SLM and remain fixed during the first phase of training. In contrast, during this initial phase, mention vectors are modified at each epoch through fine-tuning of the SLM. A second training phase is then conducted, where label vectors are recalculated using the final finetuned SLM from the first phase. Finally, the regression parameters are reset, and a new training phase is initiated</figcaption>
</figure>

## BLAH8 tasks
- Packaging and integration of QuickNorm into PubAnnotation. This means cleaning up the code and creating parsers so that the method can support the PubAnnotation format;
- Integration of EL datasets into PubAnnotation: parsing and formatting the NCBI-Disease and BC5CDR corpora (others may be considered if time permits);
- Evaluation of the method on the PubAnnotation datasets;
- Experiments with different and larger LMs;
- Comparison with SOTA approaches (BioSyn).

## Team
Arnaud Ferré, Université Paris-Saclay, MaIAGE, INRAE (France) 

Louise Deléger, Université Paris-Saclay, MaIAGE, INRAE (France)

## References
Bossy R, Deléger L, Chaix E, Ba M, Nédellec C. Bacteria biotope at BioNLP open shared tasks 2019. In Proceedings of the 5th workshop on BioNLP open shared tasks 2019 Nov (pp. 121-131).  

Doğan RI, Leaman R, Lu Z. NCBI disease corpus: a resource for disease name recognition and concept normalization. Journal of biomedical informatics. 2014 Feb 1;47:1-0.  

Li J, Sun Y, Johnson RJ, Sciaky D, Wei CH, Leaman R, Davis AP, Mattingly CJ, Wiegers TC, Lu Z. BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database. 2016 Jan 1;2016.  

Sung M, Jeon H, Lee J, Kang J. Biomedical entity representations with synonym marginalization.  In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020 (pp 3641–3650). Association for Computational Linguistics.  

