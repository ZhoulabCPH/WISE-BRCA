# WISE-BRCA
Code for 'Artificial intelligence enables precision detection of pathogenic germline BRCA1/2 mutations in breast cancer from histology images: A multi-centre, retrospective study'
![flowchart](https://github.com/ZhoulabCPH/WISE-BRCA/blob/master/checkpoints/flowchart.png)
****
## Abstract
**Background** Genetic testing for pathogenic germline BRCA1/2 variants is essential for personalised management of high-risk breast cancers and guiding targeted therapies. We developed and tested a biologically interpretable model to detect these mutations from routine histology images.

**Methods** In this multi-centre, retrospective cohort study, we collected 2278 whole-slide images (WSIs) from 634 breast cancer patients. We developed a cross-attention, multi-scale, Transformer-based deep learning model, WISE-BRCA (Whole-slide Images Systematically Extrapolate BRCA1/2 mutations), to predict pathogenic germline BRCA1/2 mutations from WSIs obtained at Cancer Hospital, Chinese Academy of Medical Sciences (CHCAMS). We evaluated WISE-BRCA’s performance through within-cohort cross-validation in this cohort and externally validated the model in Yantai Yuhuangding Hospital (YYH) and Harbin Medical University Cancer Hospital (HMUCH) cohorts. BRCA1/2 mutation carriers and non-carriers were matched according to onset age, clinical stage, and molecular subtype (CHCAMS and YYH) or enrolled consecutively (HMUCH).

**Findings** WISE-BRCA demonstrated robust patient-level performance across cohorts, with an area under the curve (AUC)-receiver operating characteristic of 0·824 (95% CI 0·725–0·923; CHCAMS test set) and AUCs of 0·798 (95% CI 0·682–0·913; YYH) and 0·800 (95% CI 0·590–1·000; HMUCH). Quantitative analysis of WISE-BRCA’s prediction mechanisms revealed that breast cancers with germline BRCA1/2 mutations were associated with increased inflammatory cell infiltration, stromal proliferation and necrosis, and nuclear heterogeneity. A multi-modality prediction model integrating WSIs with clinicopathological information demonstrated superior performance, with AUC=0·925 (95% CI 0·868–0·982) in the unmatched external cohort, outperforming the single-modality prediction model.

