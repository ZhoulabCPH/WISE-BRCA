# WISE-BRCA [![DOI](https://zenodo.org/badge/850167237.svg)](https://doi.org/10.5281/zenodo.14058685)
Code for 'Artificial intelligence enables precision detection of pathogenic germline BRCA1/2 mutations in breast cancer from histology images: A multi-centre, retrospective study'
![flowchart](https://github.com/ZhoulabCPH/WISE-BRCA/blob/master/checkpoints/flowchart.png)
****
## Abstract
**Background** Genetic testing for pathogenic germline BRCA1/2 variants is essential for personalised management of high-risk breast cancers and guiding targeted therapies. We developed and tested a biologically interpretable model to detect these mutations from routine histology images.

**Methods** In this multi-centre, retrospective cohort study, we collected 2278 whole-slide images (WSIs) from 634 breast cancer patients. We developed a cross-attention, multi-scale, Transformer-based deep learning model, WISE-BRCA (Whole-slide Images Systematically Extrapolate BRCA1/2 mutations), to predict pathogenic germline BRCA1/2 mutations from WSIs obtained at Cancer Hospital, Chinese Academy of Medical Sciences (CHCAMS). We evaluated WISE-BRCA’s performance through within-cohort cross-validation in this cohort and externally validated the model in Yantai Yuhuangding Hospital (YYH) and Harbin Medical University Cancer Hospital (HMUCH) cohorts. BRCA1/2 mutation carriers and non-carriers were matched according to onset age, clinical stage, and molecular subtype (CHCAMS and YYH) or enrolled consecutively (HMUCH).

**Findings** WISE-BRCA demonstrated robust patient-level performance across cohorts, with an area under the curve (AUC)-receiver operating characteristic of 0·824 (95% CI 0·725–0·923; CHCAMS test set) and AUCs of 0·798 (95% CI 0·682–0·913; YYH) and 0·800 (95% CI 0·590–1·000; HMUCH). Quantitative analysis of WISE-BRCA’s prediction mechanisms revealed that breast cancers with germline BRCA1/2 mutations were associated with increased inflammatory cell infiltration, stromal proliferation and necrosis, and nuclear heterogeneity. A multi-modality prediction model integrating WSIs with clinicopathological information demonstrated superior performance, with AUC=0·925 (95% CI 0·868–0·982) in the unmatched external cohort, outperforming the single-modality prediction model.

****
## Dataset
- CHCAMS, Chinese Academy of Medical Sciences.
- YYH, Yantai Yuhuangding Hospital.
- HMUCH, Harbin Medical University Cancer Hospital.

  The datasets are available from the corresponding author upon reasonable request.

## checkpoints
- CTransPath: CTransPath model pretrained by [CTransPath](https://github.com/Xiyue-Wang/TransPath).
- Tumour_segmentation_model_224: Tumour segmentation model on patche of size 224.
- Tumour_segmentation_model_512: Tumour segmentation model on patche of size 512.
- WISE-BRCA: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations.
- WISE-BRCA-biopsy: Whole-slide Images Systematically Extrapolate BRCA1/2 mutations on biopsy samples.
- mcVAE: mcVAE is used to lean a joint common latent space of heterogeneous histopathological and phenotypic data.
- WISE-BRCA-combined: Joint prediction of BRCA1/2 mutation carriers from histology images and clinical information.
All checkpoints can be found at [WISE-BRCA](https://drive.google.com/drive/folders/1g4M8utv8-lPsp0yvJKDFEXheYQ6gPEti?usp=sharing).
## data_preprocessing
- <code>tiling_WSI_multi_thread.py</code>: Used to segment and filter patches from WSIs. Implemented based on <code>histolab</code> package.
- <code>stain_normalization_multi_thread.py</code>: Patches stain normalization. Implemented based on <code>ParamNet</code>.
- <code>cluster_sample.py</code>: A clustering-based sampling strategy implement to extract patches with distinct histomorphological features from the tumour area.

## tumour_segmentation
- <code>dataset.py</code>: Generate datasets.
- <code>model.py</code>: Implementation of tumour segmentation model.
- <code>train.py</code>: Training the tumour segmentation model.
- <code>inference_to_datasets.py</code>: Using tumour segmentation model to automatically extract tumour areas from each WSI.

## get_patches_feature
- <code>ctran.py</code>: Implementation of CTransPath.
- <code>get_CTransPath_feature.py</code>: Using pre-trained CTransPath to obtain histopathological features of patches.
  
  Part of the implementation here is based on [CTransPath](https://github.com/Xiyue-Wang/TransPath).

## WISE-BRCA
- <code>dataset.py</code>: Generate datasets.
- <code>model.py</code>: Implementation of WISE-BRCA.
- <code>train.py</code>: Training the WISE-BRCA model.
- <code>inference.py</code>: Predicting germline BRCA1/2 mutation status from histology images using WISE-BRCA.

## WISE-BRCA-combined
**mcVAE**
- <code>dataset.py</code>: Generate datasets.
- <code>model_mcVAE.py</code>: Implementation of mcVAE.
- <code>model_WISE-BRCA.py</code>: Implementation of WISE-BRCA.
- <code>train.py</code>: Training the WISE-BRCA model.
- <code>inference.py</code>: Predicting germline BRCA1/2 mutation status from histology images using WISE-BRCA.

- <code>model.py</code>: Implementation of WISE-BRCA-combined.
- <code>train.py</code>: Training the WISE-BRCA-combined model.
- <code>inference.py</code>: Predicting germline BRCA1/2 mutation status from histology images and clinical information using WISE-BRCA-combined.

## Usage
If you intend to utilize it for paper reproduction or your own WSI dataset, please adhere to the following workflow:
  1) Configuration Environment.
  2) Create a folder for your data and clinical information in <code>datasets</code> and download or move the WSIs there.
  3) Use <code>data_preprocessing/tiling_WSI_multi_thread.py</code> to segment WSIs into patches of size 224 and 512 at mpp of 0.488.
  4) Use <code>data_preprocessing/stain_normalization_multi_thread.py</code> to perform stain normalization for patches (If computing resources are limited, consider applying stain normalization only to patches sampled from the cluster sample).
  5) Use <code>get_patches_feature/get_CTransPath_feature.py</code> to obtain representation vector of patches.
  6) Use <code>tumour_segmentation/inference_to_datasets.py</code> to extract tumour areas from each WSI.
  7) Use <code>data_preprocessing/cluster_sample.py</code> clustering-based sampling strategy to extract patches with distinct histomorphological features from the tumour area.
  8) For the processing of clinical information, please refer to our previously published work 'DrABC: deep learning accurately predicts germline pathogenic mutation status in breast cancer patients based on phenotype data'.
  9) After preparation, use <code>WISE-BRCA/inference.py</code> or <code>WISE-BRCA-combined/inference.py</code> to predict germline BRCA1/2 mutation status on your own datasets. Or use <code>WISE-BRCA/train.py</code> or <code>WISE-BRCA-combined/train.py</code> on your own datasets.
  






  





  
