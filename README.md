ACP-DA
===============================
An implementation of ACP-DA, a new model for predicting anticancer peptides.

Reference
========================
Our manuscipt titled with "ACP-DA: Improving the prediction of anticancer peptides using data augmentation" is being reviewed by Frontiers in Genetics.

Requirements
========================
    [python 3.6](https://www.python.org/downloads/)
    [iFeature](https://github.com/Superzchen/iFeature)
    [pymrmr](https://github.com/fbrundu/pymrmr)


Usage
========================
if you want to run the prediction of anticancer peptides without data augmentation, you can run:
python 4ACP.py

if you want to run the prediction of anticancer peptides with data augmentation, you can run:
python 5ACPaugment.py

1processACPlen.py is used for converting sequences of unequal length to sequences of equal length.

2computeAAindex.py is used for AAindex feature calculation, which needs to be run in the directory where iFeature is located.

3featureSelect.py is uesd for feature selection.


Data
=====================
In this work, we use the datasets and part code organized by ACP-DL(https://github.com/haichengyi/ACP-DL).




Contact
=====================
Author: Xian-gan Chen
Maintainer: Xian-gan Chen
Mail: chenxg@mail.scuec.edu.cn
Date: 2021-4-16
School of Biomedical Engineering, South-Central University for Nationalities, China
