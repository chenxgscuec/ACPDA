# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:14:08 2021

@author: sxg
"""
import numpy as np
import pandas as pd
import pymrmr
import pickle

df = pd.read_csv('AAINDEX_ACP240_50.csv',encoding="utf-8")   
print(1)    
feats = pymrmr.mRMR(df, 'MIQ', 50)
X = np.array(df[feats])
print(2)
label = np.array(df["class"])
data = {}
data['X'] = X
data['feats'] = feats
data['y'] = label
save_path = 'data240_50_50.pkl'
pickle.dump(data,open(save_path, 'wb'))