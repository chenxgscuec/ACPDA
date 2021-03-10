# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:36:26 2021

@author: sxg
"""

def processACPlen(file,file_new,peptide_len):

    label = []
    protein_seq_new = []
    len_array = []
    with open(file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                label.append(label_temp)
#                proteinName = values[0]
#                if label_temp == '1':
#                    label.append(1)
#                else:
#                    label.append(0)
            else:
                seq = line[:-1]
                len_pc = len(seq)
                len_array.append(len_pc)   
                if len_pc < peptide_len:
                    len_add = peptide_len - len_pc
                    seq_new = seq + "X"*len_add
                else:
                    seq_new = seq[0:peptide_len]
                protein_seq_new.append(seq_new)
                
    with open(file_new,'w+') as f1:
        for i in range(len(protein_seq_new)):
            f1.write(">"+label[i]+"\n")
            f1.write(protein_seq_new[i]+"\n")

file = 'acp240.txt'
file_new = 'acp240_40.txt'  
peptide_len = 40        
processACPlen(file,file_new,peptide_len)    