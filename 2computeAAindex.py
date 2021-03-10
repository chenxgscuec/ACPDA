import re
from codes import *
import csv

file = "acp240_60.txt"

filePath=''
trainFile=''
labelFile=''
myOrder=''

type_choices=['AAINDEX']

# Calculation features
def cal_feature(outfile_name):
    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    myOrder = 'ACDEFGHIKLMNPQRSTVWY'
    kw = {'path': filePath, 'train': trainFile, 'label': labelFile, 'order': myOrder}
    myFun = type_choices + '.' + type_choices + '(myFasta,)'
    # print('Descriptor type: ' + type_choices)
    encodings = eval(myFun)

    outFile = outfile_name if outfile_name != None else 'encoding.tsv'
    saveCode.savetsv(encodings, outFile)
    """
    txt Convert csv
    """
    list_dp = []
    with open(outfile_name) as f:
        # list_add = [ i for i in ]
        for line in f:
            list = line.strip('\n').split('\t')
            list_dp.append(list)
    with open(outfile_name,"w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list_dp)



# Delete the tag column of a file
def del_firstcol(outpath):
    df = pd.read_csv(outpath)
    df = df.drop(["class"],axis=1)
    df.to_csv(outpath, index=0)


for i in type_choices:
    outfile_name = str(i)+'acp240_60.csv'
    print(outfile_name)
    type_choices = i
    try:
        cal_feature(outfile_name)
#        del_firstcol(outfile_name)
    except:
        print(outfile_name+"，文件有错误")
    # print(i)

