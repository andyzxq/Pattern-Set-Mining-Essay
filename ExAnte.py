#!/usr/bin/python
from collections import OrderedDict
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori

def ExAnte(df, Cm, min_supp, item_num):
    I = []
    itemDict = {}
    data = df
    for t in data:
        #判断是否符合Cm
        Ct = 0
        for item in t:
            Ct = Ct + int(item)
        if Ct > Cm:
            for item in t:
                if item in itemDict.keys():
                    itemDict[item] = itemDict[item] + 1
                else:
                    itemDict[item] = 1
    #print(itemDict)
    #将符合min_sup的items加入列表
    for item in itemDict.keys():
        if itemDict[item] / len(data) >= min_supp:
            I.append(item)
    #print(I)
    old_number_interesting_items = item_num
    newdata = data
    while(len(I) < old_number_interesting_items):
        newdata = alpha_reduction(data, I)
        #print(newdata)
        newdata = miu_reduction(newdata, Cm)
        #print(newdata)
        old_number_interesting_items = len(I)
        I=[]
        #更新I
        itemDict = {}
        for t in newdata:
            for item in t:
                if item in itemDict.keys():
                    itemDict[item] = itemDict[item] + 1
                else:
                    itemDict[item] = 1
        for item in itemDict.keys():
            if itemDict[item] / len(newdata) >= min_supp:
                I.append(item)
    return newdata

def alpha_reduction(data, I):
    newdata = []
    for t in data:
        temp = []
        for item in t:
            if item in I:
                temp.append(item)
        newdata.append(temp)
    return newdata

def miu_reduction(data, Cm):
    newdata = []
    for t in data:
        Ct = 0
        for item in t:
            Ct = Ct + int(item)
        if Ct > Cm:
            newdata.append(t)
    return newdata


if __name__ == '__main__':
    info = [['1', '2', '5', '6'],
     ['1', '2', '5'],
     ['1', '2'],
     ['1', '4'],
     ['3', '2', '5', '6'],
     ['3', '2', '5'],
     ['3', '2'],
     ['1', '4']]

    TE = TransactionEncoder()
    datas = TE.fit_transform(info)

    df = pd.DataFrame(datas, columns=TE.columns_)
    data = ExAnte(info, 5, 0.5, df.shape[1])
    print(data)

    print(apriori(df, min_support=0.5))

    #print(df)