#!/usr/bin/python
from collections import OrderedDict
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

# Generate Ck based on Lk-1
def apriori_gen(Lkminus1):
    Ck = OrderedDict()
    lkeys = list(Lkminus1.keys())
    for i in range(0, len(lkeys)):
        l1 = lkeys[i]
        for j in range(i+1, len(lkeys)):
            l2 = lkeys[j]

            pos1 = l1.rfind(" ")
            pos2 = l2.rfind(" ")

            if pos1 == -1 and pos2 == -1:
                Ck[ l1 + " " + l2 ] = 0
            else:
                if l1[0:pos1] == l2[0:pos2]:
                    c = l1 + " " + l2[pos2+1:]
                    if not has_infrequent_subset(c, Lkminus1):
                        Ck[c] = 0

    return Ck

# Check whether a candidate k-itemset has an infrequent (k-1)-itemset
def has_infrequent_subset(c, Lkminus1):
    fields = c.split(" ")

    for i in range( 0, len(fields)-1 ):
        subset = ""
        for j in range( 0, len(fields)-1 ):
            if i != j:
                subset += fields[j] + " "
        subset += fields[ len(fields) - 1 ]
        #print(c+"    :      "+subset)
        if subset not in Lkminus1:
            return True
    return False

# 判断K项的sup是否与K-1项相等
def has_equal_subset(c, sup, Lkminus1):
    fields = c.split(" ")
    for i in range( 0, len(fields)-1 ):
        subset = ""
        for j in range( 0, len(fields)-1 ):
            if i != j:
                subset += fields[j] + " "
        subset += fields[ len(fields) - 1 ]
        if sup >= Lkminus1[subset]:
            return True
    return False

#transcations 是Dataframes, 与mlxtend所需数据一致
def apriori_closed(transcations):
    genreqItemset = OrderedDict()

    # determine min_sup
    min_sup = 0.9

    # determine L1
    L = OrderedDict()
    for i in range(0, transcations.shape[1]):
        npls = np.array(transcations)
        if npls[:,i].tolist().count(True)/npls[:,0].size >= min_sup:
            L[transcations.columns.values[i]] = npls[:,i].tolist().count(True)/npls[:,i].size
    genreqItemset.update(L)
    #print(freqItemset)

    # determine L2, L3, ...
    while len(L) != 0:
        Ck = apriori_gen(L)
        if len(Ck) == 0:
            break

        #计算K项的sup值
        for key in Ck.keys():
            kyli = key.split(" ")
            temp_t = transcations
            #print(key)
            for i in range(0, len(kyli)):
                temp_t = pd.DataFrame(temp_t)[temp_t[kyli[i]] == True]
            Ck[key] = temp_t.shape[0]

        old_L = L
        L = OrderedDict()
        for key in Ck.keys():
            if len(key.split(" ")) > 2:
                if Ck[key] / npls[:, 0].size >= min_sup:
                    if not has_equal_subset(key, Ck[key] / npls[:, 0].size, old_L):
                        L[key] = Ck[key] / npls[:, 0].size
            else:
                if Ck[key] / npls[:, 0].size >= min_sup:
                    if Ck[key] / npls[:, 0].size < old_L[key.split(" ")[0]] and Ck[key] / npls[:, 0].size < old_L[key.split(" ")[1]]:
                        L[key] = Ck[key] / npls[:, 0].size
        genreqItemset.update(L)

    #print(genreqItemset)

    closure = []
    #根据generator获得closure
    for key in genreqItemset:
        kyli = key.split(" ")
        temp_t = transcations
        for i in range(0, len(kyli)):
            temp_t = pd.DataFrame(temp_t)[temp_t[kyli[i]] == True]
        k = ""
        for item in temp_t.columns.values:
            temp = pd.DataFrame(temp_t)[temp_t[item] == True]
            if temp.shape[0] == temp_t.shape[0]:
                k += item + " "
        if k not in closure and k != "":
            closure.append(k)
    #print(closure)

    maximal_Itemsets = []
    #判断是否一个closure是其他元素的子集
    for i in range(0, len(closure)):
        flag = 0
        for j in range (0, len(closure)):
            l1 = closure[i].split(" ")
            l2 = closure[j].split(" ")
            if i != j and set(l1) <= set(l2):
                flag = 1
        if flag == 0 :
            maximal_Itemsets.append(closure[i])

    #print(maximal_Itemsets)
    #print("maximal itemset: " + str(len(maximal_Itemsets)))

    return maximal_Itemsets
    #print(closurefreqItemset)

if __name__ == '__main__':
    info = [['Apple', 'Beer', 'Rice', 'Chicken'],
     ['Apple', 'Beer', 'Rice'],
     ['Apple', 'Beer'],
     ['Apple', 'Bananas'],
     ['Milk', 'Beer', 'Rice', 'Chicken'],
     ['Milk', 'Beer', 'Rice'],
     ['Milk', 'Beer'],
     ['Apple', 'Bananas']]

    TE = TransactionEncoder()
    datas = TE.fit_transform(info)

    df = pd.DataFrame(datas, columns=TE.columns_)
    #print(df)

    apriori_closed(df)

