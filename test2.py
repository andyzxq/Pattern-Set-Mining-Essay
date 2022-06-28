import AprioriClosed as a
import ExAnte as ex
import numpy as np
import efficient_apriori as ea
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder

dict = {'?':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'g':8, 'h':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'r':16, 's':17, 't':18,
 'u':19, 'v':20, 'w':21, 'x':22, 'y':23}

def DataCleaning():
    file = "data\\agaricus-lepiota.data"
    pd_data = pd.read_table(file, header=None)
    #print(pd_data)
    data = []
    for i in range(0, len(pd_data)):
        item = pd_data.iat[i,0].split(",")
        data.append(item)
    print("finish data reading1")
    return data

def getSubset(a):
    inputList = a.split(" ")
    output = []
    lyq = pow(2,len(inputList))
    for i in range(0, lyq):
        ii = i
        temp = []
        count = 0
        while (ii != 0):
            if(ii%2==1):
                temp.append(inputList[count])
            ii = int(ii / 2)
            count += 1
        if temp != []:
            output.append(temp)
    return output

if __name__ == '__main__':
    Cm = 280
    min_Sup = 0.9

    data = DataCleaning()

    dataint = []
    for d in data:
        temp = []
        for item in d:
            temp.append(str(dict[item]))
        dataint.append(temp)

    dataintC = []
    for d in dataint:
        count = 0
        for item in d:
            count = count + int(item)
        dataintC.append(count)

    print("start")

    #Apriori-closed + filter 所用时间
    start1 = time.time()
    dataIntMC = []
    for data in dataint:
        count = 0
        for item in data:
            count = count + int(item)
        if count > Cm:
            dataIntMC.append(data)
    TE = TransactionEncoder()
    datas = TE.fit_transform(dataIntMC)
    df = pd.DataFrame(datas, columns=TE.columns_)
    print("数据量1: " + str(df.shape[0]))
    print("项目量1: " + str(df.shape[1]))
    mis = a.apriori_closed(df)
    out = []
    end1 = time.time()
    print("time for apriori-closed + filter:" + str(end1-start1))


    #ExAnte + Apriori-closed 所用时间
    start2 = time.time()
    dataByExAnte = ex.ExAnte(dataint, Cm, min_Sup, len(dict.keys()))
    TE = TransactionEncoder()
    datass = TE.fit_transform(dataByExAnte)
    df2 = pd.DataFrame(datass, columns=TE.columns_)
    print("数据量2: " + str(df2.shape[0]))
    print("项目量2: " + str(df2.shape[1]))
    mis2 = a.apriori_closed(df2)
    out2 = []
    for mi in mis2:
        fql = getSubset(mi[:len(mi)-1])
        for fq in fql:
            out2.append(fq)
    end2 = time.time()
    print("time for ExAnet + apriori-closed:" + str(end2 - start2))