import AprioriClosed as a
import idea1 as b
import idea2 as c
import numpy as np
import efficient_apriori as ea
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax

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

if __name__ == '__main__':
    data = DataCleaning()
    TE = TransactionEncoder()
    datas = TE.fit_transform(data)
    df = pd.DataFrame(datas, columns=TE.columns_)
    print(df.columns.values)
    print(df)

    print("start")

    frequent_item, rules = ea.apriori(data, min_support=0.9, min_confidence=1)
    fi = {}
    for k in frequent_item:
        fi.update(frequent_item[k])
    #print(fi)
    print("frequent itemsets: " + str(len(fi)))
    r = fpmax(df, min_support=0.9)
    print("maximal itemsets: "+str(r.shape[0]))

    start1 = time.time()
    a.apriori_closed(df)
    end1 = time.time()
    print("time for apriori-closed:" + str(end1-start1))

    start2 = time.time()
    order = b.createItemOrder(df)
    tree = b.createFPtree(df, order)
    ht = b.createHTable(tree, order)
    b.idea1(order, ht, df.shape[0], df)
    end2 = time.time()
    print("time for idea1:" + str(end2 - start2))

    start3 = time.time()
    order = c.createItemOrder(df)
    tree = c.createFPtree(df, order)
    ht = c.createHTable(tree, order)
    c.idea2(ht, df.shape[0])
    end3 = time.time()
    print("time for idea2:" + str(end3 - start3))

    start4 = time.time()
    fpmax(df, min_support=0.5)
    end4 = time.time()
    print("time for fpmax:" + str(end4 - start4))

