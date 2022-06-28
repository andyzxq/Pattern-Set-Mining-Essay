#按照apriori的思想用FP-tree去寻找candidate的sup
#!/usr/bin/python
from collections import OrderedDict
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from treelib import Node,Tree

min_sup = 0.9
class fp_tree:
    def __init__(self, parent=None, item=None):
        self.item = item
        self.parent = parent
        self.count = 1
        self.child = []
        self.next = None

    def setChild(self, child):
        self.child.append(child)

    def addCount(self):
        self.count += 1

    def setNext(self, next):
        self.next = next

class head_pointer:
    def __init__(self, item):
        self.item = item
        self.next = None

    def setnext(self, next):
        self.next = next

def createHTable(fptree, order):
    HTable = []
    #先初始化头表
    for i in range(0, len(order)):
        HTable.append(head_pointer(order[i]))

    tree_node_l = []
    travelcontrol = [fptree]
    #广度优先遍历树
    while(len(travelcontrol)!=0):
        node = travelcontrol[0]
        travelcontrol.pop(0)
        for child in node.child:
            tree_node_l.append(child)
            travelcontrol.append(child)

    #构建表头
    for node in tree_node_l:
        ind = order.index(node.item)
        if HTable[ind].next == None:
            HTable[ind].next = node
        else:
            temp = HTable[ind].next
            while(temp.next != None):
                temp = temp.next
            temp.setNext(node)

    return HTable

def createItemOrder(transcations):
    order = []
    L = OrderedDict()
    for key in transcations.columns.values:
        temp = pd.DataFrame(transcations)[transcations[key] == True]
        if temp.shape[0]/transcations.shape[0] >= min_sup:
            L[key] = temp.shape[0]
    sorted_d = dict(sorted(L.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_d.keys():
        order.append(key)
    return order

def createFPtree(transcations, order):
    tlist = np.array(transcations)
    keylist = transcations.columns.values.tolist()
    root = fp_tree()
    #生成fp-tree
    for t in tlist:
        lyq = 0
        lyqlist = []
        lyqnode = root
        for i in range(0, len(order)):
            ind = keylist.index(order[i])
            if t[ind] == True:
                lyq += 1
                lyqlist.append(order[i])
                #顺利到达新增节点要到的层数的所在节点
                for j in range(0, lyq-1):
                    for child in lyqnode.child:
                        if child.item == lyqlist[j]:
                            lyqnode = child
                #插入节点
                flag = 0
                for child in lyqnode.child:
                    if child.item == order[i]:
                        #已有次数+1
                        child.addCount()
                        flag = 1
                        break
                if(flag==0):
                    #需要创建新节点
                    node = fp_tree(lyqnode, order[i])
                    lyqnode.setChild(node)

    """
    vnode = Node()
    vtree = Tree()
    showTree(vnode, vtree, root, "None", "")
    vtree.show()
    """
    return root

def showTree(vnode, vtree, tree, ptreeStr, sltr):
    sstr = sltr + "+" + str(tree.item)
    treeStr = "t"+","+str(tree.item)+":"+str(tree.count)+"t"+sstr
    if ptreeStr != "None":
        vtree.create_node(treeStr, treeStr, parent=ptreeStr)
    else:
        vtree.create_node(treeStr, treeStr)
    if len(tree.child) != 0:
        for child in tree.child:
            showTree(vnode,vtree,child,treeStr,sstr)
    return

#transcations 是Dataframes, 与mlxtend所需数据一致
def idea2(htable, tcount):
    maitList = []
    for i in range(0, len(htable)):
        ii = len(htable)-1-i
        node = htable[ii].next
        while(node != None):
            temp = node
            tempstr = ""
            while (temp != None):
                if temp.item != None:
                    tempstr = " " + str(temp.item) + tempstr
                else:
                    tempstr = " None" + tempstr
                temp = temp.parent
            tempstr = tempstr.replace(" None ", "")
            kl = tempstr.split(" ")
            #print(kl)
            count = getCount(kl, htable)
            #print(count)
            poplist = []
            if count / tcount >= min_sup:
                flag = 1
                for s in maitList:
                    l1 = s.split(" ")
                    l2 = tempstr.split(" ")
                    if set(l2) <= set(l1):
                        flag = 0
                        break
                    if set(l1) <= set(l2):
                        poplist.append(s)
                if flag == 1:
                    maitList.append(tempstr)
                if poplist != []:
                    for s in poplist:
                        maitList.remove(s)
            node = node.next
    #print(maitList)

def getCount(keyList , htable):
    dic = {}
    lyq = keyList[len(keyList)-1]
    for h in htable:
        if h.item == lyq:
            node = h.next
    #初始条件基
    while(node!=None):
        temp = node
        tempstr = ""
        while(temp!=None):
            if temp.item != None:
                tempstr = " " + str(temp.item) + tempstr
            else:
                tempstr = " None" + tempstr
            temp = temp.parent
        dic[tempstr] = node.count
        node = node.next
    #print(dic)
    #逐步筛选
    for i in range(0, len(keyList)-1):
        ii = len(keyList) - 2 - i
        tempdd = {}
        cc = 0
        for item in dic.items():
            nlyq = " " + str(keyList[ii])
            if nlyq in item[0]:
                tempdd[item[0][:item[0].index(nlyq)]+str(cc)] = item[1]
                cc += 1
        dic = tempdd
    #print(dic)
    lyqcount = 0
    for val in dic.values():
        lyqcount = lyqcount + val

    return lyqcount

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

    order = createItemOrder(df)
    tree = createFPtree(df, order)
    ht = createHTable(tree,order)

    """
    #展示ht是否正确
    for h in ht:
        print(h.item)
        node = h.next
        while(node!=None):
            print(node.item + str(node.count))
            node = node.next
    """

    idea2(ht, df.shape[0])