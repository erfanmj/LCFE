#!/usr/bin/env python
# coding: utf-8

# In[1]:


################################################### ImportLibraries #######################################################
import pandas as pd
import networkx as nx
import numpy as np
import time
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import statistics 


# In[34]:


################################################## Read&MakeNetwork #######################################################

#Read From Data Source 
data = pd.read_excel("D:/1- IUST/3- Article/4- Benchmarks/3- For parameter testing/1/1.xlsx", header=None)


#Convert the data to list of edges 
edgeList = data.values.tolist()

#Creating an empty networkx directed Graph
Graph = nx.DiGraph()

#Inserting the edges into the empty Graph: 
for i in range(len(edgeList)):
    Graph.add_edge(int(edgeList[i][0]), int(edgeList[i][1]), weight=edgeList[i][2])
    
#Calculating edge centrality and adding it to graph
eb = nx.edge_betweenness_centrality(Graph, normalized=False)
nx.set_edge_attributes(Graph, eb, 'betweenness')

#test the added attribute using following code
#nx.get_edge_attributes(Graph,name = 'betweenness')

# Setting Edge IDs
mynum = len(Graph.edges())
EdgeIds = eb
for i in EdgeIds.keys():
    EdgeIds[i] = mynum
    mynum = mynum - 1
    if mynum < 1:
        break
nx.set_edge_attributes(Graph,EdgeIds,'EdgeID')


#######################################################Reading Community of benchmark######################################

df = pd.read_excel("D:/1- IUST/3- Article/4- Benchmarks/3- For parameter testing/1/1.xlsx", header=None)
d1 = df.groupby(1)[0].agg(list).to_dict()
res = [{k:v} for k,v in d1.items()]
toCompare = []
for j in res: 
    toCompare.append(list(j.values())[0])
    
    
#######################################################Read OSLOM Partitions################################################

# # if its an csv you can define sep=' '
# df = pd.read_excel("D:/1- IUST/3- Article/4- Benchmarks/1- For Accuracy/Group 4/16/16o.xlsx", header=None)
# # name the columns to filter results
# df.columns = ['temp']
# # search for rows where "module" exists in string and get the opposite (the ~ before)
# df = df[~df['temp'].str.contains('module')].reset_index()
# # split() the values of the column expanding into new ones
# df = df['temp'].str.split(" ", expand=True)
# # transform into list
# list_values = df.values.tolist()
# # Filter Nones
# filtered_list = [list(filter(None, l)) for l in list_values]

# OSLOM = []
# for element in filtered_list:
#     tempList = []
#     for item in element: 
#         tempList.append(int(item))
#     OSLOM.append(tempList)
#     tempList = []
    

##################################################### FUNCTIONS ##########################################################

# Function to calculate Weighted Flow Measure: 
def WFM(g,subg,added):
    #mInt
    mInt = 0
    inerEdges = []
    for i in sorted(subg.edges(data=True), key = lambda x: x[2]['betweenness']):
        mInt = mInt + i[2]['weight']
        inerEdges.append(i[2]['EdgeID'])
    
    #mExt
    mExt = 0
    for i in sorted(g.edges(data=True), key = lambda x: x[2]['betweenness']):
        if i[2]['EdgeID'] in inerEdges:
            continue
        for k in g.nodes():
            if i[0] == k and i[1] != k:
                mExt = mExt + i[2]['weight']
            if i[0] != k and i[1] == k:
                mExt = mExt + i[2]['weight']
    
    #first Part of WFM:
    if mExt == 0: 
        mExt = 1
    firstPart = mInt / mExt
    
    #second part of wfm
    secondPart = 0
    try:
        if added == None: 
            cycle = nx.find_cycles(subg,orientation='original')
        else: 
            cycle = nx.find_cycles(subg,source=added,orientation='original')
        for h in cycle: 
            if int(h[0]) != int(added) or int(h[1]) != int(added):
                cycle = None
                break
    except: 
        cycle = None
    if cycle != None:
        for s in sorted(subg.edges(data=True), key = lambda x: x[2]['betweenness']):
            for t in cycle: 
                if t[0] == s[0] and t[1] == s[1]:
                    secondpart = secondpart + s[2]['weight']
                    break
    else:
        secondpart = 0
    return(firstPart + secondpart)

# function for calculating LCtemp (a part of LC)
def LCtemp(gg,ee,bb,res):
    LCtemp1 = []
    com0 = []
    com1 = []
    if len(bb) == 0: 
        CN = nx.common_neighbors(nx.Graph(gg),ee[0],ee[1])
        CNL = copy.deepcopy(list(CN))
        added = None
        for j in CNL:
            LCtemp1.append(ee[0])
            LCtemp1.append(ee[1])
            WFM1 = WFM(gg,nx.subgraph(gg,LCtemp1),added)
            LCtemp1.append(j)
            WFM2 = WFM(gg,nx.subgraph(gg,LCtemp1),j)
            if WFM2 <= WFM1: 
                LCtemp1.remove_node(j)
            added = j
        if len(LCtemp1) > res:
            return(nx.subgraph(gg,LCtemp1))
        else: 
            return(0)
    elif len(bb) > 0:
        for u in bb: 
            if ee[0] in u.nodes():
                com0.append(u)
            if ee[1] in u.nodes(): 
                com1.append(u)
        if len([x for x in com0 if x in com1]) == 0: 
            CN = nx.common_neighbors(nx.Graph(gg),ee[0],ee[1])
            CNL = copy.deepcopy(list(CN))
            added = None
            for j in CNL:
                LCtemp1.append(ee[0])
                LCtemp1.append(ee[1])
                WFM1 = WFM(gg,nx.subgraph(gg,LCtemp1),added)
                LCtemp1.append(j)
                WFM2 = WFM(gg,nx.subgraph(gg,LCtemp1),j)
                if WFM2 <= WFM1: 
                    LCtemp1.remove_node(j)
                added = j
            if len(LCtemp1) > res:
                return(nx.subgraph(gg,LCtemp1))
            else: 
                return(0)
        elif len([x for x in com0 if x in com1]) > 0: 
            return(0)

# LC function returns local communities 
def LC(g,res):
    b = []
    for i in sorted(g.edges(data=True), key = lambda x: x[2]['betweenness']):
        c = LCtemp(g,i,b,res)
        if c != 0: 
            b.append(copy.deepcopy(c))
    return(b)

# Function for calculating count of common vertices of two local communities 
def commonVertices(C1,C2):
    l1 = list(C1.nodes())
    l2 = list(C2.nodes())
    return(len([x for x in l1 if x in l2]))

# Function for calculating count of common edges of two local communities 
def commonEdges(g,C1,C2):
    edges1 = []
    edges2 = []
    for i in sorted(C1.edges(data=True), key = lambda x: x[2]['betweenness']):
        edges1.append(i[2]['EdgeID'])
    for j in sorted(C2.edges(data=True), key = lambda x: x[2]['betweenness']):
        edges2.append(j[2]['EdgeID'])
    return(len([x for x in edges1 if x in edges2]))


# Function for calculation LOS 
def WDLI(g,C1,C2,alpha):
    nv = commonVertices(C1,C2)
    ne = commonEdges(g,C1,C2)
    
    if nv == 0: 
        a = math.log10(1)
    else: 
        a = math.log10(nv)
    
    if ne == 0: 
         b = math.log10(1)
    else: 
        b =  b = math.log10(ne)
   
    if a == 0: 
        firstPart = 0
    else: 
        firstPart = alpha/a
    if b == 0: 
        secondPart = 0
    else:
        secondPart = (1-alpha)/b
    maxpart = max(firstPart,secondPart)
    
    if maxpart == 0: 
        return(0)
    else: 
        return((firstPart+secondPart)/maxpart)


# Function for merging local communities 
def LCmerger(l,g,alpha,beta):
    mergedCommunities = []
    for i in l:
        for j in l:
            if list(i.nodes()) == list(j.nodes()) :
                continue
            if WDLI(g,i,j,alpha) >= beta:
                i = nx.compose(i,j)
                l.remove(j)
        mergedCommunities.append(i)
    for z in l: 
        if z not in mergedCommunities:
            mergedCommunities.append(z)
    mergedFinal = []
    for k in mergedCommunities: 
        if k not in mergedFinal: 
            mergedFinal.append(k)
    return(mergedFinal)

#function to retreive in community nodes: 
def incomnodes(m):
    inCommNodes1 = []
    for i in m: 
        inCommNodes1.append(list(i.nodes()))
    inCommNodes1 = list(dict.fromkeys(sum(inCommNodes1, [])))
    inCommNodes = []
    for j in inCommNodes1:
        if j not in inCommNodes:
            inCommNodes.append(j)
    return(inCommNodes)


# function to calculate out of community nodes 
def outComNodes(g,inCom):
    outNodes = []
    for i in sorted(g.edges(data=True), key = lambda x: x[2]['betweenness']):
            outNodes.append(i[0])
            outNodes.append(i[1])
            outNodes = list(dict.fromkeys(set(outNodes)))
            for j in outNodes: 
                if j in inCom:
                    outNodes.remove(j)
            for n in outNodes: 
                if n == 'from':
                    outNodes.remove('from')
                if n == 'to':
                    outNodes.remove('to')
    return(outNodes)


# Refinement Function 
def communityRefining(g,allNode,nodeInCommunity,mergedCommunities):
    finalCommunities = []
    for i in mergedCommunities:
        for j in allNode:
            if j in nodeInCommunity:
                continue
            WFMBefore = WFM(g,i,None)
            unfrozen = nx.DiGraph(i)
            unfrozen.add_node(j)
            WFMMAfter = WFM(g,unfrozen,j)
            unfrozen.remove_node(j)
            fitness = WFMMAfter - WFMBefore
            if fitness > 0: 
                allNode.remove(j)
                nodeInCommunity.append(j)
                unfrozen.add_node(j)
                finalCommunities.append(unfrozen)
        finalCommunities.append(i)
    final = []
    for x in finalCommunities:
        if x not in final: 
            final.append(x)
    return(final)


#Cluster adjuster 
def pur(l1,l2):
    dif = len(l1) - len(l2)
    if dif > 0:
        while dif > 0:
            l2.append(0)
            dif = dif - 1 
    elif dif < 0: 
        while dif < 0: 
            l1.append(0)
            dif = dif + 1

# Final Sets adjuster 
def pur2(l1,l2):
    dif = len(l1) - len(l2)
    if dif > 0:
        shortest = min(l2, key=len)
        while dif > 0: 
            l2.append(shortest)
            dif = dif - 1
    elif dif < 0:
        shortest2 = min(l1, key = len)
        while dif < 0: 
            l1.append(shortest2)
            dif = dif + 1
    
#NMI Calculator for two clusters: 
def NMI(l1,l2):
    lst = []
    for i in range(0,len(l1)):
        lst.append(metrics.normalized_mutual_info_score(copy.deepcopy(l1[i]),copy.deepcopy(l2[i])))
    return(statistics.mean(lst))
        

# Main Function 
def OCDWFC(digraph,alpha,beta,res):
    l = LC(digraph,res)
    merge = LCmerger(l,digraph,alpha,beta)
    inCommunityNodes = incomnodes(merge)
    outOfCommunityNodes = outComNodes(digraph,inCommunityNodes)
    final = communityRefining(digraph,outOfCommunityNodes,inCommunityNodes,merge)
    finallist = []
    for i in final: 
        finallist.append(list(i.nodes))
    return(finallist)


# NMI Analysis function 
def NMI_Analysis(l1,l2):
    a = sorted(l1, key = len, reverse = True)
    b = sorted(l2, key = len, reverse = True)
    pur2(a,b)
    for i in range(0,len(a)):
        pur(a[i],b[i])
    print("NMI Value between two clustering sets is: ", NMI(a,b))
    

    
    
    
# communityCount function
# a is a node; c is the set of communities after the community detection process

def communityCount(a,c):
    out = 0 
    for item in c: 
        if a in item: 
            out = out + 1 
    return(out)


# EQ Measure function 
def EQ_inside(g,c,ct):
    m = len(list(g.edges()))
    out = 0
    tabu = []
    for i in c:
        if i not in g.nodes():
            continue
        for j in c:
            if j not in g.nodes():
                continue
            if i == j or j in tabu: 
                continue 
            a = communityCount(i,ct)
            b = communityCount(j,ct)
            if i in g.neighbors(j): 
                ad = 1
            elif i not in g.neighbors(j):
                ad = 0
            deg1 = g.degree(i)
            deg2 = g.degree(j)
            tabu.append(j)
            out = out + (1 / (a * b)) * (ad - (deg1 * deg2)/2*m)   
    return(out)

# EQ final 
def EQ(g,c):
    m = len(list(g.edges()))
    out = 0
    for item in c: 
        out = out + EQ_inside(g,item,c)
    return((1/2*m) * out)


# In[29]:


start_time2 = time.time()
final = OCDWFC(Graph,0.5,0.75,4)
print("--- %s seconds ---" % (time.time() - start_time2))


# In[30]:


NMI_Analysis(final,toCompare)


# In[31]:


print(len(final),len(toCompare))


# In[35]:


EQ(Graph,final)


# In[ ]:




