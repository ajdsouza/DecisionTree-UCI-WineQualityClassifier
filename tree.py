# CSE6242/CX4242 Homework 4 Sketch Code
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import pprint
import math
import numbers
import decimal
import random
import json
import sys
from collections import defaultdict
from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle
import copy
from operator import itemgetter 


if sys.version_info[0] != 2 or sys.version_info[1] < 6:
    print("This script requires Python version 2.6")
    sys.exit(1)

# class for saveing decision tree to disk as json and reading it back
class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct


#scipy only for calculating the inverse norm of the cdf of a standard gaussian
#from scipy.stats import norm

# Enter You Name Here
myname = "Dsouza-Ajay-" # or "Doe-Jane-"
pp = pprint.PrettyPrinter(indent=4)


#------------------------------------------
# decison tree
#------------------------------------------
# Implement your decision tree below
class DecisionTree():

    
    # get the k tiles dynamically at each split
    def getKTiles(self, K, dataIdx,decisionAttrib,resultAttrib):
        
        # get all the data and sort it ascending
        ld = len(dataIdx)
        
        dvals={}
        vals=[]

        # keep the list of data indexes for each value
        for i in dataIdx:
            
            # get the value fo the attrib in that data index
            rVal = self.data[i][self.attribs[decisionAttrib]]

            if rVal not in dvals:
                dvals[rVal]=[i]
            else:
                dvals[rVal].append(i)
                
            vals.append(rVal)
                
        vals.sort()

        # make K cuts and return the values at those positions
        # get the cuts for the bin and the records at each bin as well
        # along with a count for each set of resultParam values for entropy calc
        results = {}
        
        # for 1 or 2 data points it is trivial
        if ld <= 2:

            dkvals = dvals.keys()
            dkvals.sort()
            
            for vl in dkvals:
                
                if vl not in results:
                    results[vl] = {}
                    results[vl]['bin_records'] = []
                    results[vl]['bin_records_result_count'] = {}

                results[vl]['bin_records']=dvals[vl]
            
                # keep count of the result values in each bin
                for il in dvals[vl]:
                    
                    rVal = self.data[il][self.attribs[resultAttrib]] 
                    
                    if rVal in results[vl]['bin_records_result_count']:
                        results[vl]['bin_records_result_count'][rVal] += 1
                    else:
                        results[vl]['bin_records_result_count'][rVal] = 1
                        
                del dvals[vl]

        else:
            # get the cuts for the bin and the records at each bin as well
            # along with a count for each set of resultParam values for entropy calc
            if (ld < 2*K):
                nk = ld/2
            else:
                nk = K
            
            gk = ld / nk
            rk = ld % nk
            previ = 0
            
            # the cuts of bins
            for i in range(1,nk+1):
                
                # the remainder gets distributed to the earlier cuts
                ki = i*gk-1
                if (i <= rk):
                    ki += i
                else:
                    ki += rk

                if vals[ki] not in results:
                    results[vals[ki]] = {}
                    results[vals[ki]]['bin_records'] = []
                    results[vals[ki]]['bin_records_result_count'] = {}
                    
                # get the records which belong to this bin
                for vl in vals[previ:ki+1]:

                    if vl in dvals:

                        # the total records of this bin
                        results[vals[ki]]['bin_records'].extend(dvals[vl])

                        # keep count of the result values in each bin
                        for il in dvals[vl]:

                            rVal = self.data[il][self.attribs[resultAttrib]] 

                            if rVal in results[vals[ki]]['bin_records_result_count']:
                                results[vals[ki]]['bin_records_result_count'][rVal] += 1
                            else:
                                results[vals[ki]]['bin_records_result_count'][rVal] = 1

                        del dvals[vl]

                # special case where this bin and prev bin have same value
                # if we have repeating values in successive ktiles
                if len(results[vals[ki]]['bin_records']) == 0:
                    del results[vals[ki]]

                previ = ki+1
                        

        return results





    # perform 10 fold crossvalidation on the training set provided
    # to learn a decision Tree and tuning the parameters
    # select the K for Ktile
    # select complexityCost to be used for tree pruning
    #
    # The best tree from crossvalidation results is returned
    # and is used to measure accuracy on the test data which 
    # has been held back
    def learn(self, training_set, title):

        # implement this function
        self.attribs = title.copy()
        self.attribType = {k:'numeric' for k in self.attribs.keys()}
        
        # go thru the attribs and determine the type ie either 
        # categorical or continuous
        # in our case string or numeric
        for v in training_set:
            if 'numeric' in self.attribType.values():
                for k in [ ak for ak in self.attribType.keys() if self.attribType[ak] == 'numeric' ]:
                    if ( not isinstance( v[self.attribs[k]], (int,float,long)) ):
                        self.attribType[k]='not_numeric'
            else:
                break


        cvResults={}

        # perform traning for range of Ktile, and complexity costs for best tree pruning
        # and 10 fold cross validation
        # 

        # complexity cost
        bestNodeList = {}
        bestAvgAcc = float("-inf")
        bestBuf = None


        # K tile to be used
        for k in [2,4,5,6,8,10,15,16,20]:

            for useGini in [True,False]:
                
                self.useGini = useGini

                # Split training/test sets
                # You need to modify the following code for cross validation.
                cv_avgAcc=0

                self.kTile = k

                # Print CROSS validation - training accuracy       
                pbuf = "kTile=%d,costComplexityFactor=%.4f,Gini=%s" % ( self.kTile , float(self.ccf), self.useGini) 

                count=0.0
                for N in xrange(0, 10):

                    training_data = []
                    cv_set = []

                    training_data = [x for i, x in enumerate(training_set) if (i-N) % 10 != 0]
                    cv_set = [x for i, x in enumerate(training_set) if (i-N) % 10 == 0]

                    self.nodeList.clear()
                    del self.data[:]
                    self.data = training_data

                    # Construct a tree using training set
                    # print("Training Tree:N=%d->%s" % (N, pbuf) )
                    self.train( training_data , title )

                    if self.pruneTree:
                        # Prune Tree now based on cost complexity
                        # print("Pruning Tree:N=%d->%s" % (N, pbuf) )
                        self.prune(cv_set)

                    # self.printTree()

                    # CROSS Validation
                    # Classify the cross validation set using the tree we just constructed
                    # Cross validation - Training Accuracy
                    cv_accuracy = self.getCVAccuracy(cv_set)

                    sbuf = "CrossValidation:N=%d->%s,training accuracy=%.4f" % (N, pbuf, cv_accuracy) 

                    print(sbuf)

                    # Writing results to a file (DO NOT CHANGE)
                    f = open(myname+"result-crossvalidation.txt", "ab+")
                    f.write(sbuf+"\n")
                    f.close()

                    cv_avgAcc += cv_accuracy
                    count +=1

                    del training_data[:]
                    del cv_set[:]
                

            # save average Training/Cross Validation accuracy with training with one set of paramaters
            cv_avgAcc = float(cv_avgAcc)/float(count)

            cvResults[cv_avgAcc] = {
                'buf':pbuf
                } 

            # Check if the accuracy of the tree trained with these parameters is the best so far
            # if so save the parameters in a buffer
            if cv_avgAcc > bestAvgAcc:
                bestAvgAcc = cv_avgAcc
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)
                bestBuf = pbuf



        # Cross validation completed
        #
        # stick the best parameters from cross validation back to this tree and 
        # return the trained tree for use and testing
        self.nodeList.clear()
        self.nodeList = bestNodeList
                    
        # Write the best tree from cross validation to be used
        sbuf = "Best Tree from Cross Validation:%s,training accuracy=%.4f" % (bestBuf, bestAvgAcc ) 

        print(sbuf)

        # Writing results to a file (DO NOT CHANGE)
        f = open(myname+"result-crossvalidation.txt", "ab+")
        f.write(sbuf+"\n")
        f.close()

        # Parameter settings giving best N=10 results in training:cross validation
        f = open(myname+"result-crossvalidation.txt", "ab+")
        
        cvKs = cvResults.keys()
        cvKs.sort(reverse=True)
        for svr in cvKs[:N]:
        
            res=cvResults[svr]
            
            sbuf = "Best Training Results:%s,Average Training accuracy: %.4f" % ( res['buf'], svr)
            
            print(sbuf)
            f.write(sbuf+"\n")
            
        f.close()




    # save the decision tree to the disk as json
    def saveToDiskAsJson(self,fileName):
        
        saveDict  = {}
        saveDict['attribs'] = self.attribs
        saveDict['attribType'] = self.attribType
        saveDict['nodeList'] = self.nodeList

        # save the best tree to disk as json
        jSaveDict = dumps(saveDict, cls=PythonObjectEncoder)
        with open(fileName, 'w') as fp:
            json.dump(jSaveDict, fp)
            fp.close()
 

    # Build the decision tree from disk file as json
    def readTreeFromJsonFile(self, fileName):
        with open(fileName, 'r') as fp:
            jSaveDict = json.load(fp)
            saveDict = loads(jSaveDict, object_hook=as_python_object)
            self.attribs = saveDict['attribs']
            self.attribType = saveDict['attribType']
            self.nodeList = saveDict['nodeList']
  
        
    
            
    # default constructor
    def __init__(self):
        self.attribs = []
        self.attribType = {}
        self.nodeList = {}
        self.data = []
        self.kTile = 15
        self.useGini = False
        # constant parameters, these values have been found to give the best performance
        # to save time, no need to cross validate on these
        self.useBinEntropy=False
        self.branchForEntropyGainOnly = False
        self.returnMajorityNotDefault = False
        self.pruneTree = True
        self.target = 'quality'
        self.ccf = 0



    # train the decision tree on the training set passed
    def train ( self, training_data, title):
        
        rootNode = {
            'key':'root',
            'parent':None,
            'type' : 'root',
            'attribs':title.copy(), 
            'dataIdx': set(range(len(training_data))) , 
            'children': [], 
            'bin':None,
            'parentAttrib':None,
            'level':0,
            'chosenAttrib':None,
            'parentKey':None
            }


        self.nodeList['root']=rootNode
                    
        stack = []
        stack.append('root')

        while stack:

            nodeKey = stack.pop(0)

            node = self.nodeList[nodeKey]
            
            # get the stats for the node 
            # max of the data value in the target column of the remaining set
            lv = [self.data[i][self.attribs[self.target]] for i in node['dataIdx']]
            
            if len(lv):
                majorityValue = max(lv,key=lv.count)
                node['majorityValue']=majorityValue

            # count of values
            node['total']=len(node['dataIdx'])

            # count of correct classification based on majorityValue
            errCount=0
            for i in node['dataIdx']:
                if self.data[i][self.attribs[self.target]] != majorityValue:
                    errCount +=1

            node['error_for_majority']=errCount


            # leaf node OR finished dividing the whole tree TBD
            # we do not add 0 item bins to stack for branching, so we will not come here
            #if ( len(node['dataIdx']) == 0 ):
            #    node['type']='leaf'
            if ( len(node['attribs']) == 1):
                node['type']='leaf'
                continue
            else:
                
                etp1 = Entropy(self,node['dataIdx'],self.target,self.target)
                
                # all target values are same
                if ( etp1 == 0 ):
                    node['type']='leaf'
                    continue
                else:
                    # find the best of the remaining attribs to classify the remaining data points on this tree branch
                    bestAttrib = ''
                    betp2=1.01
                    for bAttrib in set(set(node['attribs'])-set([self.target])):
                        
                        etp2 = Entropy(self,node['dataIdx'],self.target,bAttrib)
                        
                        if etp2 < betp2:
                            bept2 = etp2
                            bestAttrib = bAttrib
                            
                    
                    
                    # this check not for rootNode
                    # if the entripy of the branch is not less then the entropy
                    # at this node, then end branching along this node
                    if node['type'] != 'root':
                        if self.branchForEntropyGainOnly:
                            if betp2 >= etp1:
                                node['type']='leaf'
                                continue

                    # find the values for the chosen attrib to split next
                    #  find the data rows divided for each attrib value
                    #  add a node for each of them to the stack
                    

                    if ( self.attribType[bestAttrib] == 'numeric' ):
                        aVResult = self.getKTiles(self.kTile, node['dataIdx'],bestAttrib,self.target)   
                        aVSet = set( aVResult.keys() )
                    else:
                        aVSet = set( self.data[i][self.attribs[bestAttrib]] for i in node['dataIdx'] )
                    
        
                    node['chosenAttrib']=bestAttrib
                    td=node['children']

                    for aVal in aVSet:

                        # get the records with values in that range to be branched to same branch
                        if ( self.attribType[bestAttrib] == 'numeric' ):
                            vIdx = set( aVResult[aVal]['bin_records'] )                            
                        else:
                            vIdx = set( i for i in node['dataIdx'] if self.data[i][self.attribs[bestAttrib]] == aVal )
                        
                        vAttribs = node['attribs'].copy()
                        vAttribs.pop(bestAttrib)
                        
                        childNodeKey = nodeKey+"--"+bestAttrib+"--"+str(aVal)
                        td.append(childNodeKey)

                        childNode =  {
                            'type':'subTree',
                            'key':childNodeKey,
                            'attribs':vAttribs, 
                            'dataIdx': vIdx , 
                            'bin':aVal,
                            'children': [],
                            'parentAttrib':bestAttrib,
                            'parentKey': nodeKey,
                            'level': node['level']+1,
                            'chosenAttrib':None}

                        self.nodeList[childNodeKey]=childNode

                        stack.append(childNodeKey )
                         




    # Prune Tree now based on cost complexity
    # we know error at each node from majorityValue
    # compare that with error from subtree
    # We start from the bottom
    #
    # A)
    # for each node
    #  compute costComplexityFactor 
    #    ccf = Error(Node) - Error(Subtree)/(Leaves in Subtree -1 )
    #
    #  Remove the node with the lowest ccf to get new node T_i
    # 
    #  check the error now with cv_set
    #   save tree as best tree if accuracy is best so far
    #   
    #  Repeat A again till we come to a one node tree
    # 
    #  return the best tree
    #

    def prune(self,cv_set):
                
        # as long as there are more than 1 node, try every subset of tree
        # by removing one subtree at a time
            
        # Cross validation - Training Accuracy
        bestAccuracy = self.getCVAccuracy(cv_set)
        bestNodeList = copy.deepcopy(self.nodeList)

        # remove all subtrees that have more errors than nodes
        self.pruneAllCostlierSubTrees(cv_set)
        
        # if this tree is better than the earlier best
        # then save this
        cv_accuracy = self.getCVAccuracy(cv_set)
        if cv_accuracy > bestAccuracy:
            bestAccuracy = cv_accuracy
            bestNodeList.clear()
            bestNodeList = copy.deepcopy(self.nodeList)

        # now prune one at a time the subtree which gives the least cost/complexity
        # and cross validate
        while len(self.nodeList.keys()) > 1:

            #print("Prune: with len(self.nodeList)=%d" % len(self.nodeList.keys()))
            
            # stack compute nodes from the bottom most level, breadth first
            prStack = {}

            # add the leaf nodes to the stack for bottom up traversal
            for nodeKey in self.nodeList:
                node = self.nodeList[nodeKey] 
                if node['type'] == 'leaf':
                    if node['level'] not in prStack:
                        prStack[node['level']] = []
                    prStack[node['level']].append(nodeKey)


            # Go thru all the nodes and determine the cost complexity factor for each node
            visited = {}
            minCcF = float('inf')
            maxLevel = max(prStack.keys())
            pruneNodeKey = None

            # traverse one level at a time bottom up, breadth first
            for level in reversed(xrange(1,maxLevel+1)):

                # go thru the nodes at that level
                while prStack[level]:
                
                    # BFS, FIFO
                    node = self.nodeList[prStack[level].pop(0)]

                    if node['key'] in visited:
                        continue;

                    visited[node['key']]=1

                    # add a nodes parent, if multiple children add the same parent
                    # the visited list will ensure we skip it
                    if node['parentKey']:
                        if node['level'] not in prStack:
                            prStack[node['level']] = []
                        prStack[node['level']].append(node['parentKey'])


                    # complexity is the number of leaf nodes in a subtree
                    # for leaf the complexity and errors are its own
                    if node['type'] == 'leaf':
                        
                        node['errors'] = node['error_for_majority'] 
                        node['leaves'] = 1
                        node['ccf'] = float('inf')

                    # get the errors of its subtree
                    elif node['type'] != 'leaf':

                        # get the errors and complexity from having the subree as it is
                        cErrors = 0.0
                        leaves =  0.0
                        
                        for ck in node['children']:
                            
                            cNode = self.nodeList[ck]
                            cErrors +=  cNode['errors']
                            leaves += float(cNode['leaves'])

                        # save the subtree values for each node and get ccf
                        node['leaves'] = leaves
                        node['errors'] = cErrors                            
                        if float(leaves) != 1:
                            node['ccf'] =  ( float(node['error_for_majority']) - cErrors ) / float(leaves-1.0)
                        else:
                            node['ccf'] = float('-inf')

                        # keep track of the node with the least ccf for pruning
                        if node['ccf'] <= minCcF:
                            minCcF = node['ccf']
                            pruneNodeKey = node['key']

           

            # find the node with the minimum cost complexity factor and prune its subtree it
            if pruneNodeKey:
                self.pruneSubTree(pruneNodeKey)
                

            # if this tree is better than the earlier best
            # then save this
            cv_accuracy = self.getCVAccuracy(cv_set)
            if cv_accuracy > bestAccuracy:
                bestAccuracy = cv_accuracy
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)

        # take the bestNodeList as the best pruned tree based on cv validation
        self.nodeList.clear()
        self.nodeList = bestNodeList
        




    # Prune Tree in one pass
    # remove all subtrees that have higher error than 
    # majority classification at parent node
    #
    # use cross validation to get the best ccf
    #
    def pruneAllCostlierSubTrees(self,cv_set):
                

        bestAccuracy = float('-inf')
        bestNodeList = copy.deepcopy(self.nodeList)
        savedNodeList = copy.deepcopy(self.nodeList)

        # try pruning the costlier subtrees for various cost complexity factors
        # to pick the best tree based on cross validation
        for ccf in [0.0,0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0]:

            
            # take the bestNodeList as the best pruned tree based on cv validation
            self.nodeList.clear()
            self.nodeList = copy.deepcopy(savedNodeList)

            # compute nodes from the bottom most level, breadth first
            prStack = {}
            
            # add the leaf nodes to the stack for bottom up traversal
            for nodeKey in self.nodeList:
                node = self.nodeList[nodeKey] 
                if node['type'] == 'leaf':
                    if node['level'] not in prStack:
                        prStack[node['level']] = []
                    prStack[node['level']].append(nodeKey)



            visited = {}

            maxLevel = max(prStack.keys())

            for level in reversed(xrange(1,maxLevel+1)):

                while prStack[level]:

                    # BFS, FIFO
                    node = self.nodeList[prStack[level].pop(0)]

                    if node['key'] in visited:
                        continue;

                    visited[node['key']]=1

                    # add a nodes parent, if multiple children add the same parent
                    # the visited list will ensure we skip it
                    if node['parentKey']:
                        if node['level'] not in prStack:
                            prStack[node['level']] = []                       
                        prStack[node['level']].append(node['parentKey'])


                    # for an leaf the errors are the errors of majority classification
                    if node['type'] == 'leaf':

                        node['errors'] = node['error_for_majority'] 
                        node['leaves'] = 1

                    # for other nodes get the errors from its substree
                    if node['type'] != 'leaf':

                        # get the errors from the subree
                        cErrors = 0
                        leaves = 0

                        for ck in node['children']:

                            cNode = self.nodeList[ck]
                            cErrors +=  cNode['errors']
                            leaves += float(cNode['leaves'])

                        node['leaves'] = leaves
                        # subree is less cost/complex than majorityvalue at current node - keep the subtree
                        if  node['error_for_majority']  > ( cErrors + ccf*(leaves-1) ):
                            node['errors'] = cErrors
                        # else the node has less error than subtree, prune subtree
                        else:
                            self.pruneSubTree(node['key'])



            # if this tree is better than the earlier best
            # then save this
            cv_accuracy = self.getCVAccuracy(cv_set)
            if cv_accuracy > bestAccuracy:
                bestAccuracy = cv_accuracy
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)
                self.ccf = ccf



        # take the bestNodeList as the best pruned tree based on cv validation
        self.nodeList.clear()
        self.nodeList = bestNodeList




               
          
    # prune a subtree below a node
    # remove children from the nodeList
    # and set the current nodes children to null
    def pruneSubTree(self,pruneNodeKey):

        node = self.nodeList[pruneNodeKey]
        
        # prune the subtree under this node with min ccf
        # print("Pruning at Node="+node['key'])
        
        node['errors']= node['error_for_majority']
        node['type']='leaf'
        node['chosenAttrib']=None
        node['leaves']=1
        node['ccf']=float('inf')

        # remove the child nodes from the nodelist
        pruneStack = node['children']

        node['children']=[]

        while pruneStack:

            prNodeKey = pruneStack.pop(0)
            prNode = self.nodeList[prNodeKey]

            # get the children of this subtree node before deleting it
            if prNode['type'] != 'leaf':
                for ck in prNode['children']:
                    pruneStack.append(ck)

            del self.nodeList[prNode['key']]




    # get average classification accuracy with the cv_set
    def getCVAccuracy(self,cv_set):

        # cross validate with the cv_set to get the accuracy of this tree
        # if it is better than the previous best save it in best tree
        # CROSS Validation
        # Classify the cross validation set using the tree we just constructed
        cv_results = []
        for instance in cv_set:
            cv_result = self.classify( instance[:-1] )
            cv_results.append( cv_result == instance[-1])
            
        # Cross validation - Training Accuracy
        cv_accuracy = float(cv_results.count(True))/float(len(cv_results))

        return cv_accuracy


        

    def printTree(self):

        pStack = []
        pStack.append('root')

        while pStack:

            node = self.nodeList[pStack.pop()]
            
            idC = '\t' * node['level']
            if node['type']=='root':
                print("Root")
            else:
                print("%s%s=%s" % (idC,node['parentAttrib'],node['bin']) )
            
            if node['type'] == 'leaf':
                print("%s%s=%s" % (idC,self.target,node['majorityValue']) )

            for chk in node['children']:
                pStack.append(chk)





    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0

        # implement this function
 
        nodeKey = 'root'

        while nodeKey:
                      
            node = self.nodeList[nodeKey]

            # reached the leaf , return the result value
            if node['type'] == 'leaf':
                return node['majorityValue']

            attrib = node['chosenAttrib']
 
            # value of that attrib in test data
            tval = test_instance[self.attribs[attrib]]

            # next decide which branch based on value
            # if numeric attrib, check range
            nodeKey=None
            if self.attribType[attrib] == 'numeric':

                prev=float("-inf")
                lastNK=None

                nkDs = {self.nodeList[cnK]['bin']:cnK for cnK in node['children']}
                nkKs = nkDs.keys()
                nkKs.sort()
                
                for binVal in nkKs:
                    nK = nkDs[binVal]
                    if prev < tval <= binVal:
                        nodeKey = nK
                        break
                    prev=binVal
                    lastNK=nK
 
                # Did not find a bin this value should belong to
                # pick the the largest bin as
                if self.returnMajorityNotDefault and not nodeKey:
                    #nodeKey=lastNK
                    return node['majorityValue']
   
            # not numeric, treat it as categorical data
            else:
                
                for nK,binVal in [(cnK,self.nodeList[cnK]['bin']) for cnK in node['children']]:   
                    if tval.lower() == binVal.lower():
                        nodeKey=nk
                        break
                    
                # Did not find a bin this value should belong to
                # return the MajorityValue of this node as result
                if self.returnMajorityNotDefault and not nodeKey:
                    return node['majorityValue'] 
             

        return result
    










#------------------------------------------
# RandomForest
#------------------------------------------
# Implement your RandomForest below
class RandomForest():

    
    # get the k tiles dynamically at each split
    def getKTiles(self, K, dataIdx,decisionAttrib,resultAttrib):
        
        # get all the data and sort it ascending
        ld = len(dataIdx)
        
        dvals={}
        vals=[]

        # keep the list of data indexes for each value
        for i in dataIdx:
            
            # get the value fo the attrib in that data index
            rVal = self.data[i][self.attribs[decisionAttrib]]

            if rVal not in dvals:
                dvals[rVal]=[i]
            else:
                dvals[rVal].append(i)
                
            vals.append(rVal)
                
        vals.sort()

        # make K cuts and return the values at those positions
        # get the cuts for the bin and the records at each bin as well
        # along with a count for each set of resultParam values for entropy calc
        results = {}
        
        # for 1 or 2 data points it is trivial
        if ld <= 2:

            dkvals = dvals.keys()
            dkvals.sort()
            
            for vl in dkvals:
                
                if vl not in results:
                    results[vl] = {}
                    results[vl]['bin_records'] = []
                    results[vl]['bin_records_result_count'] = {}

                results[vl]['bin_records']=dvals[vl]
            
                # keep count of the result values in each bin
                for il in dvals[vl]:
                    
                    rVal = self.data[il][self.attribs[resultAttrib]] 
                    
                    if rVal in results[vl]['bin_records_result_count']:
                        results[vl]['bin_records_result_count'][rVal] += 1
                    else:
                        results[vl]['bin_records_result_count'][rVal] = 1
                        
                del dvals[vl]

        else:
            # get the cuts for the bin and the records at each bin as well
            # along with a count for each set of resultParam values for entropy calc
            if (ld < 2*K):
                nk = ld/2
            else:
                nk = K
            
            gk = ld / nk
            rk = ld % nk
            previ = 0
            
            # the cuts of bins
            for i in range(1,nk+1):
                
                # the remainder gets distributed to the earlier cuts
                ki = i*gk-1
                if (i <= rk):
                    ki += i
                else:
                    ki += rk

                if vals[ki] not in results:
                    results[vals[ki]] = {}
                    results[vals[ki]]['bin_records'] = []
                    results[vals[ki]]['bin_records_result_count'] = {}
                    
                # get the records which belong to this bin
                for vl in vals[previ:ki+1]:

                    if vl in dvals:

                        # the total records of this bin
                        results[vals[ki]]['bin_records'].extend(dvals[vl])

                        # keep count of the result values in each bin
                        for il in dvals[vl]:

                            rVal = self.data[il][self.attribs[resultAttrib]] 

                            if rVal in results[vals[ki]]['bin_records_result_count']:
                                results[vals[ki]]['bin_records_result_count'][rVal] += 1
                            else:
                                results[vals[ki]]['bin_records_result_count'][rVal] = 1

                        del dvals[vl]

                # special case where this bin and prev bin have same value
                # if we have repeating values in successive ktiles
                if len(results[vals[ki]]['bin_records']) == 0:
                    del results[vals[ki]]

                previ = ki+1
                        

        return results





    # perform 10 fold crossvalidation on the training set provided
    # to learn a decision Tree and tuning the parameters
    # select the K for Ktile
    # select complexityCost to be used for tree pruning
    #
    # The best tree from crossvalidation results is returned
    # and is used to measure accuracy on the test data which 
    # has been held back
    def learn(self, training_set, title):

        # implement this function
        self.attribs = title.copy()
        self.attribType = {k:'numeric' for k in self.attribs.keys()}
        
        # go thru the attribs and determine the type ie either 
        # categorical or continuous
        # in our case string or numeric
        for v in training_set:
            if 'numeric' in self.attribType.values():
                for k in [ ak for ak in self.attribType.keys() if self.attribType[ak] == 'numeric' ]:
                    if ( not isinstance( v[self.attribs[k]], (int,float,long)) ):
                        self.attribType[k]='not_numeric'
            else:
                break


        cvResults={}

        # perform traning for range of Ktile, and complexity costs for best tree pruning
        # and 10 fold cross validation
        # 

        # complexity cost
        bestRandomForest = {}
        bestAccuracy = float("-inf")
        bestBuf = None

        dln = len(training_set)
        dlnIdx = [i for i in xrange(0,dln)]

        # num of random attribs
        self.m = math.ceil(math.sqrt(len(self.attribs)))
           
        # num of bins is 2 for random forests
        self.kTile = 16
     
        # Tree Size to be used
        for B in [100 ]:

            # Split training/test sets
            # You need to modify the following code for cross validation.
            cv_avgAcc=0

            # min nodes to be reached
            #for nmin in [3,7,15,31]:
            for nmin in [5000]:
     
                self.nmin = nmin

                for newM in [True]:

                    self.newMAttribsForSplit = newM
                    
                    for samplePercent in [1.0]:
                        
                        self.samplePercent = samplePercent
       
                        for mWoReplacement in [True]:

                            self.pickMAttribWOReplacement = mWoReplacement

                            for useGini in [True,False]:
                                
                                self.useGini = useGini

                                # Print CROSS validation - training accuracy       
                                pbuf = "Trees=%d,random feature size m=%d,nmin=%f,newM=%s,sample=%.2f,mWoReplace=%s,Gini=%s" % ( 
                                    B , self.m, self.nmin, self.newMAttribsForSplit, self.samplePercent, self.pickMAttribWOReplacement, self.useGini) 

                                # keep track of voting of eahc tree on oob
                                vote_results = {}
                                cv_results = []

                                for T in xrange(0,B):

                                    #print("Start N=%d ,%s" %(T,pbuf))

                                    # get a bootstrap sample samplePercent of total size
                                    # split the data as test and training data
                                    sampleSize = int(dln*samplePercent)

                                    training_data_idx = []
                                    training_data = []
                                    cv_set_idx = []

                                    while len(training_data_idx) < sampleSize:
                                        r = random.choice(dlnIdx)
                                        training_data_idx.append(r)

                                    training_data_tuple = itemgetter(*training_data_idx)(training_set)
                                    training_data = list(training_data_tuple)
                                    del training_data_tuple
                                    cv_set_idx = list(set(dlnIdx) - set(training_data_idx)) 

                                    self.nodeList.clear()
                                    del self.data[:]  
                                    self.data = training_data

                                    # Construct a tree using training set
                                    self.train( training_data )
                                    #print("Trained N=%d ,%s" %(T,pbuf))

                                    #self.printTree()

                                    self.randomForest[T]={}
                                    self.randomForest[T]['nodeList']=copy.deepcopy(self.nodeList)

                                    # CROSS Validation with OOB
                                    # Classify the cross validation set using the tree we just constructed
                                    # Cross validation - Training Accuracy

                                    for iidx in cv_set_idx:

                                        cvrs = self.getRandomForestVote( training_set[iidx][:-1] )        

                                        if training_set[iidx][-1] != 0 and cvrs != 0:
                                            print("Training=%f, vote=%f" % (training_set[iidx][-1], cvrs))

                                        # keep track of the vote of this random forest for each index 
                                        if iidx not in vote_results:
                                            vote_results[iidx] = {}

                                        if cvrs not in vote_results[iidx]:
                                            vote_results[iidx][cvrs] = 1
                                        else:
                                            vote_results[iidx][cvrs] += 1

                                    del training_data[:]
                                    del training_data_idx[:]
                                    del cv_set_idx[:]



                                # finished training the B random trees, noe get the vote in cv_results
                                # Cross validation - Training Accuracy
                                for iidx in vote_results:
                                    vote_result = vote_results[iidx]
                                    for hVote in sorted(vote_result, key=vote_result.get,reverse=True):

                                        if hVote == 1:
                                            print("Got 1")

                                        #if training_set[iidx][-1] != 0:
                                        #    print("Training=%f" % (training_set[iidx][-1]))
                                        #    for k in vote_result:
                                        #        print("%f=%f" % (k,vote_result[k]))

                                        cv_results.append(hVote == training_set[iidx][-1])
                                        break;

                                # for this random forest get the accuracy in OOB
                                cv_accuracy = float(cv_results.count(True))/float(len(cv_results))

                                sbuf = "CrossValidation:%s,training accuracy=%.4f" % ( pbuf, cv_accuracy) 

                                print(sbuf)

                                # Writing results to a file (DO NOT CHANGE)
                                f = open(myname+"result-crossvalidation.txt", "ab+")
                                f.write(sbuf+"\n")
                                f.close()

                                # cache the results of this random forest
                                cvResults[cv_accuracy] = {
                                    'buf':pbuf
                                    } 

                                # Check if the accuracy of the tree trained with these parameters is the best so far
                                # if so save the parameters in a buffer
                                if cv_accuracy > bestAccuracy:
                                    bestAccuracy = cv_accuracy
                                    bestRandomForest = copy.deepcopy(self.randomForest)
                                    bestBuf = pbuf

                                self.randomForest.clear()
                                vote_results.clear()
                                del cv_results[:]



        # Cross validation completed
        #
        # stick the best parameters from cross validation back to this tree and 
        # return the trained tree for use and testing
        self.randomForest.clear()
        self.randomForest = bestRandomForest
                    
        # Write the best tree from cross validation to be used
        sbuf = "Best Tree from Cross Validation:%s,training accuracy=%.4f" % (bestBuf, bestAccuracy ) 

        print(sbuf)

        # Writing results to a file (DO NOT CHANGE)
        f = open(myname+"result-crossvalidation.txt", "ab+")
        f.write(sbuf+"\n")
        f.close()

        # Parameter settings giving best N=10 results in training:cross validation
        f = open(myname+"result-crossvalidation.txt", "ab+")
        
        cvKs = cvResults.keys()
        cvKs.sort(reverse=True)
        for svr in cvKs:
        
            res=cvResults[svr]
            
            sbuf = "Best Training Results:%s,Average Training accuracy: %.4f" % ( res['buf'], svr)
            
            print(sbuf)
            f.write(sbuf+"\n")
            
        f.close()




    # save the decision tree to the disk as json
    def saveToDiskAsJson(self,fileName):
        
        saveDict  = {}
        saveDict['attribs'] = self.attribs
        saveDict['attribType'] = self.attribType
        saveDict['randomForest'] = self.randomForest

        # save the best tree to disk as json
        jSaveDict = dumps(saveDict, cls=PythonObjectEncoder)
        with open(fileName, 'w') as fp:
            json.dump(jSaveDict, fp)
            fp.close()
 

    # Build the decision tree from disk file as json
    def readTreeFromJsonFile(self, fileName):
        with open(fileName, 'r') as fp:
            jSaveDict = json.load(fp)
            saveDict = loads(jSaveDict, object_hook=as_python_object)
            self.attribs = saveDict['attribs']
            self.attribType = saveDict['attribType']
            self.randomForest = saveDict['randomForest']
  
        
    
            
    # default constructor
    def __init__(self):
        self.attribs = []
        self.attribType = {}
        self.nodeList = {}
        self.data = []
        self.randomForest = {}
        self.nmin = 3
        self.m = 1
        self.samplePercent = .1
        self.kTile = 2
        self.newMAttribsForSplit = True
        self.pickMAttribWOReplacement = True
        self.useGini = True
        # constant parameters, these values have been found to give the best performance
        # to save time, no need to cross validate on these
        self.useBinEntropy=False
        self.branchForEntropyGainOnly = True
        self.returnMajorityNotDefault = False
        self.target = 'quality'



    # train the decision tree on the training set passed
    def train ( self, training_data ):

        # randomly choose m attribs with/WO replacement to be used for splitting
        mAttribs = []
        attribKeys = list(set(self.attribs.keys())-set([self.target]))
        
        if self.pickMAttribWOReplacement:
            rIdx = random.sample(xrange(0,len(attribKeys)), int(self.m))
            attribTuple = itemgetter(*rIdx)(attribKeys)
            mAttribs = list(attribTuple)
        else:
            while len(mAttribs) < int(self.m):
                r = random.choice(attribKeys)
                mAttribs.append(r)        
                    
        rootNode = {
            'key':'root',
            'parent':None,
            'type' : 'root',
            'dataIdx': set(range(len(training_data))) , 
            'children': [], 
            'bin':float('-inf'),
            'parentAttrib':None,
            'level':0,
            'chosenAttrib':None,
            'parentKey':None
            }


        self.nodeList['root']=rootNode
                    
        stack = []
        stack.append('root')

        while stack:

            nodeKey = stack.pop(0)

            node = self.nodeList[nodeKey]
            
            # get the stats for the node 
            # max of the data value in the target column of the remaining set
            lv = [self.data[i][self.attribs[self.target]] for i in node['dataIdx']]
            
            if len(lv):
                majorityValue = max(lv,key=lv.count)
                #if majorityValue > 0:
                #    print("attrib=%s,bin=%.4f,Majority Value=%.2f" % (node['parentAttrib'],node['bin'],majorityValue) )
                node['majorityValue'] = majorityValue

            # if we have reached he list of nmin nodes for the random tree
            # then do not branch further
            if len(self.nodeList) <= self.nmin:
                node['type']='leaf'
                # this is a safe step, set all the nodes with no children to leaf
                for nk in self.nodeList.values():
                    if ( nk['type']=='subTree') and (len(nk['children']) == 0 ):
                        nk['type']='leaf'
                continue

            # branching of this node
            etp1 = Entropy(self,node['dataIdx'],self.target,self.target)
                
            # all target values are same
            if ( etp1 == 0 ):
                node['type']='leaf'
                continue
            else:

                # find the best of the remaining attribs to classify the remaining data points on this tree branch
                bestAttrib = ''
                betp2=1.01

                # randomly choose new m attribs to be used for splitting
                # else use the one chosen first for the tree
                if self.newMAttribsForSplit:          
                    # randomly choose m attribs with replacement to be used for splitting
                    mAttribs = []
                    if self.pickMAttribWOReplacement:
                        rIdx = random.sample(xrange(0,len(attribKeys)), int(self.m))
                        attribTuple = itemgetter(*rIdx)(attribKeys)
                        mAttribs = list(attribTuple)
                    else:
                        while len(mAttribs) < int(self.m):
                            r = random.choice(attribKeys)
                            mAttribs.append(r)  
                            
                for bAttrib in mAttribs:
                        
                    etp2 = Entropy(self,node['dataIdx'],self.target,bAttrib)
                    
                    if etp2 < betp2:
                        bept2 = etp2
                        bestAttrib = bAttrib
                            
                    
                # this check not for rootNode
                # if the entropy of the branch is not less then the entropy
                # at this node, then end branching along this node
                if node['type'] != 'root':
                    if self.branchForEntropyGainOnly:
                        if betp2 >= etp1:
                            node['type']='leaf'
                            continue

                # find the values for the chosen attrib to split next
                #  find the data rows divided for each attrib value
                #  add a node for each of them to the stack
                    
                if ( self.attribType[bestAttrib] == 'numeric' ):
                    aVResult = self.getKTiles(self.kTile, node['dataIdx'],bestAttrib,self.target)   
                    aVSet = set( aVResult.keys() )
                else:
                    aVSet = set( self.data[i][self.attribs[bestAttrib]] for i in node['dataIdx'] )
                    
        
                node['chosenAttrib']=bestAttrib
                td=node['children']

                for aVal in aVSet:
                    
                    # get the records with values in that range to be branched to same branch
                    if ( self.attribType[bestAttrib] == 'numeric' ):
                        vIdx = set( aVResult[aVal]['bin_records'] )                            
                    else:
                        vIdx = set( i for i in node['dataIdx'] if self.data[i][self.attribs[bestAttrib]] == aVal )
                         
                    childNodeKey = nodeKey+"--"+bestAttrib+"--"+str(aVal)
                    td.append(childNodeKey)

                    childNode =  {
                        'type':'subTree',
                        'key':childNodeKey, 
                        'dataIdx': vIdx , 
                        'bin':aVal,
                        'children': [],
                        'parentAttrib':bestAttrib,
                        'parentKey': nodeKey,
                        'level': node['level']+1,
                        'chosenAttrib':None}

                    self.nodeList[childNodeKey]=childNode
                    
                    stack.append(childNodeKey)
                         



    # get average classification accuracy with the cv_set, based on a vote of random forests
    def getCVAccuracy(self,cv_set):

        # cross validate with the cv_set to get the accuracy of this tree
        # if it is better than the previous best save it in best tree
        # CROSS Validation
        # Classify the cross validation set using the tree we just constructed
        cv_results = []

        for instance in cv_set:
            
            cvrs = self.classify(instance)
            
            # take the highest vote
            cv_results.append( cvrs == instance[-1] )
 
        # Cross validation - Training Accuracy
        cv_accuracy = float(cv_results.count(True))/float(len(cv_results))

        return cv_accuracy




    # get classification for the single record based on a vote of random forests
    def classify(self,test_instance):

        savedNodeList = self.nodeList
        
        voteCount = {}
            
        # take avote of the random forests
        for rf in self.randomForest:

            self.nodeList = self.randomForest[rf]['nodeList']
            cv_result = self.getRandomForestVote( test_instance[:-1] )

            if cv_result in voteCount:
                voteCount[cv_result] += 1
            else:
                voteCount[cv_result] = 1

        self.nodeList = savedNodeList
            
        # take the highest vote
        for cvrs in sorted(voteCount, key=voteCount.get, reverse=True): 
            return cvrs




        

    def printTree(self):

        pStack = []
        pStack.append('root')

        while pStack:

            node = self.nodeList[pStack.pop()]
            
            idC = '\t' * node['level']
            if node['type']=='root':
                print("Root")
            else:
                print("%s%s=%s" % (idC,node['parentAttrib'],node['bin']) )
            
            if node['type'] == 'leaf':
                print("%s%s=%s" % (idC,self.target,node['majorityValue']) )

            for chk in node['children']:
                pStack.append(chk)





    # implement this function
    def getRandomForestVote(self, test_instance):
        result = 0 # baseline: always classifies as 0

        # implement this function
 
        nodeKey = 'root'

        while nodeKey:
                      
            node = self.nodeList[nodeKey]

            # reached the leaf , return the result value
            if node['type'] == 'leaf':
                return node['majorityValue']

            attrib = node['chosenAttrib']
 
            # value of that attrib in test data
            tval = test_instance[self.attribs[attrib]]

            # next decide which branch based on value
            # if numeric attrib, check range
            nodeKey=None
            if self.attribType[attrib] == 'numeric':

                prev=float("-inf")
                lastNK=None

                nkDs = {self.nodeList[cnK]['bin']:cnK for cnK in node['children']}
                nkKs = nkDs.keys()
                nkKs.sort()
                
                for binVal in nkKs:
                    nK = nkDs[binVal]
                    if prev < tval <= binVal:
                        nodeKey = nK
                        break
                    prev=binVal
                    lastNK=nK
 
                # Did not find a bin this value should belong to
                # pick the the largest bin as
                if self.returnMajorityNotDefault and not nodeKey:
                    #nodeKey=lastNK
                    return node['majorityValue']
   
            # not numeric, treat it as categorical data
            else:
                
                for nK,binVal in [(cnK,self.nodeList[cnK]['bin']) for cnK in node['children']]:   
                    if tval.lower() == binVal.lower():
                        nodeKey=nk
                        break
                    
                # Did not find a bin this value should belong to
                # return the MajorityValue of this node as result
                if self.returnMajorityNotDefault and not nodeKey:
                    return node['majorityValue'] 
             

        return result
    





#---------------------------------------------------------------



# compute entropy
def Entropy(tree,dataIdx,resultParam,decisionParam):

    #get the count of type of attrib1 records in tree
    decisionCount = {}

    # position of decision and result elements in data list
    r = tree.attribs[resultParam]
    d = tree.attribs[decisionParam]

    # if it is numeric get a count of the actual bins we might classify for 
    # kTile if useBinEntropy is true
    if ( tree.useBinEntropy and tree.attribType[decisionParam] == 'numeric' ):

        aVResult = tree.getKTiles(tree.kTile, dataIdx,decisionParam,resultParam)  

        aVSet = set( aVResult.keys() )

        for aVal in aVSet:
            #for each bin the total counts in them
            decisionCount[aVal]={}
            decisionCount[aVal]['totalxx']=len(aVResult[aVal]['bin_records'])

            # for each bin the count of different values
            for rVal in aVResult[aVal]['bin_records_result_count'].keys():
                decisionCount[aVal][rVal]=aVResult[aVal]['bin_records_result_count'][rVal]     


    else:
        for i in dataIdx:

            e = tree.data[i]

            # count records with different decisions
            if e[d] in decisionCount:
                decisionCount[e[d]]['totalxx'] += 1
            else:
                decisionCount[e[d]] = {}
                decisionCount[e[d]]['totalxx'] = 1

                # count records with different results in a decision
                if e[r] in decisionCount[e[d]]:
                    decisionCount[e[d]][e[r]] += 1
                else:
                    decisionCount[e[d]][e[r]] = 1




    tot=len(dataIdx)
    en=float(0)

    if not tree.useGini:

        # calculate entropy
        for dc in decisionCount:
            dct = decisionCount[dc]['totalxx']
            for rc in decisionCount[dc]:
                if rc != 'totalxx':
                    rct = decisionCount[dc][rc]
                    if resultParam == decisionParam:
                        en += -(rct * (math.log(rct,2)-math.log(tot,2)) )/tot
                    else:
                        en += -(dct*rct * (math.log(rct,2)-math.log(dct,2)) )/(dct*tot)

    else:

        # calculate gini
        if ( resultParam == decisionParam ):

            en = 1.0

            for dc in decisionCount:
                dct = decisionCount[dc]['totalxx']
                for rc in decisionCount[dc]:
                    if rc != 'totalxx':
                        rct = decisionCount[dc][rc]
                        en = en-math.pow(float(rct)/float(tot),2)


        else:

            for dc in decisionCount:
                dct = decisionCount[dc]['totalxx']
                en += float(dct)/float(tot)
                for rc in decisionCount[dc]:
                    if rc != 'totalxx':
                        rct = decisionCount[dc][rc]
                        en = en-(float(dct)*math.pow(float(rct)/float(dct),2))/(tot)



    return en



# convert from str to float
def strtof(st):
    try:
        return float(st)
    except ValueError:
        return st
    


# get the most commonly occuring element in the list
def common_val(lst):
    return max(set(lst),key=lst.count)



# get the most commonly occuring element in the list
def fill_common_val(data,cVal,i,j):
       data[i][j] = cVal
        


def run_decision_tree(useSavedDecisonTree,fileName):

    # Load data set
    with open("hw4-data.csv") as f:
    #with open("test.csv") as f:
        reader = csv.reader(f, delimiter=",")
        # read title into dict {colheader:i}
        title = { e:i for i,e in enumerate(next(reader))}
        # read all data, convert it to float where possible
        data = [map(strtof,line) for line in reader]

    dln = len(data)
    print "Number of records: %d" % dln


    # json file name to be used if one is not passed for saving or
    # reading a saved decision tree from
    if not fileName:
        fileName = "decision_tree.json"


    # PREPROCESS
    #
    # get the common values and fill blank vals with common values
    # fill empty values with majority values
    cVals = [ common_val([i[j] for i in data]) for j in range(len(title)) ]
   
    [ fill_common_val(data,cVals[j],i,j) for i in range(len(data)) for j in range(len(title)) if ( isinstance(data[i][j],str) and not data[i][j] ) ]


    # Split training/test sets
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]
    
    tree = DecisionTree()
    #tree = RandomForest()

    # the trained decision tree is saved as decision_tree.json
    #  read this decision tree and use it directly for classification
    if  useSavedDecisonTree :
    
        print("Reading the decision tree from disk based json file")
        # read the saved tree and use it to classify
        tree.readTreeFromJsonFile(fileName)

    else:     
        # Construct a tree using training set
        #
        #  The approach is to grow the tree fully and thej prune it 
        #   based on the Cost Complexity Approach
        #
        #  Dynamic binning is performed at each branch using Ktile
        #
        #  The atrribute for branching is selected based on the one which
        #   gives the highest entropy gain
        #
        #  MajorityValue is returned and branching stopped and a node is a leaf when
        #   - All values for quality are same at a node Or
        #   - No more attributes are left to branch at a node
        #   - Or there are no more records left to branch at a node
        #
        #  For pruning the decision to prune a subtree at a intermediate node is made 
        #  if  classification error from majority value at a node
        #   is lower than the classification errors from the leaves in its subtree + 
        #     costComplexityFactor * number of leaves in the subtree
        #
        # 10 fld cross validation is performed to tune the
        #  tree Ktile binning parameter, and to
        #  choose a ComplexityCost factor for pruning the 
        #   tree
        #
        tree.learn( training_set, title)

        # ----------------------------------------------
        # Save the decision tree to disk and read it from 
        #  disj and verify classification accuracy
        # ---------------------------------------------
        print("Saving the decision tree to disk as json")
        # save decision tree to disk
        tree.saveToDiskAsJson(fileName)


    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy: %.4f" % accuracy       
    
    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


     
if __name__ == "__main__":

    if len(sys.argv) > 1:
        if len(sys.argv) > 2 :
            run_decision_tree(True,sys.argv[2])
        else:
            run_decision_tree(True,None)
    else:
        run_decision_tree(False,None)
 
