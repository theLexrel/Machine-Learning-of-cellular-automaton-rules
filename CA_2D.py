import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import  make_scorer, brier_score_loss
# ML algorithm
from sklearn.linear_model import LogisticRegression

import csv
from numba import jit


@jit()
def produceTimeSeriesNeumann(nRows,nCols,nOnes,iter,updateArray):
    timeSeries = np.zeros((iter+1,nRows,nCols), dtype = np.bool_)   
    randRow = np.random.rand(nOnes)*nRows
    randCol = np.random.rand(nOnes)*nCols
    for i in range(nOnes):
        if(timeSeries[0,int(randRow[i]),int(randCol[i])]):
            newRow = int(np.random.rand()*nRows)
            newCol = int(np.random.rand()*nCols)
            timeSeries[0,newRow,newCol] = True
        else:
            timeSeries[0,int(randRow[i]),int(randCol[i])] = True
    for t in range(0,iter):
        for row in range(0,nRows):
            for col in range(0,nCols):
                index = timeSeries[t,row,col]*16 + timeSeries[t,(row+1)%nRows,col]*8 + timeSeries[t,row,(col+1)%nCols]*4 + timeSeries[t,(row-1)%nRows,col]*2 + timeSeries[t,row,(col-1)%nCols]
                if(updateArray[index]==1):
                    timeSeries[t+1,row,col] = True
                elif(updateArray[index]==0):
                    timeSeries[t+1,row,col] = False
                else:
                    rand = np.random.rand()
                    if(rand < updateArray[index]):
                        timeSeries[t+1,row,col] = True
                    else:
                        timeSeries[t+1,row,col] = False
    return timeSeries

@jit()
def retrieveNeighborhoodOutput(timeSeries,num,splits=0):
    # this function is entended to turn the time series matrix back into \vec{x} and y
    nSteps, nRows, nCols = np.shape(timeSeries)
    iter = nSteps-1
    vecX = np.zeros((iter*nCols*nRows,num+1)) # all cells that were updated + neighborhood
    vecY = np.zeros(iter*nCols*nRows)
    for series in range(splits+1):
        start = int(nSteps/(splits+1))*series
        end = int(nSteps/(splits+1))*(series+1)-1
        for t in range(start,end):
            for k in range(0,nRows):
                for i in range(0,nCols):
                    if(num==4):
                        vecX[t*nCols*nRows+k*nCols+i,0] =  timeSeries[t,k,i]
                        vecX[t*nCols*nRows+k*nCols+i,1] = timeSeries[t,(k+1)%nRows,i]
                        vecX[t*nCols*nRows+k*nCols+i,2]= timeSeries[t,k,(i+1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,3]= timeSeries[t,(k-1)%nRows,i]
                        vecX[t*nCols*nRows+k*nCols+i,4]= timeSeries[t,k,(i-1)%nCols]
                        vecY[t*nCols*nRows+k*nCols+i] =  timeSeries[t+1,k,i]
                    else:
                        vecX[t*nCols*nRows+k*nCols+i,0] =  timeSeries[t,k,i]
                        vecX[t*nCols*nRows+k*nCols+i,1] = timeSeries[t,(k)%nRows,(i+1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,2] = timeSeries[t,(k+1)%nRows,(i+1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,3] = timeSeries[t,(k+1)%nRows,(i)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,4] = timeSeries[t,(k+1)%nRows,(i-1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,5] = timeSeries[t,(k)%nRows,(i-1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,6] = timeSeries[t,(k-1)%nRows,(i-1)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,7] = timeSeries[t,(k-1)%nRows,(i)%nCols]
                        vecX[t*nCols*nRows+k*nCols+i,8] = timeSeries[t,(k-1)%nRows,(i+1)%nCols]
                        vecY[t*nCols*nRows+k*nCols+i] =  timeSeries[t+1,k,i]
    return vecX, vecY

@jit()
def Taylor_Neumann(x):
    input = np.zeros((len(x),31))
    for i in range(len(x)):
        # single
        input[i,0] =  int(x[i,0])
        input[i,1] =  int(x[i,1])
        input[i,2] = int(x[i,2])
        input[i,3] = int(x[i,3])
        input[i,4] = int(x[i,4])
        # double
        input[i,5] = int(x[i,0])*int(x[i,1])
        input[i,6] = int(x[i,0])*int(x[i,2])
        input[i,7] = int(x[i,0])*int(x[i,3])
        input[i,8] = int(x[i,0])*int(x[i,4])

        input[i,9] = int(x[i,1])*int(x[i,2])
        input[i,10] = int(x[i,1])*int(x[i,3])
        input[i,11] = int(x[i,1])*int(x[i,4])
        input[i,12] = int(x[i,2])*int(x[i,3])
        input[i,13] = int(x[i,2])*int(x[i,4])

        input[i,14] = int(x[i,3])*int(x[i,4])
        # triple
        input[i,15] = int(x[i,0])*int(x[i,1])*int(x[i,2])
        input[i,16] = int(x[i,0])*int(x[i,1])*int(x[i,3])
        input[i,17] = int(x[i,0])*int(x[i,1])*int(x[i,4])

        input[i,18] = int(x[i,0])*int(x[i,2])*int(x[i,3])
        input[i,19] = int(x[i,0])*int(x[i,2])*int(x[i,4])

        input[i,20] = int(x[i,0])*int(x[i,3])*int(x[i,4])


        input[i,21] = int(x[i,1])*int(x[i,2])*int(x[i,3])
        input[i,22] = int(x[i,1])*int(x[i,2])*int(x[i,4])

        input[i,23] = int(x[i,1])*int(x[i,3])*int(x[i,4])


        input[i,24] = int(x[i,2])*int(x[i,3])*int(x[i,4])
        # quatruple
        input[i,25] = int(x[i,0])*int(x[i,1])*int(x[i,2])*int(x[i,3])
        input[i,26] = int(x[i,0])*int(x[i,1])*int(x[i,2])*int(x[i,4])
        input[i,27] = int(x[i,0])*int(x[i,1])*int(x[i,3])*int(x[i,4])
        input[i,28] = int(x[i,0])*int(x[i,2])*int(x[i,3])*int(x[i,4])
        input[i,29] = int(x[i,1])*int(x[i,2])*int(x[i,3])*int(x[i,4])
        # quintable
        input[i,30] = int(x[i,0])*int(x[i,1])*int(x[i,2])*int(x[i,3])*int(x[i,4])
        
    return input


@jit()
def produceTimeSeriesMoore(nRows,nCols,nOnes,iter,updateArray):
    timeSeries = np.zeros((iter+1,nRows,nCols), dtype = np.bool_)   
    randRow = np.random.rand(nOnes)*nRows
    randCol = np.random.rand(nOnes)*nCols
    for i in range(nOnes):
        if(timeSeries[0,int(randRow[i]),int(randCol[i])]):
            newRow = int(np.random.rand()*nRows)
            newCol = int(np.random.rand()*nCols)
            timeSeries[0,newRow,newCol] = True
        else:
            timeSeries[0,int(randRow[i]),int(randCol[i])] = True
    for t in range(0,iter):
        for row in range(0,nRows):
            for col in range(0,nCols):
                index = timeSeries[t,row,col]*256 + timeSeries[t,row,(col+1)%nCols]*128 + timeSeries[t,(row+1)%nRows,(col+1)%nCols]*64 + timeSeries[t,(row+1)%nRows,(col)%nCols]*32 + timeSeries[t,(row+1)%nRows,(col-1)%nCols]*16 +timeSeries[t,(row)%nRows,(col-1)%nCols]*8 + timeSeries[t,(row-1)%nRows,(col-1)%nCols]*4 + timeSeries[t,(row-1)%nRows,(col)%nCols]*2 + timeSeries[t,(row-1)%nRows,(col+1)%nCols]
                if(updateArray[index]==1):
                    timeSeries[t+1,row,col] = True
                elif(updateArray[index]==0):
                    timeSeries[t+1,row,col] = False
                else:
                    rand = np.random.rand()
                    if(rand < updateArray[index]):
                        timeSeries[t+1,row,col] = True
                    else:
                        timeSeries[t+1,row,col] = False
    return timeSeries

@jit()
def Taylor_Moore(x):
    input = np.zeros((len(x),511))
    offset = [9,45,129,255,381,465,501,510]   
    for i in range(len(x)):
        counter = [0,0,0,0,0,0,0,0]
        for m in range(9):
            input[i,counter[0]] = np.int8(x[i,m])
            counter[0] += 1
            for c in range(m+1,9):
                input[i,offset[0]+counter[1]] = np.int8(x[i,m]*x[i,c])
                counter[1] += 1
                for d in range(c+1,9):
                    input[i,offset[1]+counter[2]] = np.int8(x[i,m]*x[i,c]*x[i,d])
                    counter[2] += 1
                    for e in range(d+1,9):
                        input[i,offset[2]+counter[3]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e])
                        counter[3] += 1
                        for f in range(e+1,9):
                            input[i,offset[3]+counter[4]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e]*x[i,f])
                            counter[4] += 1
                            for g in range(f+1,9):
                                input[i,offset[4]+counter[5]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e]*x[i,f]*x[i,g])
                                counter[5] += 1
                                for h in range(g+1,9):
                                    input[i,offset[5]+counter[6]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e]*x[i,f]*x[i,g]*x[i,h])
                                    counter[6] += 1
                                    for j in range(h+1,9):
                                        input[i,offset[6]+counter[7]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e]*x[i,f]*x[i,g]*x[i,h]*x[i,j])
                                        counter[7] += 1
                                        for k in range(j+1,9):
                                            input[i,offset[7]] = np.int8(x[i,m]*x[i,c]*x[i,d]*x[i,e]*x[i,f]*x[i,g]*x[i,h]*x[i,j]*x[i,k])     
    return input

def neighborConfigs(num):
    '''Gives out all possible input configs'''
    end = 2**num
    a = np.arange(0,end,dtype=np.uint8)
    b = np.unpackbits(a)
    b = np.reshape(b,(len(a),int(len(b)/len(a))))
    if(num==4):
         b = np.hsplit(b,2)[1]
    b_false = np.insert(b,0,0,axis=1)
    b_true = np.insert(b,0,1,axis=1)
    neigh = np.vstack((b_false,b_true))
    return neigh


def buildGoLArray():
    configs = neighborConfigs(8)
    update = np.zeros(len(configs))
    
    for i in range(len(update)):
        currentValue = configs[i,0]
        neighbor = configs[i,1:]
        if(currentValue and (np.sum(neighbor)<2 or np.sum(neighbor)>3)):
            update[i] = 0
        elif(currentValue and (np.sum(neighbor)==2 or np.sum(neighbor)==3)):
            update[i] = 1
        elif(not currentValue and np.sum(neighbor)==3):
            update[i] = 1
        else:
            update[i] = int(currentValue)
    return update

def BuildSpreadOfSeeds(num):
    configs = neighborConfigs(num)
    update = np.zeros(len(configs))
    
    for i in range(len(update)):
        currentValue = configs[i,0]
        neighbor = configs[i,1:]
        if(currentValue):
            update[i]=1
        else:
            update[i] = np.sum(neighbor)/num
    return update

def BuildBirthDeath(num,alpha=1.2,beta=0.6):
    configs = neighborConfigs(num)
    update = np.zeros(len(configs))
    for i in range(len(update)):
        currentValue = configs[i,0]
        neighbor = configs[i,1:]
        if(currentValue):
            update[i] = np.exp(-alpha*np.sum(neighbor))
        else:
            update[i] = 1-np.exp(-beta*np.sum(neighbor))

    return update


@jit()
def BuildHistogram(x,y,num):
    counter = np.zeros((2**(num+1),2))
    for i in range(len(x)):
        if(num==4):
            index = int(x[i,0])*16 + int(x[i,1])*8 + int(x[i,2])*4 + int(x[i,3])*2 + int(x[i,4])
        else:
            index= int(x[i,0]*256 + x[i,1]*128 + x[i,2]*64 + x[i,3]*32 + x[i,4]*16 + x[i,5]*8 +x[i,6]*4 + x[i,7]*2 + x[i,8])
        row = int(y[i])
        counter[index,row] += 1
    return counter

@jit()
def HosmerLemeshow_test(timeSeries, p_predict, num, splits=0):
    X,Y = retrieveNeighborhoodOutput(timeSeries,num,splits=splits)
    observed = BuildHistogram(X,Y,num)
    for i in range(len(observed)):
        sum = observed[i,0] + observed[i,1]
        if(sum):
            observed[i,0] = observed[i,0]/sum
            observed[i,1] = observed[i,1]/sum
    hl = 0
    for i in range(len(observed)):
        if(p_predict[i,0] and p_predict[i,1]):
            hl += (observed[i,0]-p_predict[i,0])**2/p_predict[i,0]+(observed[i,1]-p_predict[i,1])**2/p_predict[i,1]
        elif(not p_predict[i,0] and p_predict[i,1]):
            hl += observed[i,0] + (observed[i,1]-p_predict[i,1])**2/p_predict[i,1]
        else:
            hl += (observed[i,0]-p_predict[i,0])**2/p_predict[i,0] + observed[i,1]
    hl = hl/len(observed)
    return hl

@jit()
def brier_error(y_true,y_pred):
    # implementation of brier error that doen't return a negative value when used in GridSearch of cross validation
    return -np.abs(np.sum(np.square(y_true-y_pred))/len(y_true))

@jit()
def _changeRate(timeSeries):
    iter, nRows, nCols = np.shape(timeSeries)
    vec = np.zeros(iter)
    for i in range(iter-1):
        for t in range(nRows):
            for o in range(nCols):
                if(not(timeSeries[i+1,t,o]==timeSeries[i,t,o])):
                    vec[i] +=1
    vec = vec/(nCols*nRows)
    return vec
@jit()
def _countStates(timeSeries,num):
    seriesLength, nRows, nCols = np.shape(timeSeries)
    nCells = num+1
    nStates = 2**nCells
    stateCount = np.zeros((seriesLength,nStates))
    index = 0
    for t in range(seriesLength):
        for row in range(nRows):
            for col in range(nCols):
                currentState = timeSeries[t,row,col]
                if(num==4):
                    neighbors = [timeSeries[t,(row+1)%nRows,col], timeSeries[t,row,(col+1)%nCols], timeSeries[t,(row-1)%nRows,col], timeSeries[t,row,(col-1)%nCols]]
                else:
                    neighbors = [timeSeries[t,(row+1)%nRows,col],timeSeries[t,(row+1)%nRows,(col+1)%nCols], timeSeries[t,row,(col+1)%nCols], timeSeries[t,(row-1)%nRows,(col+1)%nCols],timeSeries[t,(row-1)%nRows,col], timeSeries[t,(row-1)%nRows,(col-1)%nCols], timeSeries[t,row,(col-1)%nCols], timeSeries[t,(row+1)%nRows,(col-1)%nCols]]
                if(currentState):
                    index += 2**(nCells-1)
                for q in range(nCells-1):
                    if(neighbors[q]):
                        index += 2**(nCells -2 -q)
                stateCount[t,index] += 1
                index = 0
    return stateCount
@jit()
def _spatialEntropy(timeSeries):
    #entropy of spatial clustering, uses the probabiliy of finding a cluster size n
    seriesLength, nRows, nCols = np.shape(timeSeries)
    maxCluster = nRows*nCols
    prob = np.zeros((seriesLength,maxCluster)) #prob at time t for size n: prob[t,n]
    entropy = np.zeros(seriesLength)
    def mySorting(e):
        return e[0]
    def swap(q):
        if(q[0]<q[1]):
            return [q[0],q[1]]
        else:
            return[q[1],q[0]]
    for t in range( seriesLength):
        #print('check 1c%i'%t)
        max = np.count_nonzero( timeSeries[t,:,:]) # counts the number of occupied nodes 
        if(max == maxCluster):     # if all nodes occupied, no need to go through loops below
            entropy[t] = 0          # since ln(1) = 0
            print("fully occupied")
        else:
            foundLocation = []
            for i in range( nRows):
                if(np.count_nonzero( timeSeries[t,i,:])== nCols): #check if whole row is occupied
                    foundLocation.append([i,0, nCols-1])
                else:
                    wrap = False        # check if cluster wraps around due to periodic boundary condition 
                    if( timeSeries[t,i,0] and  timeSeries[t,i, nCols-1]):
                        wrap = True
                    tracker = 0
                    cluster = 0
                    while(tracker <  nCols):
                        if(wrap):  # skip the first cluster as its incomplete and will be picked up later 
                            while( timeSeries[t,i,tracker]):
                                tracker += 1
                            wrap=False # has to be set to False again because loop only for the 1st cluster
                        while( timeSeries[t,i,(tracker)% nCols]):
                            cluster += 1
                            tracker += 1
                        if(cluster):
                            foundLocation.append([i,tracker-cluster,tracker %  nCols]) #row,start,end
                        cluster = 0
                        tracker += 1
            #print('check 1c%ia'%t)
            foundLocation.sort(key=mySorting)       # sorting by rows
            connection =  []
            for h in range(len(foundLocation)):
                #print('check 1c%ia%i'%(t,h))
                newIndex = (h +1 ) %len(foundLocation) 
                senityCheck = 0
                while(((foundLocation[newIndex][0]-foundLocation[h][0]) %  nRows) < 2): 
                    #print('check 1c%ia%i%i'%(t,h,senityCheck))
                    if(senityCheck == len(foundLocation)):
                        break
                    if (foundLocation[newIndex][0] == foundLocation[h][0]+1):
                        if(foundLocation[newIndex][1] >= foundLocation[h][1] and foundLocation[newIndex][1] <= foundLocation[h][2]):
                            connection.append([h,newIndex])
                        elif(foundLocation[newIndex][2] >= foundLocation[h][1] and foundLocation[newIndex][2] <= foundLocation[h][2]):
                            connection.append([h,newIndex])
                    newIndex = (newIndex + 1) % len(foundLocation)
                    senityCheck +=1
            connection.sort(key=swap)
            connection.sort(key=mySorting)
            #print('check 1c%ib'%t)
            for p in range(len(connection)):
                newP = p+1
                if(newP >= len(connection)):
                    continue
                while(connection[newP][0] <= connection[p][-1]):
                    popped = False
                    if(connection[newP][0] in connection[p]):
                        connection[p].extend([connection[newP][1]])
                        connection.pop(newP)
                        popped = True
                    elif(connection[newP][1] in connection[p]):
                        connection[p].extend([connection[newP][0]])
                        connection.pop(newP)
                        popped = True
                    elif(connection[newP][0] in connection[p] and connection[newP][1] in connection[p]):
                        connection.pop(newP)
                        popped = True
                    if(not popped):
                        newP = newP+1
                    if(newP >= len(connection)):
                        break
            #print('check 1c%ic'%t)
            for q in range(len(connection)):
                clusterSize = 0
                for w in range(len(connection[q])):
                    clusterSize += (foundLocation[connection[q][w]][2] - foundLocation[connection[q][w]][1]) %  nCols
                prob[t,clusterSize] += 1
            calc = prob[t,:]*np.arange(0,maxCluster)/maxCluster
            calc = calc[calc>0]     # log(0) not defined, therefor obmitted 
            entropy[t] = np.sum(-calc*np.log(calc))
            #print('check 1c%id'%t)
    return entropy

def neighborhoodDensity(timeSeries,num):
    if(num==4):
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
    else:
        neighbors = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    time, rows, cols = np.shape(timeSeries)
    interArray = np.zeros(time*rows*cols)
    for t in range(time):
        for row in range(rows):
            for col in range(cols):
                for j in range(len(neighbors)):
                    interArray[t*rows*cols+row*cols+col] += timeSeries[t,(row+neighbors[j][0])%rows,(col+neighbors[j][1])%cols]/num
    return np.sum(interArray)/len(interArray)

@jit()
def extractFeatures(timeSeries,num):
    # this function aimes to extract all features of relevance of the 1D cell
    seriesLength, nRows, nCols = np.shape(timeSeries)
    occ = np.sum( timeSeries,axis=(1,2))/( nRows* nCols)
    conserved = np.array_equal(np.copy(occ)/occ[0],np.ones( seriesLength))
    lamb = np.sum(occ)/( seriesLength)
    change =  _changeRate(timeSeries)  
    stats =  _countStates(timeSeries,num)
    configs = neighborConfigs(num)
    densityNeigh = neighborhoodDensity(timeSeries,num)
    tempEntropy = np.array(np.copy(stats),dtype=np.bool_)
    nPossipleStates = 2**(num+1)
    tempEntropy = np.sum(tempEntropy,axis=1)/(nPossipleStates)
    sequenceEntropy = np.sum(stats,axis=0)/( nRows* nCols* seriesLength)
    spatialEntropy =  _spatialEntropy(timeSeries)
    
    features = {
        "occupation" :occ, 
        "lambda" : lamb,
        "changeRate":change, 
        "massConservation": conserved,
        "temporalEntropy":tempEntropy, 
        "sequenceEntropy":sequenceEntropy,
        "spatialEntropy" : spatialEntropy,
        "neighborDensity" : densityNeigh
    }
    return features

@jit()
def arrayNeumannToMoore(vec):
    """ there is a large discrapency between the number of possible configs in the Neumann (32)
        and the Moore (512) neighborhood config. This function will return the equivalent 
        probability array in Moore config to the input in Neumann conifg.
        
        input: vec length 32, update array for von Neumann neighborhood
        output: equivalent of input for Moore neighborhood length 512
        
                                          1                   8  1  2
        method: compare layout conigs  4  0  2   Neumann ->   7  0  3   Moore 
                                          3                   6  5  4
        -> values for conifg array element 2, 4, 6, 8 do not contribute to probability and
           can be calculated beforhand to be added to Neumann configs
           -> indexing: add 2**6, 2**4, 2**2, 2**0=1 in all possible constillations  
    """
    '''
    additions = np.zeros(16)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    id = 8*i + 4*j + 2*k + l
                    additions[id] = 64*i + 16*j + 4*k + l
    '''
    additions = np.array([ 0,  1,  4,  5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85])
    configs = np.array(neighborConfigs(4))
    mooreVec = np.zeros(512)
    for i in range(len(configs)):
        for k in range(len(additions)):
            index = configs[i,0]*256 + configs[i,1]*128 + configs[i,2]*32 + configs[i,3]*8 + configs[i,4]*2 + additions[k]
            mooreVec[index] = vec[i]
    return mooreVec


@jit()
def arrayMooreToNeumann(vec):
    """ there is a large discrapency between the number of possible configs in the Neumann (32)
        and the Moore (512) neighborhood config. This function will return the equivalent 
        probability array in Neumann config to the input in Moore conifg.
        
        input: vec length 512, update array for von Neumann neighborhood
        output: matix 32X2 with [:,0] mean values, [:,1] std deviation 

        obviously, it isn't a 1 to 1 equivalent but the mean value and the std
        
                                          1                   8  1  2
        method: compare layout conigs  4  0  2   Neumann ->   7  0  3   Moore 
                                          3                   6  5  4
        -> values for conifg array element 2, 4, 6, 8 do not contribute to probability and
           can be calculated beforhand to be added to Neumann configs
           -> indexing: add 2**6, 2**4, 2**2, 2**0=1 in all possible constillations  
    """
    '''
    additions = np.zeros(16)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    id = 8*i + 4*j + 2*k + l
                    additions[id] = 64*i + 16*j + 4*k + l
    '''
    additions = np.array([ 0,  1,  4,  5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85])
    configs = np.array(neighborConfigs(4))
    neumannMatrix = np.zeros((32,2))
    array = np.zeros(len(additions))
    for i in range(len(configs)):      
        for k in range(len(additions)):
            index = configs[i,0]*256 + configs[i,1]*128 + configs[i,2]*32 + configs[i,3]*8 + configs[i,4]*2 + additions[k]
            array[k] = vec[index]
        neumannMatrix[i,0] = np.mean(array)
        neumannMatrix[i,1] = np.std(array)
    return neumannMatrix

@jit
def gradeForExtraction(extractedP, trueP):
    """ this function will give out an array containing the % of extracted prob. which are within a 
        certain range of the true value.
        output: % of prob deviating >0.5%,0.5-1%, 1-2%, 2-3%, 3-5%, 5-10%, 10-20%, 20-35%, 35-50%, <50%
    """
    scale = np.zeros(10)
    for i in range(len(trueP)):
        if(trueP[i]):
            dev = np.abs((extractedP[i]-trueP[i])/trueP[i])
        else:
            dev = extractedP[i]
        if(dev < 0.005):
            scale[0] +=1
        elif(dev >0.005 and dev <0.01):
            scale[1] += 1
        elif(dev > 0.01 and dev <0.02):
            scale[2] += 1
        elif(dev > 0.02 and dev < 0.03):
            scale[3] =+1
        elif(dev > 0.03 and dev < 0.05):
            scale[4] += 1
        elif(dev > 0.05 and dev < 0.1):
            scale[5] += 1
        elif(dev > 0.1 and dev < 0.2):
            scale[6] += 1
        elif(dev > 0.2 and dev < 0.35):
            scale[7] += 1
        elif(dev > 0.35 and dev < 0.5):
            scale[8] += 1
        else:
            scale[9] +=1
    scale /= len(trueP)
    return scale

def fileInfo(file,predicted,updateArray,num,accuracyLevel=0):

    configs = neighborConfigs(num)
    if(num==4):
        if(accuracyLevel==1):   
            for i in range(0,4):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"&{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}\\\ \n")
                file.write("\hline\n")
                file.write("extracted&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("true&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f&%1.7f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
        elif(accuracyLevel==0):
            for i in range(0,4):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"&{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}\\\ \n")
                file.write("\hline\n")
                file.write("extracted&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("true&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
        else:
            for i in range(0,4):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"&{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}\\\ \n")
                file.write("\hline\n")
                file.write("extracted&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("true&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
    else:
        if(accuracyLevel==1): 
            for i in range(0,64):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}{configs[8*i,5]}{configs[8*i,6]}{configs[8*i,7]}{configs[8*i,8]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}{configs[8*i+1,5]}{configs[8*i+1,6]}{configs[8*i+1,7]}{configs[8*i+1,8]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}{configs[8*i+2,5]}{configs[8*i+2,6]}{configs[8*i+2,7]}{configs[8*i+2,8]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}{configs[8*i+3,5]}{configs[8*i+3,6]}{configs[8*i+3,7]}{configs[8*i+3,8]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}{configs[8*i+4,5]}{configs[8*i+4,6]}{configs[8*i+4,7]}{configs[8*i+4,8]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}{configs[8*i+5,5]}{configs[8*i+5,6]}{configs[8*i+5,7]}{configs[8*i+5,8]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}{configs[8*i+6,5]}{configs[8*i+6,6]}{configs[8*i+6,7]}{configs[8*i+6,8]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}{configs[8*i+7,5]}{configs[8*i+7,6]}{configs[8*i+7,7]}{configs[8*i+7,8]}\\\ \n")
                file.write("\hline\n")
                file.write("%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
        elif(accuracyLevel==0):
            for i in range(0,64):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}{configs[8*i,5]}{configs[8*i,6]}{configs[8*i,7]}{configs[8*i,8]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}{configs[8*i+1,5]}{configs[8*i+1,6]}{configs[8*i+1,7]}{configs[8*i+1,8]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}{configs[8*i+2,5]}{configs[8*i+2,6]}{configs[8*i+2,7]}{configs[8*i+2,8]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}{configs[8*i+3,5]}{configs[8*i+3,6]}{configs[8*i+3,7]}{configs[8*i+3,8]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}{configs[8*i+4,5]}{configs[8*i+4,6]}{configs[8*i+4,7]}{configs[8*i+4,8]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}{configs[8*i+5,5]}{configs[8*i+5,6]}{configs[8*i+5,7]}{configs[8*i+5,8]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}{configs[8*i+6,5]}{configs[8*i+6,6]}{configs[8*i+6,7]}{configs[8*i+6,8]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}{configs[8*i+7,5]}{configs[8*i+7,6]}{configs[8*i+7,7]}{configs[8*i+7,8]}\\\ \n")
                file.write("\hline\n")
                file.write("%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
        else:
            for i in range(0,64):
                if(i):
                    file.write("\hline\hline\n")
                file.write(f"{configs[8*i,0]}{configs[8*i,1]}{configs[8*i,2]}{configs[8*i,3]}{configs[8*i,4]}{configs[8*i,5]}{configs[8*i,6]}{configs[8*i,7]}{configs[8*i,8]}&{configs[8*i+1,0]}{configs[8*i+1,1]}{configs[8*i+1,2]}{configs[8*i+1,3]}{configs[8*i+1,4]}{configs[8*i+1,5]}{configs[8*i+1,6]}{configs[8*i+1,7]}{configs[8*i+1,8]}&{configs[8*i+2,0]}{configs[8*i+2,1]}{configs[8*i+2,2]}{configs[8*i+2,3]}{configs[8*i+2,4]}{configs[8*i+2,5]}{configs[8*i+2,6]}{configs[8*i+2,7]}{configs[8*i+2,8]}&{configs[8*i+3,0]}{configs[8*i+3,1]}{configs[8*i+3,2]}{configs[8*i+3,3]}{configs[8*i+3,4]}{configs[8*i+3,5]}{configs[8*i+3,6]}{configs[8*i+3,7]}{configs[8*i+3,8]}&{configs[8*i+4,0]}{configs[8*i+4,1]}{configs[8*i+4,2]}{configs[8*i+4,3]}{configs[8*i+4,4]}{configs[8*i+4,5]}{configs[8*i+4,6]}{configs[8*i+4,7]}{configs[8*i+4,8]}&{configs[8*i+5,0]}{configs[8*i+5,1]}{configs[8*i+5,2]}{configs[8*i+5,3]}{configs[8*i+5,4]}{configs[8*i+5,5]}{configs[8*i+5,6]}{configs[8*i+5,7]}{configs[8*i+5,8]}&{configs[8*i+6,0]}{configs[8*i+6,1]}{configs[8*i+6,2]}{configs[8*i+6,3]}{configs[8*i+6,4]}{configs[8*i+6,5]}{configs[8*i+6,6]}{configs[8*i+6,7]}{configs[8*i+6,8]}&{configs[8*i+7,0]}{configs[8*i+7,1]}{configs[8*i+7,2]}{configs[8*i+7,3]}{configs[8*i+7,4]}{configs[8*i+7,5]}{configs[8*i+7,6]}{configs[8*i+7,7]}{configs[8*i+7,8]}\\\ \n")
                file.write("\hline\n")
                file.write("%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f\\\ \n"%(predicted[8*i],predicted[8*i+1],predicted[8*i+2],predicted[8*i+3],predicted[8*i+4],predicted[8*i+5],predicted[8*i+6],predicted[8*i+7]))
                file.write("%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.2f\\\ \n"%(updateArray[8*i],updateArray[8*i+1],updateArray[8*i+2],updateArray[8*i+3],updateArray[8*i+4],updateArray[8*i+5],updateArray[8*i+6],updateArray[8*i+7]))
         
    file.write("\nBreakdown\n")
    gradingN = gradeForExtraction(predicted,updateArray)
    file.write(">0.5\%&0.5-1\%&1-2\%&2-3\%&3-5\%&5-10\%&10-20\%&20-35\%&35-50\%&<50\%\\\\\n")
    file.write("\hline\n")
    file.write("%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f&%1.4f \\\\\n"%(gradingN[0],gradingN[1],gradingN[2],gradingN[3],gradingN[4],gradingN[5],gradingN[6],gradingN[7],gradingN[8],gradingN[9]))

def compareProperties(file,originalFeatures,rebuildFeatures,length):
    file.write("& original & rebuild \\\ \n")
    file.write("\hline\n")
    file.write(f"mean Change rate & {np.sum(originalFeatures['changeRate'])/length} & {np.sum(rebuidFeatures['changeRate'])/length}\\\ \n")
    file.write(f"mean occupancy rate & {np.sum(originalFeatures['occupation'])/(length+1)} & {np.sum(rebuidFeatures['occupation'])/(length+1)}\\\ \n")
    file.write(f"mean neighborhood density & {np.sum(originalFeatures['neighborDensity'])/(length+1)} & {np.sum(rebuidFeatures['neighborDensity'])/(length+1)}\\\ \n")
    file.write(f"mean spatial entropy & {np.sum(originalFeatures['spatialEntropy'])/(length+1)} & {np.sum(rebuidFeatures['spatialEntropy'])/(length+1)}\\\ \n")

def Thesis_proveOfConcept():

    plt.rcParams.update({'font.size': 23})

    fileName = "2Dca.dat"

    brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)

    refParam = {'penalty':'none','solver':'sag','tol':0.001,'max_iter':1000}
    clfNeumann = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])
    clfMoore = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])


    file = open(fileName,'a')
    
    extraFile = open("ammendedRebuild.dat",'a')

    file.write("________________________________________________________\nProve of concept\n*********************************\n\n")

    length_Neumann = 30
    width_Neumann = 50
    iter_Neumann = 59
    ones_Neumann = [1,int(0.25*length_Neumann*width_Neumann),int(0.5*length_Neumann*width_Neumann),int(0.75*length_Neumann*width_Neumann),length_Neumann*width_Neumann-1]


    file.write("von Neumann CA:\n")
    file.write("length: %i\nwidth: %i\niterations: %i\n"%(length_Neumann,width_Neumann,iter_Neumann))
    testLength_Neumann = int(length_Neumann/2)
    testWidth_Neumann = int(width_Neumann/2)
    testIter_Neumann = int(iter_Neumann)

    length_Moore =50
    width_Moore = 80
    iter_Moore = 79
    ones_Moore = [1,int(0.25*length_Moore*width_Moore),int(0.5*length_Moore*width_Moore),int(0.75*length_Moore*width_Moore),length_Moore*width_Moore-1]

    file.write("Moore CA:\n")
    file.write("length: %i\nwidth: %i\niterations: %i\n"%(length_Moore,width_Moore,iter_Moore))
    testLength_Moore = int(length_Moore/2)
    testWidth_Moore = int(width_Moore/2)
    testIter_Moore = int(iter_Moore)


    updateArrayList_Neumann= [BuildSpreadOfSeeds(4)]
    updateArrayList_Moore= [buildGoLArray(), BuildSpreadOfSeeds(8)]
    nameTrans_Neumann = ["Spread of seeds"]
    nameTrans_Moore = ["Game of Life","Spread of seeds"]
    
    
    interactionType = ["deterministic", "physical"]
    
    for num in range(2):
        file.write("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n------------------%s interaction------------------\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n"%interactionType[num])
        
        if(num==0):
            timeSeriesN = produceTimeSeriesNeumann(length_Neumann, width_Neumann, ones_Neumann[0], iter_Neumann, updateArrayList_Neumann[num])
            TestTimeSeriesN = produceTimeSeriesNeumann(testLength_Neumann, testWidth_Neumann, ones_Neumann[0], testIter_Neumann, updateArrayList_Neumann[num])
            if(num==0 or num==1):
                displaySeriesN =  np.copy(timeSeriesN)
            x_train_rawN, y_trainN = retrieveNeighborhoodOutput(timeSeriesN,4)
            x_trainN = Taylor_Neumann(x_train_rawN)
            x_test_rawN, y_testN = retrieveNeighborhoodOutput(TestTimeSeriesN,4)
            x_testN = Taylor_Neumann(x_test_rawN)
            for k in range(1,len(ones_Neumann)):
                addTimeSeriesN = produceTimeSeriesNeumann(length_Neumann, width_Neumann, ones_Neumann[k], iter_Neumann, updateArrayList_Neumann[num])
                if(k==1 and num==2):
                    displaySeriesN =  np.copy(addTimeSeriesN)
                addTestSeriesN = produceTimeSeriesNeumann(testLength_Neumann, testWidth_Neumann, int(ones_Neumann[k]/4), testIter_Neumann, updateArrayList_Neumann[num])
                x_train_raw_addN, y_train_addN = retrieveNeighborhoodOutput(addTimeSeriesN,4)
                x_train_addN = Taylor_Neumann(x_train_raw_addN)
                x_test_raw_addN, y_test_addN = retrieveNeighborhoodOutput(addTestSeriesN,4)
                x_test_addN = Taylor_Neumann(x_test_raw_addN)
                x_trainN = np.concatenate((x_trainN,x_train_addN),axis=0)
                x_testN = np.concatenate((x_testN,x_test_addN),axis=0)
                y_trainN = np.concatenate((y_trainN,y_train_addN))
                y_testN = np.concatenate((y_testN,y_test_addN))
                TestTimeSeriesN= np.concatenate((TestTimeSeriesN,np.copy(addTestSeriesN)))

            fig_1 = plt.figure()
            ax_1 = fig_1.add_subplot(1,1,1)
            ax_1.set_title("density plots for interaction %s "% (nameTrans_Neumann[num]))
            ax_1.set_xlabel("x")
            ax_1.set_ylabel("y")
            originalDensityPlotN = np.sum(displaySeriesN, axis=0)/(int(iter_Neumann))
            ax_1.pcolor(originalDensityPlotN,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)

            
            clfNeumann.fit(x_trainN,y_trainN)
            crossValN = cross_val_score(clfNeumann, x_testN,y_testN,scoring=brier_score)
            expectArrayN = clfNeumann.predict_proba(Taylor_Neumann(np.array(neighborConfigs(4))))
            predictN = expectArrayN[:,1]
            file.write("_________________________\n------ von Neumann neighborhood ------\n____________________________\n")
            file.write("---Transition: %s \n" % nameTrans_Neumann[num])
            file.write("brier score: %s \pm %s \n" % (crossValN.mean(),crossValN.std()))
            file.write("Hosmer-Lemeshow test: %s\n" % HosmerLemeshow_test(np.copy(TestTimeSeriesN),expectArrayN,4,splits=len(ones_Neumann)-1))
            fileInfo(file,predictN,updateArrayList_Neumann[num],4,accuracyLevel=1)
            
        timeSeriesM = produceTimeSeriesMoore(length_Moore, width_Moore, ones_Moore[0], iter_Moore, updateArrayList_Moore[num])
        TesttimeSeriesM = produceTimeSeriesMoore(testLength_Moore, testWidth_Moore, ones_Moore[0], testIter_Moore, updateArrayList_Moore[num])
        if(num==1):
            displaySeriesM =  np.copy(timeSeriesM)
        x_train_rawM, y_trainM = retrieveNeighborhoodOutput(timeSeriesM,8)
        x_trainM = Taylor_Moore(x_train_rawM)
        x_test_rawM, y_testM = retrieveNeighborhoodOutput(TesttimeSeriesM,8)
        x_testM = Taylor_Moore(x_test_rawM)
        for k in range(1,len(ones_Moore)):
            addtimeSeriesM = produceTimeSeriesMoore(length_Moore, width_Moore, ones_Moore[k], iter_Moore, updateArrayList_Moore[num])
            if(num==0 and k==2):
                displaySeriesM =  np.copy(addtimeSeriesM)
            if(num==2 and k==1):
                displaySeriesM =  np.copy(addtimeSeriesM)
            addTestSeriesM = produceTimeSeriesMoore(testLength_Moore, testWidth_Moore, int(ones_Moore[k]/4), testIter_Moore, updateArrayList_Moore[num])
            x_train_raw_addM, y_train_addM = retrieveNeighborhoodOutput(addtimeSeriesM,8)
            x_train_addM = Taylor_Moore(x_train_raw_addM)
            x_test_raw_addM, y_test_addM = retrieveNeighborhoodOutput(addTestSeriesM,8)
            x_test_addM = Taylor_Moore(x_test_raw_addM)
            x_trainM = np.concatenate((x_trainM,x_train_addM),axis=0)
            x_testM = np.concatenate((x_testM,x_test_addM),axis=0)
            y_trainM = np.concatenate((y_trainM,y_train_addM))
            y_testM = np.concatenate((y_testM,y_test_addM))
            TesttimeSeriesM= np.concatenate((TesttimeSeriesM,np.copy(addTestSeriesM)))

        fig_2 = plt.figure()      
        ax_2 = fig_2.add_subplot(1,1,1)
        ax_2.set_title("density plots for interaction %s "% (nameTrans_Moore[num]))
        ax_2.set_xlabel("x")
        ax_2.set_ylabel("y")
        originalDensityPlotM = np.sum(displaySeriesM, axis=0)/(int(iter_Moore))
        ax_2.pcolor(originalDensityPlotM,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)

    
        clfMoore.fit(x_trainM,y_trainM)
        crossValM = cross_val_score(clfMoore, x_testM,y_testM,scoring=brier_score)
        expectArrayM = clfMoore.predict_proba(Taylor_Moore(np.array(neighborConfigs(8))))
        predictM = expectArrayM[:,1]
        hl = HosmerLemeshow_test(np.copy(TesttimeSeriesM),expectArrayM,num=8,splits=len(ones_Moore)-1)
        file.write("_________________________\n------ Moore neighborhood ------\n____________________________\n")
        file.write("---Transition: %s \n" % nameTrans_Moore[num])
        file.write("brier score: %s \pm %s \n" % (crossValM.mean(),crossValM.std()))
        file.write("Hosmer-Lemeshow test: %s \n" % hl)
        fileInfo(file,predictM,updateArrayList_Moore[num],8,accuracyLevel=1-num)
        
        del timeSeriesN
        del TestTimeSeriesN
        del x_testN
        del x_trainN
        del x_test_rawN
        del x_train_rawN
        del y_testN
        del y_trainN

        del timeSeriesM
        del TesttimeSeriesM
        del x_testM
        del x_trainM
        del x_test_rawM
        del x_train_rawM
        del y_testM
        del y_trainM

    file.close()
    extraFile.close()
    plt.show()


def Thesis_real_correctNeighborhood():

    plt.rcParams.update({'font.size': 23})

    fileName = "2Dca.dat"

    refParam = {'penalty':'none','solver':'sag','tol':0.001,'max_iter':1000}
    clfNeumann = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])
    clfMoore = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])

    file = open(fileName,'a')
    
    extraFile = open("ammendedRebuild.dat",'a')


    file.write("________________________________________________________\nReal: known neighborhood\n*********************************\n\n")

    rebuildLength = 40
    rebuildWidth = 150
    rebuildIter = 199
    rebuildOnes = 1

    
    transitionsNeumann = BuildBirthDeath(4,alpha=0.4,beta=0.3)
    transitionsMoore = BuildBirthDeath(8,alpha=0.25,beta=0.17)
    nameTrans  ="stochastic interaction"

    dataFileNameNeumann = ["b&dNeumann_alpha=0.4_beta=0.3.dat"]
    dataFileNameMoore = ["b&dMoore_alpha=0.25_beta=0.17.dat"]

    file.write("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n------------------%s interaction------------------\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n"%nameTrans)

    originalMatrixN = produceTimeSeriesNeumann(rebuildLength, rebuildWidth, rebuildOnes, rebuildIter, transitionsNeumann)
    
    x_data_rawN, y_dataN = retrieveNeighborhoodOutput(originalMatrixN,4)
    x_dataN = Taylor_Neumann(x_data_rawN)
    timeIndexN = TimeSeriesSplit(4)
    for trainIndex, testIndex  in timeIndexN.split(x_dataN):
        x_trainN, x_testN = x_dataN[trainIndex], x_dataN[testIndex]
        y_trainN, y_testN = y_dataN[trainIndex], y_dataN[testIndex]
    clfNeumann.fit(x_trainN,y_trainN)
    expectArrayN = clfNeumann.predict_proba(Taylor_Neumann(np.array(neighborConfigs(4))))
    predictN = expectArrayN[:,1]
    rebuildMatrixN = np.zeros(np.shape(originalMatrixN),dtype=np.bool_)
    rebuildMatrixN[0,:,:] = np.copy(originalMatrixN[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = rebuildMatrixN[t,row,col]*16 + rebuildMatrixN[t,(row+1)%rebuildLength,col]*8 + rebuildMatrixN[t,row,(col+1)%rebuildWidth]*4  + rebuildMatrixN[t,(row-1)%rebuildLength,col]*2 + rebuildMatrixN[t,row,(col-1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < predictN[index]):
                    rebuildMatrixN[t+1,row,col] = True
                else:
                    rebuildMatrixN[t+1,row,col] = False 
    originalFeaturesN = extractFeatures(originalMatrixN,4)
    rebuidFeaturesN = extractFeatures(rebuildMatrixN,4)

    rebuildDataFileNeumannName = "rebuild_" + dataFileNameNeumann
    datFN = open(dataFileNameNeumann,'a')
    re_datFN = open(rebuildDataFileNeumannName,'a')
    re_datFN.write("intersection: %s \n coeficients:\n" % clfNeumann.intercept_ )
    for i in range(int(len(clfNeumann.coef_)/16)):
        for e in range(16):
            re_datFN.write("\t%s"%clfNeumann.coef_[0,16*i+e])
        re_datFN.write("\n")
    re_datFN.write("\n in array format\n%s\n"%clfNeumann.coef_)
    for t in range(rebuildIter):
        datFN.write("t = %i\n" % t)
        re_datFN.write("t = %i\n" % t)
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                datFN.write("%i\t" % originalMatrixN[t,row,col])
                re_datFN.write("%i\t" % rebuildMatrixN[t,row,col])
            datFN.write("\n")
            re_datFN.write("\n")
    datFN.write("\narray:\n%s" % originalMatrixN)
    re_datFN.write("\narray:\n%s" % rebuildMatrixN)
    datFN.close()
    re_datFN.close()
    with open("NvsN.csv","w+") as my_csv:
        newarray = csv.writer(my_csv,delimiter=',')
        newarray.writerows(originalMatrixN)

    file.write("_________________________\n------ von Neumann neighborhood ------\n____________________________\n")
    
    file.write("brier score: %s \n"% brier_score_loss(y_testN,clfNeumann.predict_proba(x_testN)[:,1]))
    file.write("Hosmer-Lemeshow test: %s \n" % HosmerLemeshow_test(np.copy(originalMatrixN),expectArrayN,4))
    compareProperties(file,originalFeaturesN,rebuidFeaturesN,rebuildIter)

    fileInfo(file,predictN,transitionsNeumann,4,accuracyLevel=2)
    file.write("exact value for transition from empty config: %s\n" %predictN[0])

    fig2 = plt.figure()
    fig2.suptitle("density plots for %s using von Neumann on von Neumann"% (nameTrans))
    ax2 = fig2.add_subplot(2,1,1)
    ax2.set_title("original" )
    
    ax2.set_ylabel("y")
    originalDensityPlot = np.sum(originalMatrixN, axis=0)/(int(rebuildIter))
    ax2.pcolor(originalDensityPlot,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    ax22 = fig2.add_subplot(2,1,2,sharex=ax2)
    ax22.set_title("recreation" )
    ax22.set_xlabel("x")
    ax22.set_ylabel("y")
    rebuildDensityPlot = np.sum(rebuildMatrixN, axis=0)/(int(rebuildIter))
    im1 = ax22.pcolor(rebuildDensityPlot,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax1 = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(im1, cax=cax1)
    
    newProbNeumann = expectArrayN[:,1]
    newProbNeumann[0] = 0
    newMatrixN = np.zeros(np.shape(originalMatrixN),dtype=np.bool_)
    newMatrixN[0,:,:] = np.copy(originalMatrixN[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = newMatrixN[t,row,col]*16 + newMatrixN[t,(row+1)%rebuildLength,col]*8 + newMatrixN[t,row,(col+1)%rebuildWidth]*4  + newMatrixN[t,(row-1)%rebuildLength,col]*2 + newMatrixN[t,row,(col-1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < newProbNeumann[index]):
                    newMatrixN[t+1,row,col] = True
                else:
                    newMatrixN[t+1,row,col] = False 
    newFeaturesN = extractFeatures(newMatrixN,4)

    extraFile.write("----Neumann on Neumann---\n")
    compareProperties(extraFile,originalFeaturesN,newFeaturesN,rebuildIter)
    
    figA = plt.figure()
    figA.suptitle("stochastic rule using von Neumann on von Neumann")
    axA1 = figA.add_subplot(2,1,1)
    axA1.set_title("original" )

    axA1.set_ylabel("y")
    axA1.pcolor(originalDensityPlot,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    axA12 = figA.add_subplot(2,1,2,sharex=axA1)
    axA12.set_title("recreation - something must come from something" )
    axA12.set_xlabel("x")
    axA12.set_ylabel("y")
    newDensityPlot = np.sum(newMatrixN, axis=0)/(int(rebuildIter))
    im = axA12.pcolor(newDensityPlot,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(im, cax=cax)


    del rebuildMatrixN
    del newMatrixN
    del originalMatrixN
    del originalFeaturesN
    del rebuidFeaturesN
    del newFeaturesN
    del x_trainN
    del y_trainN
    del x_testN
    del y_testN
    del x_dataN
    del y_dataN
    del x_data_rawN
  
    file.write("_________________________\n------ Moore neighborhood ------\n____________________________\n")

    originalMatrixM = produceTimeSeriesMoore(rebuildLength, rebuildWidth, rebuildOnes, rebuildIter, transitionsMoore)
    x_data_rawM, y_dataM = retrieveNeighborhoodOutput(originalMatrixM,8)
    x_dataM = Taylor_Moore(x_data_rawM)
    timeIndexM = TimeSeriesSplit(4)
    for trainIndex, testIndex  in timeIndexM.split(x_dataM):
        x_trainM, x_testM = x_dataM[trainIndex], x_dataM[testIndex]
        y_trainM, y_testM = y_dataM[trainIndex], y_dataM[testIndex]
    clfMoore.fit(x_trainM,y_trainM)
    expectArrayM = clfMoore.predict_proba(Taylor_Moore(np.array(neighborConfigs(8))))
    predictM = expectArrayM[:,1]
    rebuildMatrixM = np.zeros(np.shape(originalMatrixM),dtype=np.bool_)
    rebuildMatrixM[0,:,:] = np.copy(originalMatrixM[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = rebuildMatrixM[t,row,col]*256 + rebuildMatrixM[t,(row+1)%rebuildLength,col]*128 + rebuildMatrixM[t,(row+1)%rebuildLength,(col+1)%rebuildWidth]*64+ rebuildMatrixM[t,row,(col+1)%rebuildWidth]*32 + rebuildMatrixM[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]*16  + rebuildMatrixM[t,(row-1)%rebuildLength,col]*8 + rebuildMatrixM[t,(row-1)%rebuildLength,(col-1)%rebuildWidth]*4 + rebuildMatrixM[t,row,(col-1)%rebuildWidth]*2 + rebuildMatrixM[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < predictM[index]):
                    rebuildMatrixM[t+1,row,col] = True
                else:
                    rebuildMatrixM[t+1,row,col] = False 
    originalFeaturesM = extractFeatures(originalMatrixM,8)
    rebuidFeaturesM = extractFeatures(rebuildMatrixM,8)
    

    rebuildDataFileMooreName = "rebuild_" + dataFileNameMoore
    datMN = open(dataFileNameMoore,'a')
    re_datMN = open(rebuildDataFileMooreName,'a')
    re_datMN.write("intersection: %s \n coeficients:\n" % clfMoore.intercept_ )
    for i in range(int(len(clfMoore.coef_)/16)):
        for e in range(16):
            re_datMN.write("\t%s"%clfMoore.coef_[0,16*i+e])
        re_datMN.write("\n")
    re_datMN.write("\n in array format\n%s\n"%clfMoore.coef_)
    for t in range(rebuildIter):
        datMN.write("t = %i\n" % t)
        re_datMN.write("t = %i\n" % t)
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                datMN.write("%i\t" % originalMatrixM[t,row,col])
                re_datMN.write("%i\t" % rebuildMatrixM[t,row,col])
            datMN.write("\n")
            re_datMN.write("\n")
    datMN.write("\narray:\n%s" % originalMatrixM)
    re_datMN.write("\narray:\n%s" % rebuildMatrixM)
    datMN.close()
    re_datMN.close()
    with open("MvsM.csv","w+") as my_csv:
        newarray = csv.writer(my_csv,delimiter=',')
        newarray.writerows(originalMatrixM)

    file.write("brier score: %s \n"% brier_score_loss(y_testM,clfMoore.predict_proba(x_testM)[:,1]))
    file.write("Hosmer-Lemeshow test: %s \n" % HosmerLemeshow_test(np.copy(originalMatrixM),expectArrayM,8))
    compareProperties(file,originalFeaturesM,rebuidFeaturesM,rebuildIter)

    fileInfo(file,predictM,transitionsMoore,8,accuracyLevel=2)
    file.write("exact value for transition from empty config: %s\n" %predictM[0])

    fig_2 = plt.figure()
    fig_2.suptitle("density plots for %s using Moore on Moore"% (nameTrans))
    ax_2 = fig_2.add_subplot(2,1,1)
    ax_2.set_title("original" )
    ax_2.set_ylabel("y")
    originalDensityPlotM = np.sum(originalMatrixM, axis=0)/(int(rebuildIter))
    ax_2.pcolor(originalDensityPlotM,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    ax_22 = fig_2.add_subplot(2,1,2,sharex=ax_2)
    ax_22.set_title("recreation" )
    ax_22.set_xlabel("x")
    ax_22.set_ylabel("y")
    rebuildDensityPlotM = np.sum(rebuildMatrixM, axis=0)/(int(rebuildIter))
    im2 = ax_22.pcolor(rebuildDensityPlotM,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax2 = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(im2, cax=cax2)

    newProbMoore = expectArrayM[:,1]
    newProbMoore[0] = 0
    newMatrixM = np.zeros(np.shape(originalMatrixM),dtype=np.bool_)
    newMatrixM[0,:,:] = np.copy(originalMatrixM[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = newMatrixM[t,row,col]*256 + newMatrixM[t,(row+1)%rebuildLength,col]*128 + newMatrixM[t,(row+1)%rebuildLength,(col+1)%rebuildWidth]*64+ newMatrixM[t,row,(col+1)%rebuildWidth]*32 + newMatrixM[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]*16  + newMatrixM[t,(row-1)%rebuildLength,col]*8 + newMatrixM[t,(row-1)%rebuildLength,(col-1)%rebuildWidth]*4 + newMatrixM[t,row,(col-1)%rebuildWidth]*2 + newMatrixM[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < newProbMoore[index]):
                    newMatrixM[t+1,row,col] = True
                else:
                    newMatrixM[t+1,row,col] = False 
    newFeaturesM = extractFeatures(newMatrixM,8)

    extraFile.write("----Moore on Moore---\n")
    compareProperties(extraFile,originalFeaturesM,newFeaturesM,rebuldIter)

    figB = plt.figure()
    figB.suptitle("stochastic rule using Moore on Moore")
    axB1 = figB.add_subplot(2,1,1)
    axB1.set_title("original" )
    axB1.set_ylabel("y")
    axB1.pcolor(originalDensityPlotM,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    axB12 = figB.add_subplot(2,1,2,sharex=axB1)
    axB12.set_title("recreation - something must come from something" )
    axB12.set_xlabel("x")
    axB12.set_ylabel("y")
    newDensityPlotB = np.sum(newMatrixM, axis=0)/(int(rebuildIter))
    imB = axB12.pcolor(newDensityPlotB,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxB = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imB, cax=caxB)
    
    
    del rebuildMatrixM
    del newMatrixM
    del originalMatrixM
    del originalFeaturesM
    del rebuidFeaturesM
    del newFeaturesM
    del x_trainM
    del y_trainM
    del x_testM
    del y_testM
    del x_dataM
    del y_dataM
    del x_data_rawM

    file.close()
    extraFile.close()
    plt.show()


def Thesis_real_wrongNeighborhood():

    plt.rcParams.update({'font.size': 23})

    fileName = "2Dca.dat"

    refParam = {'penalty':'none','solver':'sag','tol':0.001,'max_iter':1000}
    clfNeumann = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])
    clfMoore = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])

    file = open(fileName,'a')
    
    extraFile = open("ammendedRebuild.dat",'a')

    file.write("________________________________________________________\nReal: unknown neighborhood\n*********************************\n\n")
    rebuildLength = 40
    rebuildWidth = 150
    rebuildIter = 199
    rebuildOnes = 1#1000

    transitionNeumann = BuildBirthDeath(4,0.47,0.25)
    transitionMoore = BuildBirthDeath(8,0.315,0.285)

    dataFileTooSmallName = "tooSmall_b&d_alpha=0.315_beta=0.285.dat"
    dataFileTooLargeName = "tooLarge_b&d_alpha=0.47_beta=0.25.dat"

    configsNeumann = np.array(neighborConfigs(4))
    
    mooreMatrix = produceTimeSeriesMoore(rebuildLength, rebuildWidth, rebuildOnes, rebuildIter, transitionMoore)
    
    x_data_rawN, y_dataN = retrieveNeighborhoodOutput(mooreMatrix,4)
    x_dataN = Taylor_Neumann(x_data_rawN)
    timeIndexN = TimeSeriesSplit(4)
    for trainIndex, testIndex  in timeIndexN.split(x_dataN):
        x_trainN, x_testN = x_dataN[trainIndex], x_dataN[testIndex]
        y_trainN, y_testN = y_dataN[trainIndex], y_dataN[testIndex]
    
    clfNeumann.fit(x_trainN,y_trainN)
    
    expectArrayTooSmall = clfNeumann.predict_proba(Taylor_Neumann(np.array(neighborConfigs(4))))
    predictTooSmall = expectArrayTooSmall[:,1]
    rebuildMatrixTooSmall = np.zeros(np.shape(mooreMatrix),dtype=np.bool_)
    rebuildMatrixTooSmall[0,:,:] = np.copy(mooreMatrix[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = rebuildMatrixTooSmall[t,row,col]*16 + rebuildMatrixTooSmall[t,(row+1)%rebuildLength,col]*8 + rebuildMatrixTooSmall[t,row,(col+1)%rebuildWidth]*4  + rebuildMatrixTooSmall[t,(row-1)%rebuildLength,col]*2 + rebuildMatrixTooSmall[t,row,(col-1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < predictTooSmall[index]):
                    rebuildMatrixTooSmall[t+1,row,col] = True
                else:
                    rebuildMatrixTooSmall[t+1,row,col] = False 
    originalFeaturesN = extractFeatures(mooreMatrix,4)
    rebuidFeaturesN = extractFeatures(rebuildMatrixTooSmall,4)

    rebuildDataFileTooSmallName = "rebuild_" + dataFileTooSmallName
    datTS = open(dataFileTooSmallName,'a')
    re_datTS = open(rebuildDataFileTooSmallName,'a')
    re_datTS.write("intersection: %s \n coeficients:\n" % clfNeumann.intercept_ )
    for i in range(int(len(clfNeumann.coef_)/16)):
        for e in range(16):
            re_datTS.write("\t%s"%clfNeumann.coef_[0,16*i+e])
        re_datTS.write("\n")
    re_datTS.write("\n in array format \n %s \n " %clfNeumann.coef_)
    for t in range(rebuildIter):
        datTS.write("t = %i\n" % t)
        re_datTS.write("t = %i\n" % t)
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                datTS.write("%i\t" % mooreMatrix[t,row,col])
                re_datTS.write("%i\t" % rebuildMatrixTooSmall[t,row,col])
            datTS.write("\n")
            re_datTS.write("\n")
    datTS.write("\narray:\n%s" % mooreMatrix)
    re_datTS.write("\narray:\n%s" % rebuildMatrixTooSmall)
    datTS.close()
    re_datTS.close()
    with open("NvsM.csv","w+") as my_csv:
        newarray = csv.writer(my_csv,delimiter=',')
        newarray.writerows(mooreMatrix)
    
    file.write("_________________________\n------ von Neumann neighborhood instead of Moore ------\n____________________________\n")

    file.write("brier score: %s \n"% brier_score_loss(y_testN,clfNeumann.predict_proba(x_testN)[:,1]))
    convertExpect = arrayNeumannToMoore(predictTooSmall)
    compareProperties(file,originalFeaturesN,rebuidFeaturesN,rebuildIter)
    file.write("Transitions in Moore format\n************************\n")
    fileInfo(file,convertExpect,transitionMoore,8,accracyLevel=0)
    file.write("Transitions in Neumann format\n************************\n")
    shrinkTrue = arrayMooreToNeumann(transitionMoore)
    for i in range(0,4):
        if(i):
            file.write("\hline\hline\n")
        file.write(f"&{configsNeumann[8*i,0]}{configsNeumann[8*i,1]}{configsNeumann[8*i,2]}{configsNeumann[8*i,3]}{configsNeumann[8*i,4]}&{configsNeumann[8*i+1,0]}{configsNeumann[8*i+1,1]}{configsNeumann[8*i+1,2]}{configsNeumann[8*i+1,3]}{configsNeumann[8*i+1,4]}&{configsNeumann[8*i+2,0]}{configsNeumann[8*i+2,1]}{configsNeumann[8*i+2,2]}{configsNeumann[8*i+2,3]}{configsNeumann[8*i+2,4]}&{configsNeumann[8*i+3,0]}{configsNeumann[8*i+3,1]}{configsNeumann[8*i+3,2]}{configsNeumann[8*i+3,3]}{configsNeumann[8*i+3,4]}&{configsNeumann[8*i+4,0]}{configsNeumann[8*i+4,1]}{configsNeumann[8*i+4,2]}{configsNeumann[8*i+4,3]}{configsNeumann[8*i+4,4]}&{configsNeumann[8*i+5,0]}{configsNeumann[8*i+5,1]}{configsNeumann[8*i+5,2]}{configsNeumann[8*i+5,3]}{configsNeumann[8*i+5,4]}&{configsNeumann[8*i+6,0]}{configsNeumann[8*i+6,1]}{configsNeumann[8*i+6,2]}{configsNeumann[8*i+6,3]}{configsNeumann[8*i+6,4]}&{configsNeumann[8*i+7,0]}{configsNeumann[8*i+7,1]}{configsNeumann[8*i+7,2]}{configsNeumann[8*i+7,3]}{configsNeumann[8*i+7,4]}\\\ \n")
        file.write("\hline\n")
        file.write("extr.&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(predictTooSmall[8*i],predictTooSmall[8*i+1],predictTooSmall[8*i+2],predictTooSmall[8*i+3],predictTooSmall[8*i+4],predictTooSmall[8*i+5],predictTooSmall[8*i+6],predictTooSmall[8*i+7]))
        file.write("true&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(shrinkTrue[8*i,0],shrinkTrue[8*i+1,0],shrinkTrue[8*i+2,0],shrinkTrue[8*i+3,0],shrinkTrue[8*i+4,0],shrinkTrue[8*i+5,0],shrinkTrue[8*i+6,0],shrinkTrue[8*i+7,0]))
        file.write("error&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(shrinkTrue[8*i,1],shrinkTrue[8*i+1,1],shrinkTrue[8*i+2,1],shrinkTrue[8*i+3,1],shrinkTrue[8*i+4,1],shrinkTrue[8*i+5,1],shrinkTrue[8*i+6,1],shrinkTrue[8*i+7,1]))

    file.write("exact value for transition from empty config: %s\n" %predictTooSmall[0])

    figA = plt.figure()
    figA.suptitle("density plots for larger neighborhood than expected")
    ax2 = figA.add_subplot(2,1,1)
    ax2.set_title("original" )
    #ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    originalDensityPlot_1 = np.sum(mooreMatrix, axis=0)/(int(rebuildIter))
    ax2.pcolor(originalDensityPlot_1,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    ax22 = figA.add_subplot(2,1,2,sharex=ax2)
    ax22.set_title("recreation" )
    ax22.set_xlabel("x")
    ax22.set_ylabel("y")
    rebuildDensityPlot_1 = np.sum(rebuildMatrixTooSmall, axis=0)/(int(rebuildIter))
    imA = ax22.pcolor(rebuildDensityPlot_1,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxA = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imA, cax=caxA)

    newProbTS = expectArrayTooSmall[:,1]
    newProbTS[0] = 0
    newMatrixTs = np.zeros(np.shape(mooreMatrix),dtype=np.bool_)
    newMatrixTs[0,:,:] = np.copy(mooreMatrix[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = newMatrixTs[t,row,col]*16 + newMatrixTs[t,(row+1)%rebuildLength,col]*8 + newMatrixTs[t,row,(col+1)%rebuildWidth]*4  + newMatrixTs[t,(row-1)%rebuildLength,col]*2 + newMatrixTs[t,row,(col-1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < newProbTS[index]):
                    newMatrixTs[t+1,row,col] = True
                else:
                    newMatrixTs[t+1,row,col] = False 
    newFTS = extractFeatures(newMatrixTs,4)

    extraFile.write("----von Neumann on Moore---\n")
    compareProperties(extraFile,originalFeaturesN,newFTS,rebuildIter)
    
    figC = plt.figure()
    figC.suptitle("density plots for larger neighborhood than expected")
    axC1 = figC.add_subplot(2,1,1)
    axC1.set_title("original" )
    #axC1.set_xlabel("x")
    axC1.set_ylabel("y")
    axC1.pcolor(originalDensityPlot_1,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    axC12 = figC.add_subplot(2,1,2,sharex=axC1)
    axC12.set_title("recreation - something must come from something" )
    axC12.set_xlabel("x")
    axC12.set_ylabel("y")
    newDensityPlotC = np.sum(newMatrixTs, axis=0)/(int(rebuildIter))
    imC = axC12.pcolor(newDensityPlotC,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxC = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imC, cax=caxC)
    
    del rebuildMatrixTooSmall
    del newMatrixTs
    del mooreMatrix
    del originalFeaturesN
    del rebuidFeaturesN
    del newFTS
    del x_trainN
    del y_trainN
    del x_testN
    del y_testN
    del x_dataN
    del y_dataN
    del x_data_rawN
    
    file.write("_________________________\n------ Moore neighborhood instead of von Neumann ------\n____________________________\n")
  
    neumannMatrix = produceTimeSeriesNeumann(rebuildLength, rebuildWidth, rebuildOnes, rebuildIter, transitionNeumann)
    x_data_rawM, y_dataM = retrieveNeighborhoodOutput(neumannMatrix,8)
    x_dataM = Taylor_Moore(x_data_rawM)
    timeIndexM = TimeSeriesSplit(4)
    for trainIndex, testIndex  in timeIndexM.split(x_dataM):
        x_trainM, x_testM = x_dataM[trainIndex], x_dataM[testIndex]
        y_trainM, y_testM = y_dataM[trainIndex], y_dataM[testIndex]
    clfMoore.fit(x_trainM,y_trainM)

    expectArrayTooLarge = clfMoore.predict_proba(Taylor_Moore(np.array(neighborConfigs(8))))
    predictTooLarge = expectArrayTooLarge[:,1]
    rebuildMatrixTooLarge = np.zeros(np.shape(neumannMatrix),dtype=np.bool_)
    rebuildMatrixTooLarge[0,:,:] = np.copy(neumannMatrix[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = rebuildMatrixTooLarge[t,row,col]*256 + rebuildMatrixTooLarge[t,(row+1)%rebuildLength,col]*128 + rebuildMatrixTooLarge[t,(row+1)%rebuildLength,(col+1)%rebuildWidth]*64+ rebuildMatrixTooLarge[t,row,(col+1)%rebuildWidth]*32 + rebuildMatrixTooLarge[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]*16  + rebuildMatrixTooLarge[t,(row-1)%rebuildLength,col]*8 + rebuildMatrixTooLarge[t,(row-1)%rebuildLength,(col-1)%rebuildWidth]*4 + rebuildMatrixTooLarge[t,row,(col-1)%rebuildWidth]*2 + rebuildMatrixTooLarge[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < predictTooLarge[index]):
                    rebuildMatrixTooLarge[t+1,row,col] = True
                else:
                    rebuildMatrixTooLarge[t+1,row,col] = False 
    originalFeaturesM = extractFeatures(neumannMatrix,8)
    rebuidFeaturesM = extractFeatures(rebuildMatrixTooLarge,8)

    rebuildDataFileTooLargeName = "rebuild_" + dataFileTooLargeName
    datTL = open(dataFileTooLargeName,'a')
    re_datTL = open(rebuildDataFileTooLargeName,'a')
    re_datTL.write("intersection: %s \n coeficients:\n" % clfMoore.intercept_ )
    for i in range(int(len(clfMoore.coef_)/16)):
        for e in range(16):
            re_datTL.write("\t%s"%clfMoore.coef_[0,16*i+e])
        re_datTL.write("\n")
    re_datTL.write("\n in array format\n%s\n"%clfMoore.coef_)
    for t in range(rebuildIter):
        datTL.write("t = %i\n" % t)
        re_datTL.write("t = %i\n" % t)
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                datTL.write("%i\t" % neumannMatrix[t,row,col])
                re_datTL.write("%i\t" % rebuildMatrixTooLarge[t,row,col])
            datTL.write("\n")
            re_datTL.write("\n")
    datTL.write("\narray:\n%s" % neumannMatrix)
    re_datTL.write("\narray:\n%s" % rebuildMatrixTooLarge)
    datTL.close()
    re_datTL.close()
    with open("MvsN.csv","w+") as my_csv:
        newarray = csv.writer(my_csv,delimiter=',')
        newarray.writerows(neumannMatrix)

    file.write("brier score: %s \n"% brier_score_loss(y_testM,clfMoore.predict_proba(x_testM)[:,1]))
    compareProperties(file,originalFeaturesM,rebuidFeaturesM,rebuildIter)

    convertOriginal = arrayNeumannToMoore(transitionNeumann)
    file.write("Transitions in Moore format\n************************\n")
    fileInfo(file,predictTooLarge,convertOriginal,8,accuracyLevel=0)
    file.write("Transitions in Neumann format\n************************\n")
    shrinkExpect = arrayMooreToNeumann(predictTooLarge)
    for i in range(0,4):
        if(i):
            file.write("\hline\hline\n")
        file.write(f"&{configsNeumann[8*i,0]}{configsNeumann[8*i,1]}{configsNeumann[8*i,2]}{configsNeumann[8*i,3]}{configsNeumann[8*i,4]}&{configsNeumann[8*i+1,0]}{configsNeumann[8*i+1,1]}{configsNeumann[8*i+1,2]}{configsNeumann[8*i+1,3]}{configsNeumann[8*i+1,4]}&{configsNeumann[8*i+2,0]}{configsNeumann[8*i+2,1]}{configsNeumann[8*i+2,2]}{configsNeumann[8*i+2,3]}{configsNeumann[8*i+2,4]}&{configsNeumann[8*i+3,0]}{configsNeumann[8*i+3,1]}{configsNeumann[8*i+3,2]}{configsNeumann[8*i+3,3]}{configsNeumann[8*i+3,4]}&{configsNeumann[8*i+4,0]}{configsNeumann[8*i+4,1]}{configsNeumann[8*i+4,2]}{configsNeumann[8*i+4,3]}{configsNeumann[8*i+4,4]}&{configsNeumann[8*i+5,0]}{configsNeumann[8*i+5,1]}{configsNeumann[8*i+5,2]}{configsNeumann[8*i+5,3]}{configsNeumann[8*i+5,4]}&{configsNeumann[8*i+6,0]}{configsNeumann[8*i+6,1]}{configsNeumann[8*i+6,2]}{configsNeumann[8*i+6,3]}{configsNeumann[8*i+6,4]}&{configsNeumann[8*i+7,0]}{configsNeumann[8*i+7,1]}{configsNeumann[8*i+7,2]}{configsNeumann[8*i+7,3]}{configsNeumann[8*i+7,4]}\\\ \n")
        file.write("\hline\n")
        file.write("extr.&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(shrinkExpect[8*i,0],shrinkExpect[8*i+1,0],shrinkExpect[8*i+2,0],shrinkExpect[8*i+3,0],shrinkExpect[8*i+4,0],shrinkExpect[8*i+5,0],shrinkExpect[8*i+6,0],shrinkExpect[8*i+7,0]))
        file.write("error&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(shrinkExpect[8*i,1],shrinkExpect[8*i+1,1],shrinkExpect[8*i+2,1],shrinkExpect[8*i+3,1],shrinkExpect[8*i+4,1],shrinkExpect[8*i+5,1],shrinkExpect[8*i+6,1],shrinkExpect[8*i+7,1]))
        file.write("true&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(transitionNeumann[8*i],transitionNeumann[8*i+1],transitionNeumann[8*i+2],transitionNeumann[8*i+3],transitionNeumann[8*i+4],transitionNeumann[8*i+5],transitionNeumann[8*i+6],transitionNeumann[8*i+7]))

    file.write("exact value for transition from empty config: %s\n" %predictTooLarge[0])

    fugH = plt.figure()
    fugH.suptitle("density plots for smaller neighborhood than expected")
    ax_2 = fugH.add_subplot(2,1,1)
    ax_2.set_title("original" )
    #ax_2.set_xlabel("x")
    ax_2.set_ylabel("y")
    originalDensityPlot_2 = np.sum(neumannMatrix, axis=0)/(int(rebuildIter))
    ax_2.pcolor(originalDensityPlot_2,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    ax_22 = fugH.add_subplot(2,1,2,sharex=ax_2)
    ax_22.set_title("recreation" )
    ax_22.set_xlabel("x")
    ax_22.set_ylabel("y")
    rebuildDensityPlot_2 = np.sum(rebuildMatrixTooLarge, axis=0)/(int(rebuildIter))
    imH = ax_22.pcolor(rebuildDensityPlot_2,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxH = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imH, cax=caxH)
    
    newProbTL= expectArrayTooLarge[:,1]
    newProbTL[0] = 0
    newMatrixTL = np.zeros(np.shape(neumannMatrix),dtype=np.bool_)
    newMatrixTL[0,:,:] = np.copy(neumannMatrix[0,:,:])
    for t in range(rebuildIter):
        for row in range(rebuildLength):
            for col in range(rebuildWidth):
                index = newMatrixTL[t,row,col]*256 + newMatrixTL[t,(row+1)%rebuildLength,col]*128 + newMatrixTL[t,(row+1)%rebuildLength,(col+1)%rebuildWidth]*64+ newMatrixTL[t,row,(col+1)%rebuildWidth]*32 + newMatrixTL[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]*16  + newMatrixTL[t,(row-1)%rebuildLength,col]*8 + newMatrixTL[t,(row-1)%rebuildLength,(col-1)%rebuildWidth]*4 + newMatrixTL[t,row,(col-1)%rebuildWidth]*2 + newMatrixTL[t,(row-1)%rebuildLength,(col+1)%rebuildWidth]
                rand = np.random.rand()
                if(rand < newProbTL[index]):
                    newMatrixTL[t+1,row,col] = True
                else:
                    newMatrixTL[t+1,row,col] = False 
    newFTL = extractFeatures(newMatrixTL,8)
    extraFile.write("----Moore on von Neumann---\n")
    compareProperties(extraFile,originalFeaturesM,newFTL,rebuildIter)
    
    figD = plt.figure()
    figD.suptitle("density plots for smaller neighborhood than expected")
    axD1 = figD.add_subplot(2,1,1)
    axD1.set_title("original" )
    #axD1.set_xlabel("x")
    axD1.set_ylabel("y")
    axD1.pcolor(originalDensityPlot_2,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    axD12 = figD.add_subplot(2,1,2,sharex=axD1)
    axD12.set_title("recreation - something must come from something" )
    axD12.set_xlabel("x")
    axD12.set_ylabel("y")
    newDensityPlotD = np.sum(newMatrixTL, axis=0)/(int(rebuildIter))
    imD = axD12.pcolor(newDensityPlotD,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxD = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imD, cax=caxD)
    
    del rebuildMatrixTooLarge
    del newMatrixTL
    del neumannMatrix
    del originalFeaturesM
    del rebuidFeaturesM
    del newFTL
    del x_trainM
    del y_trainM
    del x_testM
    del y_testM
    del x_dataM
    del y_dataM
    del x_data_rawM
    print("end part b")
    file.close()
    extraFile.close()
    plt.show()

def Thesis_visual():

    plt.rcParams.update({'font.size': 22})
    iter = 500
    nRows = 60
    nCols = 100
    nOnes = 1000
    cutOff = 100
    steps = [100,300]
    update = buildGoLArray()
    updateName = "Game of Life"
    matrix = produceTimeSeriesMoore(nRows, nCols, nOnes, iter, update)
    fig1 = plt.figure()
    fig1.suptitle(" %s"% updateName)
    ax11 = fig1.add_subplot(2,2,1)
    ax11.set_title("initial configuration")
    ax11.set_xlabel("x")
    ax11.set_ylabel("y")
    ax11.pcolor(matrix[0,:,:],cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
    ax12 = fig1.add_subplot(2,2,2)
    ax12.set_title("after %i time steps" %steps[0] )
    ax12.set_xlabel("x")
    ax12.set_ylabel("y")
    ax12.pcolor(matrix[steps[0],:,:],cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
    ax13 = fig1.add_subplot(2,2,3)
    ax13.set_title("after %i time steps" % steps[1] )
    ax13.set_xlabel("x")
    ax13.set_ylabel("y")
    ax13.pcolor(matrix[steps[1],:,:],cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
    ax14 = fig1.add_subplot(2,2,4)
    ax14.set_title("original after %i iterations" % iter )
    ax14.set_xlabel("x")
    ax14.set_ylabel("y")
    ax14.pcolor(matrix[iter,:,:],cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title("density plots for %s of %i steps skipping first %i steps"% (updateName,iter,cutOff))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    originalDensityPlot = np.sum(matrix[cutOff:,:,:], axis=0)/(int(iter-cutOff))
    imD = ax2.pcolor(originalDensityPlot,cmap = 'binary',edgecolors='gray',vmin=0.0,vmax=1.0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    caxD = plt.axes([0.85, 0.1, 0.075, 0.8])

    plt.colorbar(imD, cax=caxD)
    del matrix
    del originalDensityPlot
    plt.show()

