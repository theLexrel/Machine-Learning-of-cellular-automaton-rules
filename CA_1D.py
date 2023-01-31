'''
Many of the differences between this code and the others are due to this one being the first iteration
which was built on top off in the subsequent steps. However, it isn't the first code I wrote for the thesis
and it therefore borrows a lot from even earlier iterations. 
Some of the choices are only feasible in 1D and had to be changed for higher order systems.
Hence, the length of the code could conceivably be reduced drastically, but I saw no need for that 
while writting the thesis.
'''
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve, cross_val_score

class CA():

    def __init__(self,Length, Iter, Ones):
        self.length = Length
        self.iter = Iter
        self.nOnes = Ones
        self.timeSeries = np.zeros((Iter+1,Length), np.bool_)
        self.nRows = Iter + 1

        self.transProb = np.random.rand(8)
        self.figurecount = 0

        self.inputConfig = [[True,True,True],[True,True,False],[True,False,True],[True,False,False], 
                            [False,True,True], [False,True,False], [False,False,True], [False,False,False]]


    def updateStep(self,i,k):
        lastLeft = self.timeSeries[i,(k-1)%self.length]
        lastMiddle = self.timeSeries[i,k]
        lastRight = self.timeSeries[i,(k+1)%self.length]
        rand = np.random.random()
        if( lastLeft and  lastMiddle and  lastRight):
            if(rand < self.transProb[0]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if( lastLeft and  lastMiddle and not lastRight):
            if(rand < self.transProb[1]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if( lastLeft and not lastMiddle and  lastRight):
            if(rand < self.transProb[2]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if( lastLeft and not lastMiddle and not lastRight):
            if(rand < self.transProb[3]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if(not lastLeft and  lastMiddle and  lastRight):
            if(rand < self.transProb[4]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if(not lastLeft and  lastMiddle and not lastRight):
            if(rand < self.transProb[5]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if(not lastLeft and not lastMiddle and  lastRight):
            if(rand < self.transProb[6]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False
        if(not lastLeft and not lastMiddle and not lastRight):
            if(rand < self.transProb[7]):
                self.timeSeries[i+1,k] = True
            else:
                self.timeSeries[i+1,k] = False

    def neighbors(self,k,l):
        # gives out the neighbors of iteratin k cell l as array [left,right]
        return np.array([self.timeSeries[k,(l-1)%self.length],self.timeSeries[k,(l+1)%self.length]])

    def setOnes(self,number):
        self.nOnes = number

    def setTransition(self,array):
        if(not len(array)==8):
            print("the threshold array needs to be length 8 and of form: [000,001,010,011,100,101,110,111]")
            return 0
        else:
            self.transProb = np.copy(array)

    def MakeTimeSeries(self):   
        initSetting = np.zeros(self.length,dtype=np.bool_)
        for i in range(self.nOnes):     
            initSetting[i]=True
        np.random.shuffle(initSetting)
        self.timeSeries[0,:] = initSetting
        for i in range(0,self.iter):
            for k in range(0,self.length):
                self.updateStep(i,k)


def globalFeatures(matrix):
    rows, length = np.shape(matrix)
    iter = rows -1
    occ = np.sum(matrix,axis=1)/length      
    lamb = np.sum(occ)/rows
    change = np.zeros(iter)
    totalStats = np.zeros(8) #000,001,010,011,100,101,110,111
    iterStats = np.zeros((rows,8))
    drift = np.zeros(iter)
    prob = np.zeros((rows,length)) #prob at time t for size n: prob[t,n]
    entropy = np.zeros(rows)
    for i in range(rows):
        if(not(i == iter)):
            if(np.sum(matrix[i,:])==0 or np.sum(matrix[i+1,:])==0):
                drift[i] = 0
            else:
                drift[i] = int(np.sum(np.multiply(matrix[i+1,:],np.arange(length)))/np.sum(matrix[i+1,:])-np.sum(np.multiply(matrix[i,:],np.arange(length)))/np.sum(matrix[i,:]))
        wrap = False        # check if cluster wraps around due to periodic boundary condition 
        if(matrix[i,0] and matrix[i,length-1]):
            wrap = True
        tracker = 0
        cluster = 0
        max = np.count_nonzero(matrix[i,:]) # counts the number of occupied nodes 
        if(max == length):     # if all nodes occupied, no need to go through loops below
            entropy[i] = 0          # since ln(1) = 0
        else:
            while(tracker < length):
                if(wrap):  # skip the first cluster as its incomplete and will be picked up later 
                    while(matrix[i,tracker]):
                        tracker += 1
                    wrap=False # has to be set to False again because loop only for the 1st cluster
                while(matrix[i,(tracker)%length]):
                    cluster += 1
                    tracker += 1
                prob[i,cluster] += 1
                cluster = 0
                tracker += 1
            calc = prob[i,:]*np.arange(0,length)/length
            calc = calc[calc>0]     # log(0) not defined, therefor obmitted 
            entropy[i] = np.sum(-calc*np.log(calc))
        for t in range(length):
            if(not(i == iter)):
                if(not(matrix[i+1,t]==matrix[i,t])):
                    change[i] +=1
            neigh = np.array([matrix[i,(t-1)%length],matrix[i,(t+1)%length]])
            if(neigh[0] and matrix[i,t] and neigh[1]):
                totalStats[7] +=1
                iterStats[i,7] +=1
            if(neigh[0] and matrix[i,t] and not neigh[1]):
                totalStats[6] +=1
                iterStats[i,6] +=1
            if(neigh[0] and not matrix[i,t] and neigh[1]):
                totalStats[5] +=1
                iterStats[i,5] +=1
            if(neigh[0] and not matrix[i,t] and not neigh[1]):
                totalStats[4] +=1
                iterStats[i,4] +=1
            if(not neigh[0] and matrix[i,t] and neigh[1]):
                totalStats[3] +=1
                iterStats[i,3] +=1
            if(not neigh[0] and matrix[i,t] and not neigh[1]):
                totalStats[2] +=1
                iterStats[i,2] +=1
            if(not neigh[0] and not matrix[i,t] and neigh[1]):
                totalStats[1] +=1
                iterStats[i,1] +=1
            if(not neigh[0] and not matrix[i,t] and not neigh[1]):
                totalStats[0] +=1
                iterStats[i,0] +=1
    change = change/length
    tempEntropy = np.array(np.copy(iterStats),dtype=np.bool_)
    tempEntropy = np.sum(tempEntropy,axis=1)/8
    occ_in_neigh = np.array([0,1,1,1,2,2,2,3])
    av_neigh_occ = np.multiply(occ_in_neigh,iterStats)
    av_neigh_density = np.sum(av_neigh_occ,axis=1)/(3*length)
    sequenceEntropy = totalStats/(length*rows)
    features = {
        "occupation" :occ, 
        "lambda" : lamb,
        "changeRate":change, 
        "temporalEntropy":tempEntropy, 
        "sequenceEntropy":sequenceEntropy,
        "spatialEntropy" : entropy,
        "neighborhoodDensity" : av_neigh_density,
        "drift" : drift
    }
    return features

def TaylorCoef(input):
    output = []
    for i in range(len(input)):
        dummy = []
        dummy.append(int(input[i,0]))
        dummy.append(int(input[i,1]))
        dummy.append(int(input[i,2]))
        dummy.append(int(input[i,0])*int(input[i,1]))
        dummy.append(int(input[i,0])*int(input[i,2]))
        dummy.append(int(input[i,1])*int(input[i,2]))
        dummy.append(int(input[i,0])*int(input[i,1])*int(input[i,2]))
        output.append(dummy)
    output = np.array(output)
    return output

def BuildingInOutput(matrix):
    iter, length = np.shape(matrix)
    vecX = np.zeros((iter*length,3)) # all cells that were updated + neighborhood
    output = np.zeros(iter*length)
    for i in range(0,iter-1):
        for k in range(0,length):
            vecX[k + i*length,0] = matrix[i,(k-1)%length]
            vecX[k + i*length,1] = matrix[i,k]
            vecX[k + i*length,2] = matrix[i,(k+1)%length]
            output[k + i*length] = int(matrix[i+1,k])           
    input = TaylorCoef(vecX)
    return input,output

def RebuildMatrix(length,iter,initialState,Prob):
    rebuildMatrix = np.zeros((iter+1,length),dtype=np.bool_)
    rebuildMatrix[0,:] = np.copy(initialState)
    for t in range(iter):
        for l in range(length):
            lastLeft = rebuildMatrix[t,(l-1)%length]
            lastMiddle = rebuildMatrix[t,l]
            lastRight = rebuildMatrix[t,(l+1)%length]
            rand = np.random.random()
            if( lastLeft and  lastMiddle and  lastRight):
                if(rand < Prob[0]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if( lastLeft and  lastMiddle and not lastRight):
                if(rand < Prob[1]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if( lastLeft and not lastMiddle and  lastRight):
                if(rand < Prob[2]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if( lastLeft and not lastMiddle and not lastRight):
                if(rand < Prob[3]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if(not lastLeft and  lastMiddle and  lastRight):
                if(rand < Prob[4]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if(not lastLeft and  lastMiddle and not lastRight):
                if(rand < Prob[5]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if(not lastLeft and not lastMiddle and  lastRight):
                if(rand < Prob[6]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
            if(not lastLeft and not lastMiddle and not lastRight):
                if(rand < Prob[7]):
                    rebuildMatrix[t+1,l] = True
                else:
                    rebuildMatrix[t+1,l] = False
    return rebuildMatrix

def CompareGlobalFeatures(original,rebuild, file):
    origFeat = globalFeatures(original)
    rebuildFeat = globalFeatures(rebuild)
    relax = 0 #int(0.1*cell.iter)
    origLam = origFeat["lambda"]
    origChange = np.sum(origFeat["changeRate"][relax:])/len(origFeat['changeRate'][relax:])
    origOcc = np.sum(origFeat["occupation"][relax:])/len(origFeat["occupation"][relax:])
    origDens = np.sum(origFeat["neighborhoodDensity"][relax:])/len(origFeat["neighborhoodDensity"][relax:])
    origTemp = np.sum(origFeat["temporalEntropy"][relax:])/len(origFeat["temporalEntropy"][relax:])
    origEnt = np.sum(origFeat["spatialEntropy"][relax:])/len(origFeat["spatialEntropy"][relax:])
    reLam = rebuildFeat["lambda"]
    reChange = np.sum(rebuildFeat["changeRate"][relax:])/len(rebuildFeat['changeRate'][relax:])
    reOcc = np.sum(rebuildFeat["occupation"][relax:])/len(rebuildFeat["occupation"][relax:])
    reDens = np.sum(rebuildFeat["neighborhoodDensity"][relax:])/len(rebuildFeat["neighborhoodDensity"][relax:])
    reTemp = np.sum(rebuildFeat["temporalEntropy"][relax:])/len(rebuildFeat["temporalEntropy"][relax:])
    reEnt = np.sum(rebuildFeat["spatialEntropy"][relax:])/len(rebuildFeat["spatialEntropy"][relax:])
    print("\t\t & original & rebuild\\\ ")
    print("\hline")
    print(f"lambda & {origLam} & {reLam}\\\ ")
    print(f"mean Change rate & {origChange} & {reChange}\\\ ")
    print(f"mean occupancy rate & {origOcc} & {reOcc}\\\ ")
    print(f"mean neighborhood density rate & {origDens} & {reDens}\\\ ")
    print(f"mean temporal entropy & {origTemp} & {reTemp}\\\ ")
    print(f"mean spatial entropy &{origEnt} & {reEnt}\\\ ")
    file.write("\t\t & original & rebuild\\\ \n")
    file.write("\hline\n")
    file.write(f"lambda & {origLam} & {reLam}\\\ \n")
    file.write(f"mean Change rate & {origChange} & {reChange}\\\ \n")
    file.write(f"mean occupancy rate & {origOcc} & {reOcc}\\\ \n")
    file.write(f"mean neighborhood density rate & {origDens} & {reDens}\\\ \n")
    file.write(f"mean temporal entropy & {origTemp} & {reTemp}\\\ \n")
    file.write(f"mean spatial entropy &{origEnt} & {reEnt}\\\ \n")

def BuildHistogram(matrix, splits = 0):
    counter = np.zeros((2**3,2))
    nRows, nCols = np.shape(matrix)
    for series in range(splits+1):
        start = int(nRows/(splits+1))*series
        end = int(nRows/(splits+1))*(series+1)-1
        for i in range(start,end):
            for k in range(nCols):
                index = (not matrix[i,(k-1)%nCols])*4 + (not matrix[i,k])*2 + (not matrix[i,(k+1)%nCols])*1 # inversely written
                row = int(matrix[i+1,k])
                counter[index,row] += 1
    for i in range(len(counter)):
        sum = np.copy(counter[i,0])+np.copy(counter[i,1])
        counter[i,0] = counter[i,0]/sum
        counter[i,1] = counter[i,1]/sum
    return counter

def HosmerLemeshow(matrix,y_expect, splits = 0):
    observ = BuildHistogram(matrix, splits = splits)    
    #hl = np.sum((observ[0]-y_expect[0])**2/y_expect[0]+(observ[1]-y_expect[1])**2/y_expect[1])
    hl = 0
    for i in range(len(observ)):
        if(y_expect[i,0] and y_expect[i,1]):
            hl += (observ[i,0]-y_expect[i,0])**2/y_expect[i,0]+(observ[i,1]-y_expect[i,1])**2/y_expect[i,1]
        elif(not y_expect[i,0] and y_expect[i,1]):
            hl += observ[i,0] + (observ[i,1]-y_expect[i,1])**2/y_expect[i,1]
        else:
            hl += (observ[i,0]-y_expect[i,0])**2/y_expect[i,0] + observ[i,1]
    hl = hl/len(observ)
    return hl

def plot_learning_curve(estimator, title, X, y, scoring = None, train_sizes=np.linspace(0.1, 1.0, 10)):
    colours = [#"#882b46",
    "#298A08",'#ae1212',"#FAAC58","#008bd3"]
    _, axes = plt.subplots(1, 3)#, figsize=(20, 5))

    _.suptitle("performance analysis for "+title)

    axes[0].set_title("error evolution")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("error")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y,scoring=scoring,train_sizes=train_sizes, return_times=True )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    #axes[0].set_yscale('log')
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=colours[1],
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=colours[3],
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color=colours[1], label="Training error"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color=colours[3], label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-",color=colours[3])
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
        color=colours[3]
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    #axes[2].set_yscale('log')
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-",color=colours[3])
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
        color=colours[3]
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("error")
    axes[2].set_title("Performance of the model")

    return plt

def brier_error(y_true,y_pred):
    # implementation of brier error that doen't return a negative value when used in GridSearch of cross validation
    return -np.abs(np.sum(np.square(y_true-y_pred))/len(y_true))

def fileInfo(file,brier, HL, predictProb,transitions,accuracyLevel=0):
    file.write("brier score: %s \n"% brier)
    file.write("Hosmer-Lemeshow test: %s\n" % HL)
    file.write("&111&110&101&110&011&010&001&000\\\ \n")
    file.write("\hline")
    if(accuracyLevel==0):
        file.write("predict&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(predictProb[0],predictProb[1],predictProb[2],predictProb[3],predictProb[4],predictProb[5],predictProb[6],predictProb[7]))
        file.write("true&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f&%1.3f\\\ \n"%(transitions[0],transitions[1],transitions[2],transitions[3],transitions[4],transitions[5],transitions[6],transitions[7]))
    elif(accuracyLevel==1):
        file.write("predict&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f\\\ \n"%(predictProb[0],predictProb[1],predictProb[2],predictProb[3],predictProb[4],predictProb[5],predictProb[6],predictProb[7]))
        file.write("true&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f&%1.6f\\\ \n"%(transitions[0],transitions[1],transitions[2],transitions[3],transitions[4],transitions[5],transitions[6],transitions[7]))
    else:
        file.write("predict&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f\\\ \n"%(predictProb[0],predictProb[1],predictProb[2],predictProb[3],predictProb[4],predictProb[5],predictProb[6],predictProb[7]))
        file.write("true&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f&%1.8f\\\ \n"%(transitions[0],transitions[1],transitions[2],transitions[3],transitions[4],transitions[5],transitions[6],transitions[7]))

def Thesis_proveOfConcept(show=True):

    param_grid = [  
        {'solver' : ['newton-cg'],  'penalty' : ['none'],  'tol' : [1e-3,1e-4], 'max_iter' : [100,1000]},
        {'solver' : ['lbfgs'],      'penalty' : ['none'],  'tol' : [1e-3,1e-4], 'max_iter' : [100,1000]},
        {'solver' : ['sag'],        'penalty' : ['none'], 'tol' : [1e-3,1e-4], 'max_iter' : [100,1000]},
        {'solver' : ['saga'],       'penalty' : ['none'],  'tol' : [1e-3,1e-4], 'max_iter' : [100,1000]} 
        ]
    brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)

    grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, scoring=brier_score)
    
    refParam = {'penalty':'none','solver':'sag','tol':0.001, 'max_iter':1000}
    refClf = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])
    virginClf = LogisticRegression()

    plt.rcParams.update({'font.size': 23})
    fileName = "ca1D_real.dat"

    file = open(fileName,'w')

    file.write("prove of concept:\n_______________\n\n")
    length = 40
    iter = 49
    file.write("length: %i\n iterations: %i\n"%(length,iter))
    cell = CA(length,iter,1)
    oneArray=[1,int(0.25*length),int(0.5*length),int(0.75*length),length-1]
    file.write("initial configs: %s\n"%oneArray)
    test = CA(int(length/2),int(iter/2),1)
    rule110= [0, 1, 1, 0, 1, 1, 1, 0]
    spreadSeed = [1, 1, 1, 0.5, 1, 1, 0.5, 0] 
    #randomArray = np.random.rand(len(rule110))
    randomArray = [0.81408472, 0.49263764, 0.11561787, 0.00748981, 0.06600323, 0.28441917 ,0.85482194, 0.96709556]
    transList = [rule110,spreadSeed,randomArray]
    transName = ["Rule 110","Spread of seeds","Random array" ]
    
    for num, trans in enumerate(transList):
        cell.setOnes(oneArray[0])
        test.setOnes(oneArray[0])
        cell.setTransition(trans)
        test.setTransition(trans)
        cell.MakeTimeSeries()
        test.MakeTimeSeries()
        matrix = np.copy(cell.timeSeries)
        testMat = np.copy(test.timeSeries)
        x_train, y_train = BuildingInOutput(matrix)
        x_test, y_test = BuildingInOutput(testMat)
        testingSeries = np.copy(testMat)
        for k in range(1,len(oneArray)):
            cell.setOnes(oneArray[k])
            test.setOnes(int(oneArray[k]/2))
            cell.MakeTimeSeries()
            test.MakeTimeSeries()
            matrix2 = np.copy(cell.timeSeries)
            testMat2 = np.copy(test.timeSeries)
            x_train_add, y_train_add = BuildingInOutput(matrix2)
            x_test_add, y_test_add = BuildingInOutput(testMat2)
            x_train = np.concatenate((x_train,x_train_add),axis=0)
            x_test = np.concatenate((x_test,x_test_add),axis=0)
            y_train = np.concatenate((y_train,y_train_add))
            y_test = np.concatenate((y_test,y_test_add))
            testingSeries = np.concatenate((testingSeries,np.copy(testMat2)))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(f"Transition: {transName[num]}" )
        ax.set_xlabel("Node")
        ax.set_yticks([0,int(cell.nRows/5),int(2*cell.nRows/5),int(3*cell.nRows/5),int(4*cell.nRows/5),cell.nRows], labels= [cell.nRows,int(4*cell.nRows/5),int(3*cell.nRows/5),int(2*cell.nRows/5),int(cell.nRows/5),0])
        ax.set_ylabel("Iteration")
        displaymatrix = np.flip(matrix,axis=0)
        ax.pcolor(displaymatrix,cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)

        print(f"Transition: {transName[num]} ")
        print("--> virgin Logistic regression")
        virginClf.fit(x_train,y_train)
        VexpectArray = virginClf.predict_proba(TaylorCoef(np.array(cell.inputConfig)))
        crossValv = cross_val_score(virginClf, x_test,y_test,scoring=brier_score)
        predictProbV = VexpectArray[:,1]
        file.write("transition: %s \n" % transName[num])
        file.write("---prove of concept--\n")
        file.write("Coeficients: %s\n" % virginClf.coef_)
        if(num):
            level=0
        else:
            level =1
        fileInfo(file,crossValv.mean(),HosmerLemeshow(testingSeries,VexpectArray,splits=len(oneArray)-1),predictProbV,trans,accuracyLevel=level)
    
        print("---> Grid search")
        grid.fit(x_train,y_train)
        bestPara = grid.cv_results_['params'][grid.best_index_]
        file.write("---> Grid search\\")
        file.write(f"best error: {grid.best_score_} using parametersetting {bestPara}\\")
        expectArray = grid.predict_proba(TaylorCoef(np.array(cell.inputConfig)))
        crossVal = cross_val_score(grid, x_test,y_test)
        predictProb = expectArray[:,1]
        fileInfo(file,crossVal.mean(),HosmerLemeshow(testingSeries,expectArray,splits=len(oneArray)-1),predictProb,trans,accuracyLevel=level)
        title = "Transition: " + transName[num]
        plot_learning_curve(refClf,title,x_train,y_train,scoring=brier_score)
        print("--> reference Logistic Regression")
        refClf.fit(x_train,y_train)
        RexpectArray = refClf.predict_proba(TaylorCoef(np.array(cell.inputConfig)))
        crossValr = cross_val_score(refClf, x_test,y_test,scoring=brier_score)
        predictProbR = RexpectArray[:,1]
        file.write("---improved version--\n")
        file.write("Coeficients: %s\n" % refClf.coef_)
        fileInfo(file,crossValr.mean(),HosmerLemeshow(testingSeries,RexpectArray,splits=len(oneArray)-1),predictProbR,trans,accuracyLevel=level)
    if(show):
        plt.show()

def Thesis_realData(show=True):

    refParam = {'penalty':'none','solver':'sag','tol':0.001, 'max_iter':1000}
    refClf = LogisticRegression(penalty=refParam['penalty'],solver=refParam['solver'],tol=refParam['tol'], max_iter=refParam['max_iter'])
    brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)

    plt.rcParams.update({'font.size': 23})
    fileName = "ca1D_real.dat"

    file = open(fileName,'w')

    file.write("----------------------------------\n")
    file.write("Simulation of real data:\n__________________________\n\n")
    rebuildLength = 50
    rebuildIter = 99
    rebuildOnes = 26
    file.write("length: %i\n"%rebuildLength)
    file.write("iterations: %i\n"%rebuildIter)
    file.write("initially occupied nodes: %i\n"%rebuildOnes)
    rebuildCell = CA(rebuildLength,rebuildIter,rebuildOnes)
    rule30 = [0, 0, 0, 1, 1, 1, 1, 0]
    physical = [1,0.2,1,1,1,0,0,0]
    rebuildTransList = [rule30,physical]
    rebuildName  =["deterministic rule", "physical interaction"]
    numStat = 100
    for number, transitions in enumerate(rebuildTransList):
        rebuildCell.setTransition(transitions)
        rebuildCell.MakeTimeSeries()
        rebuildMatrix = rebuildCell.timeSeries

        x_data, y_data = BuildingInOutput(rebuildMatrix)
        timeIndex = TimeSeriesSplit(4)
        for trainIndex, testIndex  in timeIndex.split(x_data):
            x_train, x_test = x_data[trainIndex], x_data[testIndex]
            y_train, y_test = y_data[trainIndex], y_data[testIndex]
        refClf.fit(x_train,y_train)
        ExpectedArray = refClf.predict_proba(TaylorCoef(np.array(rebuildCell.inputConfig)))
        predictProb = ExpectedArray[:,1]
        
        rebuildSto = RebuildMatrix(rebuildLength,rebuildIter,rebuildMatrix[0,:],predictProb)
        CompareGlobalFeatures(rebuildMatrix,rebuildSto,file)
        changeList = []
        occupayList = []
        densList = []
        entroList = []
        features = globalFeatures(rebuildSto)
        changeList.append(features["changeRate"])
        occupayList.append(features["occupation"])
        densList.append(features["neighborhoodDensity"])
        entroList.append(features["spatialEntropy"])
        for i in range(numStat):
            R = RebuildMatrix(rebuildLength,rebuildIter,rebuildMatrix[0,:],predictProb)
            features = globalFeatures(R)
            changeList.append(features["changeRate"])
            occupayList.append(features["occupation"])
            densList.append(features["neighborhoodDensity"])
            entroList.append(features["spatialEntropy"])
        changeRate = np.array(changeList)
        occupancy  = np.array(occupayList)
        neighDens  = np.array(densList)
        entropy    = np.array(entroList)
        
        file.write("Tranistion: %s\n" % rebuildName[number] )
        fileInfo(file, brier_score_loss(y_test,refClf.predict_proba(x_test)[:,1]),HosmerLemeshow(rebuildMatrix,ExpectedArray),predictProb,transitions,accuracyLevel = 2*(1-number))
        file.write("---> Statistic of %i rebuilding of time series with extracted probs" % numStat)
        file.write("Change Rate: %1.5f  +- %1.5f" % (changeRate.mean(),changeRate.std()))
        file.write("Lattice occupation: %1.5f  +- %1.5f" % (occupancy.mean(),occupancy.std()))
        file.write("density in neighborhood: %1.5f  +- %1.5f" % (neighDens.mean(),neighDens.std()))
        file.write("Entropy: %1.5f  +- %1.5f" % (entropy.mean(),entropy.std()))


        fig = plt.figure()
        fig.suptitle("Transition: %s" % rebuildName[number])
        ax = fig.add_subplot(1,2,1)
        ax.set_title("original" )
        ax.set_xlabel("Node")
        ax.set_yticks([0,int(rebuildCell.nRows/5),int(2*rebuildCell.nRows/5),int(3*rebuildCell.nRows/5),int(4*rebuildCell.nRows/5),rebuildCell.nRows], labels= [rebuildCell.nRows,int(4*rebuildCell.nRows/5),int(3*rebuildCell.nRows/5),int(2*rebuildCell.nRows/5),int(rebuildCell.nRows/5),0]) 
        ax.set_ylabel("Iteration")
        displaymatrix = np.flip(rebuildMatrix,axis=0)
        ax.pcolor(displaymatrix,cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
        ax2 = fig.add_subplot(1,2,2,sharey=ax)
        ax2.set_title("recreation" )
        ax2.set_xlabel("Node")
        #ax2.set_yticks([0,int(rebuildCell.nRows/5),int(2*rebuildCell.nRows/5),int(3*rebuildCell.nRows/5),int(4*rebuildCell.nRows/5),rebuildCell.nRows], labels= [rebuildCell.nRows,int(4*rebuildCell.nRows/5),int(3*rebuildCell.nRows/5),int(2*rebuildCell.nRows/5),int(rebuildCell.nRows/5),0])
        #ax2.set_ylabel("Iteration")
        displaymatrix2 = np.flip(rebuildSto,axis=0)
        ax2.pcolor(displaymatrix2,cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
        file.close()
    if(show):
        plt.show()

def Thesis_visuals(show=True):
    rule30 = [0, 0, 0, 1, 1, 1, 1, 0]
    rule110= [0, 1, 1, 0, 1, 1, 1, 0]
    rule184= [1, 0, 1, 1, 1, 0, 0, 0]
    rule54=  [0, 0, 1, 1, 1, 0, 0, 0]
    rule248= [1, 1, 1, 1, 1, 0, 0, 0]
    rule152= [1, 0, 0, 1, 1, 0, 0, 0]
    rule168= [1, 0, 1, 0, 1, 0, 0, 0]
    rule176= [1, 0, 1, 1, 0, 0, 0, 0]
    rule188= [1, 0, 1, 1, 1, 1, 0, 0]
    rule186= [1, 0, 1, 1, 1, 0, 1, 0]
    inter1 = [1, 0, 0.9, 1, 1, 0, 0, 0]
    inter2 = [1, 0, 0.5, 1, 1, 0, 0, 0]
    inter3 = [1, 0, 0.1, 1, 1, 0, 0, 0]
    rule204 = [1, 1, 0, 0, 1, 1, 0, 0]
    rule224 = [1, 1, 1, 0, 0, 0, 0, 0]
    diplayList = [rule54,rule248, rule184,inter1,inter2,inter3, rule152, rule168, rule176, rule188, rule186,  rule224,rule204, rule30, rule110 ]
    displayName = ["184","54","248","1 0 1 1 1 0 0 0","1 0 0.9 1 1 0 0 0","1 0 0.5 1 1 0 0 0", "1 0 0.1 1 1 0 0 0","1 0 0 1 1 0 0 0","168","176","188","186","Rule 224","Rule 204","Rule 30","Rule 110"
    ]
    ca = CA(50,99,26)
    
    plt.rcParams.update({'font.size': 27})
    for index, rule in enumerate(diplayList):
        ca.setTransition(rule)
        ca.MakeTimeSeries()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(displayName[index] )
        ax.set_xlabel("lattice node")
        ax.set_yticks([0,20,40,60,80,100], labels= [100,80,60,40,20,0])
        ax.set_ylabel("time step")
        displaymatrix = np.flip(ca.timeSeries,axis=0)
        ax.pcolor(displaymatrix,cmap = 'binary',edgecolors='gray',vmin=False,vmax=True)
    plt.show()


def main():
    Thesis_visuals(show=False)
    Thesis_proveOfConcept(show=False)
    Thesis_realData()
    
if __name__ == '__main__':
    main()
