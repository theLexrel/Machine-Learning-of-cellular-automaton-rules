import sys
sys.path.insert(1,"/home/max/Dokumente/Code/BIO-LGCA/biolgca-develop")
from lgca import get_lgca
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer


def brier_error(y_true,y_pred):
    # implementation of brier error that doen't return a negative value when used in GridSearch of cross validation
    N, K = np.shape(y_pred)
    occurred = np.zeros((N,K))
    states = np.unique(y_true)
    for i in range(N):
        index = int(np.argwhere(states==y_true[i])[0][0])
        occurred[index] = 1
    return -np.abs(np.sum(np.square(occurred-y_pred))/N)/len(states)

def timeSeriesAnalysis(timeSeries):
    time, nodes, channels = np.shape(timeSeries)
    aLdens = 0
    aLflux = 0
    aNdens = 0
    aNflux = 0
    aFlux = 0
    aDens = 0
    aChange  = 0
    for t in range(time):
        aLdens += np.sum(timeSeries[t,:,:])/(time*nodes*channels)
        aLflux += (np.sum(timeSeries[t,:,0],axis=0)-np.sum(timeSeries[t,:,1],axis=0))/(time*nodes)
        for r in range(nodes):
            aDens += np.sum(timeSeries[t,r,:])/(time*nodes*channels)
            aFlux += (int(timeSeries[t,r,0])-int(timeSeries[t,r,1]))/(time*nodes)
            aNdens += (np.sum(timeSeries[t,(r-1)%nodes,:])+np.sum(timeSeries[t,(r+1)%nodes,:]))/(2*time*nodes*channels)
            aNflux += (int(timeSeries[t,(r-1)%nodes,0])+int(timeSeries[t,(r+1)%nodes,0])-int(timeSeries[t,(r-1)%nodes,1])-int(timeSeries[t,(r+1)%nodes,1]))/(2*time*nodes)
            if(t):
                aChange += (np.sum(timeSeries[t,r,:])-np.sum(timeSeries[t-1,r,:]))/(time*nodes)

    anArray = {
        "densityLattice" : aLdens,
        "densityNeighborhood" : aNdens,
        "densityNode" : aDens,
        "fluxLattice" : aLflux,
        "fluxNeighborhood" : aNflux,
        "fluxNode" : aFlux,
        "changeRate" : aChange
        }
    return anArray

class MLonLGCA:
    """
    This class is the heart of the code. There are multiple methods which take in a time series, split it into
    inputs and outputs and feed both into the ML algorithm for analysis.
    """
    def __init__(self,_Restchannels,_interaction='total',_interactionMode=True,_delta=False):
        """
        There are 2 modes set up:
        _interactionMode=True:  this is the ML algorithm version of section 6.2, with which the 
                                interaction strength can be extracted
        _interactionMode=False: the version of section 6.3 which performs a linear approximation
        ........
        _interaction: dictates the interaction term input if _interactionMode=True
            -'alignment': input for interaction alignment, total flux in interaction neighborhood
            -'aggregation': input for aggregation, mass gradient in interaction neighborhood
            -'total': the 'total version' contains both of the above
            - else: if non of the above, then random walk is implemented with input = 1
        _delta: dictates whether or not the delta-fct approx is included if _interactionMode=False
        """
        self.interImple = _interaction
        self.interactionMode = _interactionMode
        self.restChannels = _Restchannels
        self.channels = 2 + _Restchannels       # +2 velocity channels
        self.nConfigs = (2**self.channels)**3    # left neighboor, middle, right
        self.delta = _delta
        if(_interactionMode):
            if(_interaction=='total'):
                self.nFeatures = 2 + self.channels+1
            else:
                self.nFeatures = 1 + self.channels+1
        else:
            if(_delta):
                self.nFeatures = 6 + self.channels+1
            else:
                self.nFeatures = 6 


    def _hypothesis(self,input):
        ''''
            builds the fearues representation of an raw input containing the node state
            and the state of the neighborhood 
        '''
        vecX = np.zeros(self.nFeatures)
        if(self.interactionMode):
            if(self.interImple=='total'):
                vecX[0] = (int(input[0,0])-int(input[0,1]) )+( int(input[2,0])-int(input[2,1]))
                vecX[1] = np.sum(input[2,:])-np.sum(input[0,:])
                #vecX[2] = np.sum(input[1,:])
                # approximating the delta function 
                for i in range(self.channels+1):
                    vecX[i+2] = (np.sum(input[1,:])-i)**2
            else:
                if(self.interImple=='alignment'):
                    vecX[0] = (int(input[0,0])-int(input[0,1]) )+( int(input[2,0])-int(input[2,1]))
                elif(self.interImple == 'aggregation'):
                    vecX[0] = np.sum(input[2,:])-np.sum(input[0,:])
                else:
                    vecX[0] = 1
                # approximating the delta function 
                for i in range(self.channels+1):
                    vecX[i+1] = (np.sum(input[1,:])-i)**2
            return vecX
        else:
            vecX[0] = np.sum(input[1,:])
            vecX[1] = int(input[1,0])-int(input[1,1])
            vecX[2] = np.sum(input[0,:])
            vecX[3] = np.sum(input[2,:])
            vecX[4] = int(input[0,0])-int(input[0,1])
            vecX[5] = int(input[2,0])-int(input[2,1])
            if(self.delta):
                for i in range(self.channels+1):
                    vecX[i+6] = (np.sum(input[1,:])-i)**2
            return -vecX


    def _possibleInputs(self):            
        inputs = np.zeros((self.nConfigs,self.nFeatures))
        decimal  = np.arange(self.nConfigs)
        for i in range(self.nConfigs):
            rawConfigs = np.zeros((3,self.channels), dtype=np.bool_)   # config occupation init as empty
            for k in range(self.channels*3):
                if(decimal[i]&2**(3*self.channels-1-k)):        # check if the bit is flipped
                    rawConfigs[int(k/self.channels), k % self.channels] = True
            inputs[i,:] =  self._hypothesis(rawConfigs)
        return inputs

    def _nodeStateToConfig(self,number):
        '''
            A small function to covert the node state into a configuration vector
        '''
        config = np.zeros(self.channels, dtype=np.bool_)
        for i in range(self.channels):
            if(number & 2**(self.channels - i -1)):
                config[i] = True
        return config
        

    def _inputNumber(self,config):
        configLenght, channels = np.shape(config)
        array = np.array(config)
        number = 0
        for n in range(configLenght):
            for c in range(channels):
                number += array[n,c]*2**(configLenght*channels-n*channels-c-1)
        return number


    def buildInAndOutput(self,rawData):
        """
        The raw data is a 3-dim matrix consiting of:
        - [t,:,:] the lattice configs at time step t (t+1 entries due to initial state)
        - [t,r,:] the node state of node r at time step t
        - [t,r,i] the state of the channel i in node r at time step t
        -- order of channels: right, left, rest1, rest2,....
        The boundary conditions are for now limited to periodic boundary conditions.
        Only nearest neighbors are implemented
        _________________________________________________________________________
        -------------------------------------------------------------------------
        
        Since machine learing might be able the have a vector as input but unbale to map it to
        another vector (y \in \mathbb{N}) the node state has to be translated into a number and 
        Logistic regression has to work for a multi_class problem:
            convert the node state into binary and translate this into decimal like for the
            Wolfram convention for 1D CA rules but for N channels in a node:
            r,l,r1,r2,...,rN-2 -> 101000...1 -> 1*2^N-1 + 0*2^N-2 + 1*2^N-3 + ... + 1*2^0 = n 
        """
        """
        alternative version of self.buildInAndOutput, where the transportation step is 
        reversed by shifting the whole lattice back instead of element wise
        """
        timestepsNumber, latticeLength, channelNumber = np.shape(rawData)
        dataEntries = timestepsNumber*latticeLength
        originalInput = np.zeros((dataEntries,3,channelNumber),dtype=np.bool_)
        originalOutput = np.zeros((dataEntries,channelNumber),dtype=np.bool_)
        for t in range(timestepsNumber-1):
            preMoveLattice = np.empty_like(rawData[t+1,:,:])
            preMoveLattice[:,0] = np.roll(rawData[t+1,:,0],-1,axis=0)
            preMoveLattice[:,1] = np.roll(rawData[t+1,:,1],1,axis=0)
            preMoveLattice[:,2:] = np.copy(rawData[t+1,:,2:])
            for node in range(latticeLength):
                originalInput[t*latticeLength+node,0,:] = rawData[t,(node-1)%latticeLength,:]
                originalInput[t*latticeLength+node,1,:] = rawData[t,node,:]
                originalInput[t*latticeLength+node,2,:] = rawData[t,(node+1)%latticeLength,:]

                originalOutput[t*latticeLength+node,:] = preMoveLattice[node,:]     # rest channels
    
        vecX = np.zeros((dataEntries,self.nFeatures))
        vecY = np.zeros(dataEntries)
        for entry in range(dataEntries):
            for i in range(channelNumber):  
                vecY[entry] +=  2**(channelNumber-1-i)*originalOutput[entry,i]
            vecX[entry,:] =  self._hypothesis(originalInput[entry,:,:])
        return vecX,vecY

    def rebuildTimeSeries(self, original,probArray,nodeStates):
        """
           alternative version of self.rebuildTimeSeries where at first the pre-transportation
           step lattice config is fully build and afterwards the transportation step performed
           lattice wide by shifting the array elements
        """
        """
            Input: 
            - original time series -> use to get number of time steps + initial lattice config
            - prob. array prediction from LogReg, format = (number possible inputs, prob node state) 
            Function: 
        """
        time, nodes, channels = np.shape(original)
        nInputs, nStates = np.shape(probArray)
        # rebuild time series initialization
        rebuild = np.zeros((time,nodes,channels), dtype=np.bool_)
        rebuild[0,:,:] = np.copy(original[0,:,:])
        for t in range(time-1):
            preMoveConfig = np.empty_like(rebuild[t,:,:])
            for node in range(nodes):       
                inputNumber = self._inputNumber([rebuild[t,(node-1)%nodes,:],rebuild[t,node,:],rebuild[t,(node+1)%nodes,:]])    # get the corresponding number to probArray format
                state = np.random.choice(nStates,p=probArray[inputNumber,:])                # choice one of the node state with the extracted distribution
                preMoveConfig[node,:] = self._nodeStateToConfig(int(nodeStates[state]))
            rebuild[t+1,:,0] = np.roll(preMoveConfig[:,0],1,axis=0)       # transported to the right node
            rebuild[t+1,:,1] =  np.roll(preMoveConfig[:,1],-1,axis=0) # transported to the left node
            rebuild[t+1, :, 2:]  = np.copy(preMoveConfig[:,2:])    # rested at the current node 
    
        return rebuild

    def _timeSeriesExtraction(self, timeSeries, plt_show=True,build_Stat = True, grid_search=False):
        """
        This method will take in a time series, perfrom the algorithm and reconstruct the time series.
        Afterwards, both are plotted together.
        """
        time, latticeLength, channels = np.shape(timeSeries)
        timeSteps = time-1
        if(self.interactionMode):
            param_grid = [{'solver' : ['saga'],      'penalty' : ['l1'], 'tol' : [1e-4], 'max_iter' : [10000], 'multi_class' : ['multinomial'], 'C' : [0.01,0.1,1,10,100],'fit_intercept':[False]}] 
        else:
            param_grid = [{'solver' : ['saga'],      'penalty' : ['l2'], 'tol' : [1e-4], 'max_iter' : [10000], 'multi_class' : ['multinomial'], 'C' : [0.01,0.1,1,10,100],'fit_intercept':[True]}] 
        vecX, vecY = self.buildInAndOutput(timeSeries)

        timeIndex = TimeSeriesSplit(4)
        for trainIndex, testIndex  in timeIndex.split(vecX):
            x_train, x_test = vecX[trainIndex], vecX[testIndex]
            y_train, y_test = vecY[trainIndex], vecY[testIndex]
        if(self.interactionMode):
            bestParams =  {'solver' : 'saga','penalty' :'l1','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':False}
        else:
            bestParams =  {'solver' : 'saga','penalty' :'l2','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':True}
        brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)

        if(grid_search):
            grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, scoring=brier_score )
            print("--> grid search")
            grid.fit(x_train,y_train)
            print(f"parametrization: {grid.cv_results_['params'][grid.best_index_]}")
            print(f"resulting score: {grid.score(x_test,y_test)}")
            
        clf=LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'],
            solver=bestParams['solver'] , tol=bestParams['tol'], max_iter=bestParams['max_iter'], multi_class=bestParams['multi_class'],fit_intercept=bestParams['fit_intercept'])

        
        clf.fit(x_train,y_train)
        crossVal = cross_val_score(clf, x_test,y_test,scoring=brier_score)
        print(f"classes seen by the classifier: {clf.classes_}")
        print("brier score: %s \pm %s \n" % (crossVal.mean(),crossVal.std()))
        probArray = clf.predict_proba(self._possibleInputs())
        #print(f"Hosmer-Lemeshow test: {HosmerLemeshow_test(timeSeries, probArray)}")

        print(f"found coefficients: \n{clf.coef_} \n intersection: \n{clf.intercept_}")

        rebuild = self.rebuildTimeSeries(timeSeries, probArray, np.unique(y_train))
        originalFeautures = timeSeriesAnalysis(timeSeries)
        rebuildFeatures = timeSeriesAnalysis(rebuild)
        #print(f"unique arrangement of the channels in the rebuild: \n {np.unique(rebuild,axis=-1)}")
        if(build_Stat):
            numStat = 100
            rebuildFeaturesArray = np.zeros(len(rebuildFeatures))
            featureNames = ["densityLattice", "densityNeighborhood","densityNode", "fluxLattice", "fluxNeighborhood", "fluxNode","changeRate"]   
            for i in range(numStat):
                rebuildTemp = self.rebuildTimeSeries(timeSeries, probArray, np.unique(y_train))
                featuresTemp = timeSeriesAnalysis(rebuildTemp)
                for j in range(len(featureNames)):
                    rebuildFeaturesArray[j] += featuresTemp[featureNames[j]]
            print(f"getting a statistic using {numStat} versions")
            for i in range(len(featureNames)):
                print(f"""comparing {featureNames[i]}: 
                recreation \t\t{originalFeautures[featureNames[i]]} \t {rebuildFeaturesArray[i]/numStat} \t -> {100*(rebuildFeaturesArray[i]/numStat-originalFeautures[featureNames[i]])/originalFeautures[featureNames[i]]}%""")
            
        plt.rcParams.update({'font.size': 23})
        cmap = "gist_heat"
        oriMap = plt.cm.get_cmap(cmap)
        revers = oriMap.reversed()
        fig1 =  plt.figure()
        fig1.suptitle(f"Density plot with {self.restChannels} restchannels\n{self.interImple} version")
        ax11 = fig1.add_subplot(1,2,1)
        ax11.set_ylabel("time")
        ax11.set_yticks([0,int((timeSteps+1)/5),int(2*(timeSteps+1)/5),int(3*(timeSteps+1)/5),int(4*(timeSteps+1)/5),timeSteps+1], labels= [timeSteps+1,int(4*(timeSteps+1)/5),int(3*(timeSteps+1)/5),int(2*(timeSteps+1)/5),int((timeSteps+1)/5),0])
        ax11.set_xlabel("node")
        ax11.pcolor(np.flip(np.sum(timeSeries, axis=2),axis=0), edgecolors = 'gray',cmap=revers)
        ax11.set_title("original")
        ax12 = fig1.add_subplot(1,2,2, sharey=ax11)
        ax12.set_xlabel("node")
        im1 = ax12.pcolor(np.flip(np.sum(rebuild, axis=2), axis=0), edgecolors = 'gray',cmap=revers)
        ax12.set_title("rebuild")

        divider1 = make_axes_locatable(ax12)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        if(plt_show):
            plt.show()

    def reconstructionTimeSeries(self, timeSeries, build_Stat = False):
        """
        This method will take in a time series, perfrom the algorithm and reconstruct the time series.
        The reconstructed time series are the function outputs.
        """
        vecX, vecY = self.buildInAndOutput(timeSeries)

        timeIndex = TimeSeriesSplit(4)
        for trainIndex, testIndex  in timeIndex.split(vecX):
            x_train, x_test = vecX[trainIndex], vecX[testIndex]
            y_train, y_test = vecY[trainIndex], vecY[testIndex]
        if(self.interactionMode):
            bestParams =  {'solver' : 'saga','penalty' :'l1','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':False}
        else:
            bestParams =  {'solver' : 'sag','penalty' :'l2','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':True}
       
        brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)
     
        clf=LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'],
            solver=bestParams['solver'] , tol=bestParams['tol'], max_iter=bestParams['max_iter'], multi_class=bestParams['multi_class'],fit_intercept=bestParams['fit_intercept'])

        
        clf.fit(x_train,y_train)
        crossVal = cross_val_score(clf, x_test,y_test,scoring=brier_score)
        print(f"classes seen by the classifier: {clf.classes_}")
        print("brier score: %s \pm %s \n" % (crossVal.mean(),crossVal.std()))
        probArray = clf.predict_proba(self._possibleInputs())
        print(f"for version {self.interImple} found coefficients: \n{clf.coef_}")   

        rebuild = self.rebuildTimeSeries(timeSeries, probArray, np.unique(y_train))
        originalFeautures = timeSeriesAnalysis(timeSeries)
        rebuildFeatures = timeSeriesAnalysis(rebuild)
        #print(f"unique arrangement of the channels in the rebuild: \n {np.unique(rebuild,axis=-1)}")
        if(build_Stat):
            numStat = 100
            rebuildFeaturesArray = np.zeros(len(rebuildFeatures))
            featureNames = ["densityLattice", "densityNeighborhood","densityNode", "fluxLattice", "fluxNeighborhood", "fluxNode","changeRate"]   
            for i in range(numStat):
                rebuildTemp = self.rebuildTimeSeries(timeSeries, probArray, np.unique(y_train))
                featuresTemp = timeSeriesAnalysis(rebuildTemp)
                for j in range(len(featureNames)):
                    rebuildFeaturesArray[j] += featuresTemp[featureNames[j]]
            print(f"getting a statistic using {numStat} versions")
            for i in range(len(featureNames)):
                print(f"""comparing {featureNames[i]}: 
                recreation \t\t{originalFeautures[featureNames[i]]} \t {rebuildFeaturesArray[i]/numStat} \t -> {100*(rebuildFeaturesArray[i]/numStat-originalFeautures[featureNames[i]])/originalFeautures[featureNames[i]]}%""")
        if(self.interactionMode):
            return rebuild
        else:
            rebuild2 = self.rebuildTimeSeries(timeSeries, probArray, np.unique(y_train))
    
            probArrayR = np.copy(probArray)
            for i in range(len(probArrayR[0,:])):
                if(i):
                    probArrayR[0,i] = 0
                else:
                    probArrayR[0,i] = 1
            rebuildR = self.rebuildTimeSeries(timeSeries, probArrayR, np.unique(y_train))
            return rebuild, rebuild2, rebuildR

def Thesis_ExtractionInteractionStrenght(timeSeries, interaction='unknonw',show=True):
    """
    Run this function to get the results for chapter 6.2.2
    """
    time,latticeLenght,channels = np.shape(timeSeries)
    restChannels = channels-2
    lMs = MLonLGCA(restChannels,_interaction='alignment')
    align = lMs.reconstructionTimeSeries(timeSeries)
    
    lMm = MLonLGCA(restChannels,_interaction='aggregation')
    aggreg = lMm.reconstructionTimeSeries(timeSeries)
    
    lMl = MLonLGCA(restChannels,_interaction='random walk')
    rand = lMl.reconstructionTimeSeries(timeSeries)
    plt.rcParams.update({'font.size': 23})
    cmap = "gist_heat"
    oriMap = plt.cm.get_cmap(cmap)
    revers = oriMap.reversed()
    fig1 =  plt.figure()
    fig1.suptitle(f"Density plot for {interaction} performed on LGCA with {restChannels} rest channels")
    ax11 = fig1.add_subplot(1,4,1)
    ax11.set_ylabel("time")
    ax11.set_yticks([0,int(time/5),int(2*time/5),int(3*time/5),int(4*time/5),time], labels= [time,int(4*time/5),int(3*time/5),int(2*time/5),int(time/5),0])
    ax11.set_xlabel("node")
    ax11.pcolor(np.flip(np.sum(timeSeries, axis=2),axis=0), edgecolors = 'gray',cmap = revers)
    ax11.set_title("original")
    ax12 = fig1.add_subplot(1,4,2, sharey=ax11)
    ax12.set_xlabel("node")
    ax12.pcolor(np.flip(np.sum(align, axis=2), axis=0), edgecolors = 'gray',cmap = revers)
    ax12.set_title("alignment")
    ax13 = fig1.add_subplot(1,4,3, sharey=ax11)
    ax13.set_xlabel("node")
    ax13.pcolor(np.flip(np.sum(aggreg, axis=2), axis=0), edgecolors = 'gray',cmap = revers)
    ax13.set_title("aggregation")
    ax14 = fig1.add_subplot(1,4,4, sharey=ax11)
    ax14.set_xlabel("node")
    im1 = ax14.pcolor(np.flip(np.sum(rand, axis=2), axis=0), edgecolors = 'gray',cmap = revers)
    ax14.set_title("random walk")

    divider1 = make_axes_locatable(ax14)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    lMt = MLonLGCA(restChannels,_interaction='total')
    tot = lMt.reconstructionTimeSeries(timeSeries)
    fig2 =  plt.figure()
    fig2.suptitle(f"Density plot for {interaction} performed on LGCA with {restChannels} rest channels\n total version LogReg")
    ax21 = fig2.add_subplot(1,2,1)
    ax21.set_ylabel("time")
    ax21.set_yticks([0,int(time/5),int(2*time/5),int(3*time/5),int(4*time/5),time], labels= [time,int(4*time/5),int(3*time/5),int(2*time/5),int(time/5),0])
    ax21.set_xlabel("node")
    ax21.pcolor(np.flip(np.sum(timeSeries, axis=2),axis=0), edgecolors = 'gray',cmap = revers)
    ax21.set_title("original")
    ax22 = fig2.add_subplot(1,2,2, sharey=ax21)
    ax22.set_xlabel("node")
    im2 = ax22.pcolor(np.flip(np.sum(tot, axis=2), axis=0), edgecolors = 'gray',cmap=revers)
    ax22.set_title("rebuild")

    divider2 = make_axes_locatable(ax22)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    if(show):
        plt.show()

def Thesis_GeneralApproach(timeSeries, interaction='unknown',show=True):
    """
    Run this function to get the results for chapter 6.3.2
    """
    time,latticeLenght,channels = np.shape(timeSeries)
    restChannels = channels-2
    lMW = MLonLGCA(restChannels,_interactionMode=False)
    rebuildW, rebuild2W, rebuildRW = lMW.reconstructionTimeSeries(timeSeries)

    lMD = MLonLGCA(restChannels,_interactionMode=False,_delta=True)
    rebuildD, rebuild2D, rebuildRD = lMD.reconstructionTimeSeries(timeSeries)
    plt.rcParams.update({'font.size': 23})
    cmapName = "gist_heat"
    oriMap = plt.cm.get_cmap(cmapName)
    cmap = oriMap.reversed()
    fig1 =  plt.figure()
    fig1.suptitle(f"Density plot for {interaction} with {restChannels} restchannels")
    ax11 = fig1.add_subplot(1,4,1)
    ax11.set_ylabel("time")
    ax11.set_yticks([0,int(time/5),int(2*time/5),int(3*time/5),int(4*time/5),time], labels= [time,int(4*time/5),int(3*time/5),int(2*time/5),int(time/5),0])
    ax11.set_xlabel("node")
    ax11.pcolor(np.flip(np.sum(timeSeries, axis=2),axis=0), edgecolors = 'gray',cmap=cmap)
    ax11.set_title("original")
    ax12 = fig1.add_subplot(1,4,2, sharey=ax11)
    ax12.set_xlabel("node")
    ax12.pcolor(np.flip(np.sum(rebuildW, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax12.set_title("rebuild")

    ax13 = fig1.add_subplot(1,4,3, sharey=ax11)
    ax13.set_xlabel("node")
    ax13.pcolor(np.flip(np.sum(rebuild2W, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax13.set_title("repetition")
    ax14 = fig1.add_subplot(1,4,4, sharey=ax11)
    ax14.set_xlabel("node")
    im1 = ax14.pcolor(np.flip(np.sum(rebuildRW, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax14.set_title("something from something")
    divider1 = make_axes_locatable(ax14)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

  
    fig2 =  plt.figure()
    fig2.suptitle(f"Density plot for {interaction} with {restChannels} restchannels\n with delta-function approximation")
    ax21 = fig2.add_subplot(1,4,1)
    ax21.set_ylabel("time")
    ax21.set_yticks([0,int(time/5),int(2*time/5),int(3*time/5),int(4*time/5),time], labels= [time,int(4*time/5),int(3*time/5),int(2*time/5),int(time/5),0])
    ax21.set_xlabel("node")
    ax21.pcolor(np.flip(np.sum(timeSeries, axis=2),axis=0), edgecolors = 'gray',cmap=cmap)
    ax21.set_title("original")
    ax22 = fig2.add_subplot(1,4,2, sharey=ax11)
    ax22.set_xlabel("node")
    ax22.pcolor(np.flip(np.sum(rebuildD, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax22.set_title("rebuild")

    ax23 = fig2.add_subplot(1,4,3, sharey=ax11)
    ax23.set_xlabel("node")
    ax23.pcolor(np.flip(np.sum(rebuild2D, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax23.set_title("repetition")
    ax24 = fig2.add_subplot(1,4,4, sharey=ax11)
    ax24.set_xlabel("node")
    im2 = ax24.pcolor(np.flip(np.sum(rebuildRD, axis=2), axis=0), edgecolors = 'gray',cmap=cmap)
    ax24.set_title("something from something")
    divider2 = make_axes_locatable(ax24)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    if(show):
        plt.show()

def main():
    """
    As all of the above functions has time series as inputs, they are produced here.
    """
    l = 100
    iter = 199
    alignment_lgca = get_lgca(restchannels = 0, geometry='lin', interaction = 'alignment',density=0.15, bc='pbc', dims = l, ib=False,beta=2)
    alignment_lgca.timeevo(timesteps= iter, record = True)
    aggregation_lgca = get_lgca(restchannels = 1, geometry='lin', interaction = 'aggregation',density=0.1, bc='pbc', dims = l, ib=False,beta=3.5)
    aggregation_lgca.timeevo(timesteps= iter, record = True)
    randomWalk_lgca = get_lgca(restchannels = 2, geometry='lin', interaction = 'random_walk',density=0.125, bc='pbc', dims = l, ib=False)
    randomWalk_lgca.timeevo(timesteps= iter, record = True)
    timeSeries = [alignment_lgca.nodes_t,aggregation_lgca.nodes_t,randomWalk_lgca.nodes_t]
    interaction = ["alignment","aggregation","random walk"]
    for i in range(len(timeSeries)):
        Thesis_ExtractionInteractionStrenght(timeSeries[i], interaction[i],show=False)
    for i in range(2):
        Thesis_GeneralApproach(timeSeries[i], interaction[i],show=False)
    plt.show()
        
if __name__=='__main__':
    main()
