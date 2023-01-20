import sys
sys.path.insert(1,"/home/max/Dokumente/Code/BIO-LGCA/biolgca-develop")
#sys.path.insert(1,"/home/max/Dokumente/Uni/Master/Masterarbeit/Code/BIO LGCA/biolgca-develop")
#sys.path.insert(1,"/mnt/1dfb303e-22df-4c6f-ad16-e8242069b27a/home/max/Dokumente/Uni-Zeug/Master/Masterarbeit/Code/BIO-LGCA/biolgca-develop")
from lgca import get_lgca
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.metrics import make_scorer



def timeSeriesAnalysis(timeSeries):
    time, nodesX, nodesY, channels = np.shape(timeSeries)
    aLdens = 0
    aLfluxX = 0
    aLfluxY = 0
    aNdens = 0
    aNfluxX = 0
    aFluxX = 0
    aNfluxY = 0
    aFluxY = 0
    aDens = 0
    aChange  = 0
    for t in range(time):
        aLdens += np.sum(timeSeries[t,:,:,:])/(time*nodesX*nodesY*channels)
        aLfluxX += (np.sum(timeSeries[t,:,0],axis=0)-np.sum(timeSeries[t,:,2],axis=0))/(time*nodesX*nodesY)
        aLfluxY += (np.sum(timeSeries[t,:,1],axis=0)-np.sum(timeSeries[t,:,3],axis=0))/(time*nodesX*nodesY)
        for x in range(nodesX):
            for y in range(nodesY):
                aDens += np.sum(timeSeries[t,x,y,:])/(time*nodesX*nodesY*channels)
                aFluxX += (int(timeSeries[t,x,y,0])-int(timeSeries[t,x,y,2]))/(time*nodesX*nodesY)
                aFluxY += (int(timeSeries[t,x,y,1])-int(timeSeries[t,x,y,3]))/(time*nodesX*nodesY)
                aNdens += (np.sum(timeSeries[t,(x-1)%nodesX,y,:])+np.sum(timeSeries[t,(x+1)%nodesX,y,:])+np.sum(timeSeries[t,x,(y+1)%nodesY,:])+np.sum(timeSeries[t,x,(y-1)%nodesY,:]))/(4*time*nodesX*nodesY*channels)
                aNfluxX += (int(timeSeries[t,x,(y-1)%nodesY,0])+int(timeSeries[t,x,(y+1)%nodesY,0])+int(timeSeries[t,(x-1)%nodesX,y,0])+int(timeSeries[t,(x+1)%nodesX,y,0])-int(timeSeries[t,(x-1)%nodesX,y,2])-int(timeSeries[t,(x+1)%nodesX,y,2])-int(timeSeries[t,x,(y-1)%nodesY,2])-int(timeSeries[t,x,(y+1)%nodesY,2]))/(4*time*nodesX*nodesY)
                aNfluxY += (int(timeSeries[t,x,(y-1)%nodesY,1])+int(timeSeries[t,x,(y+1)%nodesY,1])+int(timeSeries[t,(x-1)%nodesX,y,1])+int(timeSeries[t,(x+1)%nodesX,y,1])-int(timeSeries[t,(x-1)%nodesX,y,3])-int(timeSeries[t,(x+1)%nodesX,y,3])-int(timeSeries[t,x,(y-1)%nodesY,3])-int(timeSeries[t,x,(y+1)%nodesY,3]))/(4*time*nodesX*nodesY)
                if(t):
                    aChange += (np.sum(timeSeries[t,x,y,:])-np.sum(timeSeries[t-1,x,y,:]))/(time*nodesX*nodesY)

    anArray = {
        "densityLattice" : aLdens,
        "densityNeighborhood" : aNdens,
        "densityNode" : aDens,
        "fluxLatticeVertical" : aLfluxY,
        "fluxNeighborhoodVertical" : aNfluxY,
        "fluxNodeVertical" : aFluxY,
        "fluxLatticeHorizontal" : aLfluxX,
        "fluxNeighborhoodHorizontal" : aNfluxX,
        "fluxNodeHorizontal" : aFluxX,
        "changeRate" : aChange
        }
    return anArray


def brier_error(y_true,y_pred):
    # implementation of brier error that doen't return a negative value when used in GridSearch of cross validation
    N, K = np.shape(y_pred)
    occurred = np.zeros((N,K))
    states = np.unique(y_true)
    for i in range(N):
        index = int(np.argwhere(states==y_true[i])[0][0])
        occurred[index] = 1
    return -np.abs(np.sum(np.square(occurred-y_pred))/N)/len(states)


class MLonLGCA:

    def __init__(self,_Restchannels, _Neighborhood = 'Neumann',_interaction='total',_interactionMode=True, _delta=False):
        
        self.restChannels = _Restchannels
        self.velocityChannels = 4
        self.channels = self.velocityChannels+ _Restchannels       # +4 velocity channels
        if(_Neighborhood == 'Moore'):
            self.nNeighbors = 8
            self.neigh = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        elif(_Neighborhood == 'Neumann'):
            self.nNeighbors = 4
            self.neigh = [(0,1),(1,0),(0,-1),(-1,0)]
        else:
            self.nNeighbors = 4
            self.neigh = [(0,1),(1,0),(0,-1),(-1,0)]
        self.nNodes = self.nNeighbors+1    
        self.nConfigs = (2**self.channels)**self.nNodes    # left neighboor, middle, right
        self.interaction=_interaction
        self.interactionMode = _interactionMode
        self.delta = _delta
        if(_interactionMode):
            if(_interaction=='total'):
                self.nFeatures = 4 + self.channels+1
            else:
                self.nFeatures = 2 + self.channels+1
        else:
            if(_delta):
                self.nFeatures = 3+3*self.nNeighbors+ self.channels+1
            else:
                self.nFeatures = 3+3*self.nNeighbors

    def _hypothesis(self,input):
        ''''
            builds the fearues representation of an raw input containing the node state
            and the state of the neighborhood;
            First element must be the node state and the following entries the neighborhood 
        '''
        
        vecX = np.zeros(self.nFeatures)
        if(self.interactionMode):
            if(self.interaction=='total'):
                vecX[0] = (int(input[1,0])-int(input[1,2]) )+( int(input[2,0])-int(input[2,2]))+(int(input[3,0])-int(input[3,2]) )+( int(input[4,0])-int(input[4,2]))
                vecX[1] = (int(input[1,1])-int(input[1,3]) )+( int(input[2,1])-int(input[2,3]))+(int(input[3,1])-int(input[3,3]) )+( int(input[4,1])-int(input[4,3]))
                vecX[2] = np.sum(input[2,:])-np.sum(input[4,:])
                vecX[3] = np.sum(input[1,:])-np.sum(input[3,:])
                #vecX[2] = np.sum(input[1,:])
                # approximating the delta function 
                for i in range(self.channels+1):
                    vecX[i+4] = (np.sum(input[0,:])-i)**2
            else:
                if(self.interaction=='alignment'):
                    vecX[0] = (int(input[1,0])-int(input[1,2]) )+( int(input[2,0])-int(input[2,2]))+(int(input[3,0])-int(input[3,2]) )+( int(input[4,0])-int(input[4,2]))
                    vecX[1] = (int(input[1,1])-int(input[1,3]) )+( int(input[2,1])-int(input[2,3]))+(int(input[3,1])-int(input[3,3]) )+( int(input[4,1])-int(input[4,3]))
                elif(self.interaction == 'aggregation'):
                    vecX[0] = np.sum(input[2,:])-np.sum(input[4,:])
                    vecX[1] = np.sum(input[1,:])-np.sum(input[3,:])
                else:
                    vecX[0] = 1#np.sum(input[0,:])
                    vecX[1] = 1#int(input[0,0])-int(input[0,2])+int(input[0,1])-int(input[0,3])
                # approximating the delta function 
                for i in range(self.channels+1):
                    vecX[i+2] = (np.sum(input[0,:])-i)**2
            return vecX
        else:
            vecX[0] = np.sum(input[0,:])
            vecX[1] = int(input[0,0])-int(input[0,2])
            vecX[2] = int(input[0,1])-int(input[0,3])
            for j in range(self.nNeighbors):
                vecX[j+3] = np.sum(input[j+1,:])
                vecX[j+3+self.nNeighbors] = int(input[j+1,0])-int(input[j+1,2])
                vecX[j+3+2*self.nNeighbors] = int(input[j+1,1])-int(input[j+1,3])
            if(self.delta):
                for i in range(self.channels+1):
                    vecX[i+3+3*self.nNeighbors] = (np.sum(input[1,:])-i)**2
            return -vecX



    def _nodeStateToConfig(self,number):
        '''
            A small function to covert the node state into a configuration vector
        '''
        config = np.zeros(self.channels, dtype=np.bool_)
        for i in range(self.channels):
            if(number & 2**(self.channels - i -1)):
                config[i] = True
        return config

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
        timestepsNumber, latticeRows,latticeCols, channelNumber = np.shape(rawData)
        dataEntries = timestepsNumber*latticeRows*latticeCols
        latticeDims = latticeCols*latticeRows
        originalInput = np.zeros((dataEntries,self.nNodes,channelNumber),dtype=np.bool_)
        originalOutput = np.zeros((dataEntries,channelNumber),dtype=np.bool_)
        for t in range(timestepsNumber-1):
            perMoveLattice = np.zeros((latticeRows,latticeCols,channelNumber),dtype=np.bool_)
            perMoveLattice[:,:,0] = np.roll(rawData[t+1,:,:,0],-1,axis=0)
            perMoveLattice[:,:,1] = np.roll(rawData[t+1,:,:,1],-1,axis=1)
            perMoveLattice[:,:,2] = np.roll(rawData[t+1,:,:,2],1,axis=0)
            perMoveLattice[:,:,3] = np.roll(rawData[t+1,:,:,3],1,axis=1)
            perMoveLattice[:,:,4:] = np.copy(rawData[t+1,:,:,4:])
            for row in range(latticeRows):
                for col in range(latticeCols):
                    originalInput[t*latticeDims+row*latticeRows+col,0,:] = rawData[t,row,col,:]
                    for i in range(self.nNeighbors):
                        originalInput[t*latticeDims+row*latticeRows+col,i+1,:] = rawData[t,(row+self.neigh[i][0])%latticeRows,(col+self.neigh[i][1])%latticeCols,:]
                    # eliminating the transportation step
                    originalOutput[t*latticeDims+row*latticeRows+col,:] = perMoveLattice[row,col,:]

        vecX = np.zeros((dataEntries,self.nFeatures))
        vecY = np.zeros(dataEntries)
        for entry in range(dataEntries):
            for i in range(channelNumber):  
                vecY[entry] +=  2**(channelNumber-1-i)*originalOutput[entry,i]
            vecX[entry,:] =  self._hypothesis(originalInput[entry,:,:])
        return vecX,vecY

    def rebuildTimeSeries(self, original,nodeStates):
        """
            Input: 
            - original time series -> use to get number of time steps + initial lattice config
            - prob. array prediction from LogReg, format = (number possible inputs, prob node state) 
            Function: 
        """
        time, rows,cols, channels = np.shape(original)
        # Probable, that not all possible node state represented in the trainings set 
        # rebuild time series initialization
        rebuild = np.zeros((time,rows,cols,channels), dtype=np.bool_)
        rebuild[0,:,:] = np.copy(original[0,:,:,:])
        nStates = 2**self.channels
        for t in range(time-1):
            perMoveLattice = np.zeros((rows,cols,channels),dtype=np.bool_)
            
            for row in range(rows):
                for col in range(cols):
                    input = []
                    input.append(rebuild[t,row,col,:])
                    for i in range(self.nNeighbors):
                        input.append(rebuild[t,(row+self.neigh[i][0])%rows,(col+self.neigh[i][1])%cols,:])
                    probArray = self.clf.predict_proba([self._hypothesis(np.array(input))])
                    state = np.random.choice(nStates,p=probArray[0,:])                # choice one of the node state with the extracted distribution
                    perMoveLattice[row,col,:] = self._nodeStateToConfig(int(nodeStates[state]))
                        
            rebuild[t+1,:,:,0] = np.roll(perMoveLattice[:,:,0],1,axis=0)
            rebuild[t+1,:,:,1] = np.roll(perMoveLattice[:,:,1],1,axis=1)
            rebuild[t+1,:,:,2] = np.roll(perMoveLattice[:,:,2],-1,axis=0)
            rebuild[t+1,:,:,3] = np.roll(perMoveLattice[:,:,3],-1,axis=1)
            rebuild[t+1,:,:,4:] = np.copy(perMoveLattice[:,:,4:])
        return rebuild

    def timeSeriesExtraction(self, timeSeries, plt_show=True,produceAnimation=False):
        
        time, latticeLength, latticeWidth,  channels = np.shape(timeSeries)
        timeSteps = time-1
        
        vecX, vecY = self.buildInAndOutput(timeSeries)

        if(self.interactionMode):
            bestParams =  {'solver' : 'saga','penalty' :'l1','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':False}
        else:
            bestParams =  {'solver' : 'sag','penalty' :'l2','C':1, 'tol' : 1e-4, 'max_iter' : 100000, 'multi_class' : 'multinomial','fit_intercept':True}
       
        brier_score = make_scorer(brier_error,needs_proba=True,greater_is_better=False)
     
        self.clf=LogisticRegression(C=bestParams['C'], penalty=bestParams['penalty'],
            solver=bestParams['solver'] , tol=bestParams['tol'], max_iter=bestParams['max_iter'], multi_class=bestParams['multi_class'],fit_intercept=bestParams['fit_intercept'])

        timeIndex = TimeSeriesSplit(4)
        for trainIndex, testIndex  in timeIndex.split(vecX):
            x_train, x_test = vecX[trainIndex], vecX[testIndex]
            y_train, y_test = vecY[trainIndex], vecY[testIndex]

        self.clf.fit(x_train,y_train)
        crossVal = cross_val_score(self.clf, x_test,y_test,scoring=brier_score)
        print(f"classes seen by the classifier: {self.clf.classes_}")
        print("brier score: %s \pm %s \n" % (crossVal.mean(),crossVal.std()))

        print(f"found coefficients: \n{self.clf.coef_} ")
        rebuild = self.rebuildTimeSeries(timeSeries, np.unique(y_train))
        densityRebuild = np.sum(rebuild, axis=3)
        densityOriginal = np.sum(timeSeries, axis=3)
        
        if(produceAnimation):
            fig =  plt.figure()
            fig.suptitle(f"Density plot with {self.restChannels} restchannels\n{self.version} version")
            ax1 = fig.add_subplot(1,1,1)
            densityOriginal = np.sum(timeSeries, axis=3)
            matrice = ax1.matshow(np.flip(densityOriginal[0,:,:]),cmap='binary')
            ax1.set_title("original")
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(matrice, cax=cax1)
            ani1 = FuncAnimation(fig,_animation_update(densityOriginal,matrice),frames = timeSteps ,repeat=True)
            ani1.save("LogReg_2D_"+self.version+"_a="+str(self.restChannels)+"_original.gif")
            fig1 =  plt.figure()
            fig1.suptitle(f"Density plot with {self.restChannels} restchannels\n{self.version} version")
            ax2 = fig1.add_subplot(1,1,1)
            densityRebuild = np.sum(rebuild, axis=3)
            matrice2 = ax2.matshow(np.flip(densityRebuild[0,:,:]),cmap='binary')
            ax2.set_title("rebuild")
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(matrice2, cax=cax2)
            ani2 = FuncAnimation(fig1,_animation_update(densityRebuild,matrice2),frames = timeSteps ,repeat=True)
            ani2.save("LogReg_2D_"+self.version+"_a="+str(self.restChannels)+"_rebuild.gif")
        
        cmapName = "gist_heat"
        oriMap = plt.cm.get_cmap(cmapName)
        cmap = oriMap.reversed()
        fig2 =  plt.figure()
        if(self.interactionMode):
            fig2.suptitle(f"Density plot with {self.restChannels} restchannels\n{self.interaction} version")
        elif(self.delta):
            fig2.suptitle(f"Density plot with {self.restChannels} restchannels\n with delta-function approximation")
        else:
            fig2.suptitle(f"Density plot with {self.restChannels} restchannels")
        ax11 = fig2.add_subplot(2,4,1)
        ax11.pcolor(densityOriginal[0,:,:],cmap=cmap)
        ax11.set_title("original - initial config")
        ax12 = fig2.add_subplot(2,4,2,sharey=ax11)
        ax12.pcolor(densityOriginal[int(timeSteps/3),:,:],cmap=cmap)
        ax12.set_title("t = %i" % int(timeSteps/3))
        ax13 = fig2.add_subplot(2,4,3,sharey=ax11)
        ax13.pcolor(densityOriginal[int(2*timeSteps/3),:,:],cmap=cmap)
        ax13.set_title("t = %i" % int(2*timeSteps/3))
        ax14 = fig2.add_subplot(2,4,4,sharey=ax11)
        ax14.pcolor(densityOriginal[timeSteps,:,:],cmap=cmap)
        ax14.set_title("t = %i" % timeSteps)
        ax21 = fig2.add_subplot(2,4,5)
        ax21.pcolor(densityRebuild[0,:,:],cmap=cmap)
        ax21.set_title("rebuild - initial config")
        ax22 = fig2.add_subplot(2,4,6,sharey=ax21)
        ax22.pcolor(densityRebuild[int(timeSteps/3),:,:],cmap=cmap)
        ax22.set_title("t = %i" % int(timeSteps/3))
        ax23 = fig2.add_subplot(2,4,7,sharey=ax21)
        ax23.pcolor(densityRebuild[int(2*timeSteps/3),:,:],cmap=cmap)
        ax23.set_title("t = %i" % int(2*timeSteps/3))
        ax24 = fig2.add_subplot(2,4,8,sharey=ax21)
        im = ax24.pcolor(densityRebuild[timeSteps,:,:],cmap=cmap)
        ax24.set_title("t = %i" % timeSteps)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])

        plt.colorbar(im, cax=cax)
        if(plt_show):
            plt.show()


def _animation_update(timeSeries,matrice):
    def animate(frame_number):
        matrice.set_array(np.flip( timeSeries[frame_number,:,:]))
    return animate



def main():
    dens = 0.45
    x = 50
    y = 60
    iter = 100
    alignment_lgca = get_lgca(restchannels = 0, geometry='square', interaction = 'alignment',density=dens, bc='pbc', dims = (y,x), ib=False,beta=0.75)
    alignment_lgca.timeevo(timesteps= iter, record = True)
    aggregation_lgca = get_lgca(restchannels = 1, geometry='square', interaction = 'aggregation',density=dens, bc='pbc', dims = (y,x), ib=False,beta=1.5)
    aggregation_lgca.timeevo(timesteps= iter, record = True)
    
    timeSeries = [alignment_lgca.nodes_t,aggregation_lgca.nodes_t]

    plt.rcParams.update({'font.size': 23})

    for i in range(len(timeSeries)):
        time, latticeY, latticeX, channels = np.shape(timeSeries[i])
        restChannels = channels-4
        lM_interact = MLonLGCA(restChannels,_interaction='total')    
        lM_general = MLonLGCA(restChannels,_interactionMode=False)
        lM_interact.timeSeriesExtraction(timeSeries[i],plt_show=False)
        lM_general.timeSeriesExtraction(timeSeries[i],plt_show=False)
    plt.show()
   
    

if __name__=='__main__':
    main()