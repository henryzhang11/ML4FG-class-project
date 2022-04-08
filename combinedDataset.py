#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import gzip
import torch

from clipDataset import ClipDataset
from motifDataset import MotifDataset
from rnaDataset import RNADataset
from rgDataset import RGDataset




class CombinedDataset(Dataset):
    '''
    This class combines all of the datasets
    '''
    def __init__(self,filepath,training=True,scalers=None):
        '''
        Inputs:
            - filepath: the filepath to the overall file directory holding 
            the data
        '''
        self.filepath = filepath
        self.proteinList = os.listdir(self.filepath)
        self.currentProteinIdx = 0
        if training:
            self.trainOrTest = 'training_sample_0'
        else:
            #Then we need to load the test dataset
            self.trainOrTest = 'test_sample_0'
       
        if scalers is not None:
            self.scalers = scalers
        else:
            self.scalers = dict()
            #Load up the scalers with Nones
            self.scalers['RG'] = None
            self.scalers['Clip'] = None
            self.scalers['RNA'] = None
            self.scalers['Motif'] = None
            
    def getProtein(self):
        '''
        will return the current protein data being loaded
        '''
        return self.proteinList[self.currentProteinIdx]
    def nextProtein(self):
        '''
        Will move the dataset tot he next protein
        Output:
            - will return the the protein the dataset is set to
        '''
        self.currentProteinIdx += 1
        return self.getProtein()
    def setProtein(self,idx):
        '''
        Sets the protein in the list to look at
        '''
        self.currentProteinIdx = idx
        return self.getProtein()
    def getProteinList(self):
        '''
        Returns the protein list in the exact order being used
        by the data set
        '''
        return self.proteinList
    def loadData(self,pIdx=None):
        '''
        Will load up the data in preperation for training
        Inputs:
            - pIdx: a protein index that can be sepcified
        '''
        if pIdx is not None:
            protein = self.setProtein(pIdx)
        else:
            protein = self.getProtein()
        dataFolder = os.path.join(self.filepath,protein,'5000',self.trainOrTest)
        self.clip = ClipDataset(dataFolder,protein,self.scalers['Clip'])
        self.motif = MotifDataset(dataFolder,protein,self.scalers['Motif'])
        self.RNA = RNADataset(dataFolder,protein,self.scalers['RNA'])
        self.RG = RGDataset(dataFolder,protein,self.scalers['RG'])

        self.scalers['Clip'] = self.clip.getScaler()
        self.scalers['Motif'] = self.motif.getScaler()
        self.scalers['RNA'] = self.RNA.getScaler()
        self.scalers['RG'] = self.RG.getScaler()

        

    def getScalers(self):
        '''
        Returns the scalers
        '''
        return self.scalers
    def __len__(self):
        '''
        returns the length of the dataset
        '''
        #length=np.max(len(self.clip),len(self.motif),len(self.RNA),len(self.RG))
        return len(self.RG)
    def __getitem__(self,i):
        '''
        Will get the item at index i
        '''
        #Retrieve all of the data points and store them in a dict
        data = dict()
        data['X_RG'] = self.RG[i]
        data['X_Clip'] = self.clip[i]
        data['X_RNA'] = self.RNA[i]
        data['Motif'] = self.motif[i]
        return data,self.getProtein()

def createDataset(path,training=True):
    '''
    This method will create a concatenated dataset for all of the proteins 
    provided in the overall dataset
    Inputs:
        - path: the path to the dataset directory (Not specific proteins)
        - training (boolean): true if you want to load the training dataset, 
                                false if you want to load the test dataset
    Returns:
        - a concatenated dataset
    '''
    os.chdir(path)#we want to change directory to path
    Datasets = [] #a list to hold all protein datasets
    scalers = None
    for p,protein in enumerate(os.listdir()):
        print("Reading: {}".format(protein))
        #First, we want to create the full path to the sequence file
        dataFile = os.path.join(path)
        proteinDataset = CombinedDataset(dataFile,training,scalers)
        proteinDataset.loadData(p)
        scalers = proteinDataset.getScalers()
        #Now we can load it into our dataset class
        Datasets.append(proteinDataset)
    #We can now create our contatDateset and return it
    return ConcatDataset(Datasets)

if __name__ == "__main__":
    #then we want to run our createDataset Class
    path = "/root/ML4FG/ML4FG-class-project/dataset/clip/"
    dataset = createDataset(path)
    print(len(dataset))
