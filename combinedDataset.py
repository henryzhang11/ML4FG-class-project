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
        self.trainOrTest = training
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
        protein = self.getProtein()
        dataFolder = os.path.join(self.filepath,protein,'5000',self.trainOrTest)
        self.clip = ClipDataset(dataFolder,protein,self.scalers['Clip'])
        self.motif = MotifDatasetDataset(dataFolder,protein,self.scalers['Motif'])
        self.RNA = RNADatasetDataset(dataFolder,protein,self.scalers['RNA'])
        self.RG = RGDatasetDataset(dataFolder,protein,self.scalers['RG'])

        self.scalers['Clip'] = self.clip.getScalers()
        self.scalers['Motif'] = self.motif.getScalers()
        self.scalers['RNA'] = self.RNA.getScalers()
        self.scalers['RG'] = self.RG.getScalers()

        

    def getScalers(self):
        '''
        Returns the scalers
        '''
        return self.scalers
    def __len__(self):
        '''
        returns the length of the dataset
        '''
        length=np.max(len(self.clip),len(self.motif),len(self.RNA),len(self.RG))
        return length
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

