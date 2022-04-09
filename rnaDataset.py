#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import gzip
import torch



class RNADataset(Dataset):
    '''
    The dataset for the RNA data which takes a specific protein and loads
    in the data
    '''
    def __init__(self,filepath,protein,scaler=None):
        '''
        Initialize the data set
        Inputs:
            - filepath: path to the RNA file for the particular protein
            - protein: the name of the protein this file is associated with
                - essentially the label of the data
            - scaler variable used
        '''
        self.filepath = filepath
        self.filename =  "matrix_RNAfold.tab.gz"
        self.data = np.loadtxt(gzip.open(os.path.join(self.filepath,self.filename)),skiprows=1)
        self.data,self.scaler= self.preprocess_data(self.data,scaler=scaler)
        #convert the np array to tensor
        self.data = torch.from_numpy(self.data)
        self.protein = protein
    def getScaler(self):
        '''
        Get the scaler
        '''
        return self.scaler
    def preprocess_data(self,X, scaler=None, stand = False):
        '''
        THIS WAS TAKEN FROM THER iDEEP PROJECT
        method will scale the data
        '''
        if not scaler:
            if stand:
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler    


    def __getitem__(self,i):
        '''
        method will get the item at index i
        '''
        return self.data[i]
    def __len__(self):
        '''
        Will get the length of the dataset
        '''
        return len(self.data)

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
    if training:
        trainOrTest = 'training_sample_0'
    else:
        #Then we need to load the test dataset
        trainOrTest = 'test_sample_0'
    os.chdir(path)#we want to change directory to path
    Datasets = [] #a list to hold all protein datasets
    for protein in os.listdir():
        print("Reading: {}".format(protein))
        #First, we want to create the full path to the sequence file
        dataFile = os.path.join(path,protein,'5000',trainOrTest)
        #Now we can load it into our dataset class
        Datasets.append(RNADataset(dataFile,protein))
    #We can now create our contatDateset and return it
    return ConcatDataset(Datasets)

if __name__ == "__main__":
    #then we want to run our createDataset Class
    path = "/root/ML4FG/ML4FG-class-project/dataset/clip/"
    dataset = createDataset(path)
    print(len(dataset))
