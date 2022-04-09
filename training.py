#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
import gzip
import torch



class TrainModel:
    '''
    This method will train and test our model
    '''

