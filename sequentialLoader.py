from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import numpy as np
import gzip



class ProteinData(Dataset):
    '''
    The dataloader to load in the sequential data of a specific protein
    '''
    def __init__(self,filepath,protein):
        '''
        Initialize the data loader
        Inputs:
            - filepath: path to the sequence file for the particular protein
            - protein: the name of the protein this sequence is associated with
                        -essentially the label of the data
        '''
        self.filepath = filepath
        self.data = self.read_seq(filepath)
        self.protein = protein
   def read_seq(self, seq_file):
        '''
        THIS WAS TAKEN FROM THE iDEEP DATA LOADER SINCE
        Using their code here will ensure we are loading the data exactly as
        they did since we are using their performance as a benchmark
        '''
        seq_list = []
        seq = ''
        with gzip.open(seq_file, 'r') as fp:
            for line in fp:
                if line[0] == '>':
                    name = line[1:-1]
                    if len(seq):
                        seq_array = get_RNA_seq_concolutional_array(seq)
                        seq_list.append(seq_array)                    
                    seq = ''
                else:
                    seq = seq + line[:-1]
            if len(seq):
                seq_array = get_RNA_seq_concolutional_array(seq)
                seq_list.append(seq_array) 
        
        return np.array(seq_list)
    def get_RNA_seq_concolutional_array(self,seq,motif_len=4):
        '''
        THIS WAS TAKEN FROM THE iDEEP DATA LOADER SINCE
        Using their code here will ensure we are loading the data exactly as
        they did since we are using their performance as a benchmark
        '''
        seq = seq.replace('U', 'T')
        alpha = 'ACGT'
        #for seq in seqs:
        #for key, seq in seqs.iteritems():
        row = (len(seq) + 2*motif_len - 2)
        new_array = np.zeros((row, 4))
        for i in range(motif_len-1):
            new_array[i] = np.array([0.25]*4)
        
        for i in range(row-3, row):
            new_array[i] = np.array([0.25]*4)
            
        #pdb.set_trace()
        for i, val in enumerate(seq):
            i = i + motif_len-1
            if val not in 'ACGT':
                new_array[i] = np.array([0.25]*4)
                continue
            #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
            #    new_array[i] = np.array([0.25]*4)
            #else:
            try:
                index = alpha.index(val)
                new_array[i][index] = 1
            except:
                pdb.set_trace()
            #data[key] = new_array
        return new_array
    def __len__(self):
        '''
        REQUIRED METHOD! Will return the length of the dataset
        '''
        return len(self.data)
    def __getitem__(self,i):
        '''
        This method will return the datapoint at a specific index
        '''
        return self.data[i],self.protein





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
    seq = 'sequences.fa.gz'
    os.chdir(path)#we want to change directory to path
    proteinDatasets = [] #a list to hold all protein datasets
    for protein in os.listdir():
        #First, we want to create the full path to the sequence file
        seqFile = os.path.join(path,trainOrTest,seq)
        #Now we can load it into our dataset class
        proteinDatasets.append(ProteinData(seqFile,protein))
    #We can now create our contatDateset and return it
    return ConcatDataset(proteinDatasets)

