import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_FC(nn.Module):
    def get_rnn_fea(self,input_dim, num_hidden = 20,sequenceLen=3030):
        # this method codes dense neural nets for datasets aside from sequence
        # -input_dim: length of input layer
        # -num_hidden: length of hidden layers
        model = nn.Sequential(nn.Linear(input_dim, num_hidden),  
                                    nn.ReLU(), 
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.1),
                                    nn.Linear(num_hidden, num_hidden),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.1))
        
        return model
        
    def get_cnn_network(self):
    # this method codes the cnn network for one-hot encoded RNA sequences
        nbfilter = 101 # the default cnn input length for our case since we have 101 nucleotides windows
        model = nn.Sequential(nn.Conv1d(4,4,kernel_size = 7, padding = 3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(3),
                                    nn.Dropout(0.1),
                                    nn.Flatten(),
                                    nn.Linear(140, nbfilter),
                                    nn.ReLU(),
                                    nn.Dropout(0.1))
        return model
        # not the same as original code, original code nbfilter = 102
    
    def __init__(self, number_of_clip_experiments): 
        # this method defines CNN layer, dense layer, and the fully connected layer that follows
        rg_dim = 505 # dim[1] of matrix_regionType.tab
        rg_hid = 128
        
        clip_dim = number_of_clip_experiments * 101 # dim[1] of matrix_Cobinding.tab
        clip_hid = 256
        
        rna_dim = 101 # dim[1] of matrix_RNAfold.tab
        rna_hid = 64
        
        cnn_hid = 64
        
        motif_dim = 102 # dim[1] of Motif_fea.gz
        motif_hid = 64
        
        seq_hid = 101 # dim[1] of Sequences.fa (here it disagrees with the 
        #not the same as original code, original code seq_hid=102
        
        super(CNN_FC, self).__init__()
        
        self.rg_net = self.get_rnn_fea(rg_dim, rg_hid*2)
        self.clip_net = self.get_rnn_fea(clip_dim, clip_hid*3)
        self.rna_net = self.get_rnn_fea(rna_dim, rna_hid*2)
        self.motif_net = self.get_rnn_fea(motif_dim, motif_hid*2)
        self.seq_net = self.get_cnn_network()
                
        total_hid=rg_hid*2 + clip_hid*3 + rna_hid*2 + motif_hid*2 + seq_hid # total hid is length of shared representation as mentioned in picturial summary of iDeep
        # not the same as original code, original code doesn't have "*2"s
        self.dense_net=nn.Sequential(nn.Dropout(0.1), nn.Linear(total_hid, 1), nn.Sigmoid())
        
    def forward(self, training_data):
    # this method defines the forward function used in training and testing
        #   -training_data: a dictionary containing 5 files: "X_RG" for region type, "X_CLIP" for clip cobinding data, "X_RNA" for RNA structure data, "motif" for motif data, and "seq" for sequence data
        #                   this model assumes that training_data is preprocessed and split between training and testing
        rg_net = self.rg_net(training_data["X_RG"])
        clip_net = self.clip_net(training_data["X_CLIP"])
        rna_net = self.rna_net(training_data["X_RNA"])
        motif_net = self.motif_net(training_data["motif"])
        seq_net = self.seq_net(training_data["seq"])
        net = torch.cat((rg_net, clip_net, rna_net, motif_net, seq_net),1) # tensors of a batch are concatenated along axis 1 (the only non batch sequence dimension)
        net = self.dense_net(net)
        return(net)
    
import os
import torch
import numpy as np

import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import re

import matplotlib.pyplot as plt # for plotting
import seaborn as sns # also for plotting
import torch.nn as nn

def train_model(model, dataset, epochs=100, patience=10, verbose = True):
    
    # Train a 1D CNN model and record accuracy metrics.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    #dataset.to(device)
    proteins = dataset.getProteinList()
    onehot = OneHotEncoder(handle_unknown="ignore")
    proteins = np.array(proteins).reshape(1,-1)
    onehot.fit(proteins)
    trainLen = int(0.8 * len(dataset))
    validationLen = len(dataset) - trainLen

    #we do our randomsampling
    
    train_dataset,validation_dataset = torch.utils.data.random_split(dataset,(trainLen,validationLen))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, num_workers = 0)
    
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=100)

    # Instantiates an optimizer for the model. 
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3, eps = 1e-7, amsgrad = True)

    # Run the training loop with early stopping. 
    
    patience = patience # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt' # to save the best model fit to date
    
    import timeit
    start_time = timeit.default_timer()
    torch.set_grad_enabled(True) # we'll need gradients

    for epoch in range(epochs):
        start_time = timeit.default_timer()

        #train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device,onehot=onehot)
        #val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device,onehot=onehot)

        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc, _, _= run_one_epoch(False, validation_dataloader, model, optimizer, device)
        
        if val_loss < best_val_loss: 
           torch.save(model.state_dict(), check_point_filename)
           best_val_loss = val_loss
           patience_counter = patience
        else: 
           patience_counter -= 1
           if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" % 
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))

def run_one_epoch(train_flag, dataloader, model, optimizer, device="cuda",onehot=None):
    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    accuracies = []
    true_y = np.empty(shape = (0))
    predicted_y = np.empty(shape = (0))

    lossFunc = nn.BCELoss()
    for (x,y) in dataloader: # collection of tuples with iterator

        #(x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = model(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension
        
        y = torch.Tensor(list(y))
        loss = lossFunc(output,y) # numerically stable
        #loss= nn.NLLLoss()(torch.log(output), torch.Tensor.long(y))
        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())
        if not train_flag:
            #print(y.detach().numpy())
            true_y = np.concatenate([true_y, y.detach().numpy()])
            predicted_y = np.concatenate([predicted_y, output.detach().numpy()])

    # if training, return losses and accuracies
    if train_flag:
        return( np.mean(losses), np.mean(accuracies) )

    # if testing, return training, losses, accuracies, predictions, true results, 
    else:
        return( np.mean(losses), np.mean(accuracies), true_y, predicted_y)

if __name__ == '__main__':
    dataDir = "/content/clip/"
 #full path to the directory holding all of the data 
    #call the combined dataset class
    trainingData = CombinedDataset(dataDir,training=True)# we want to load the training set
    testingData = CombinedDataset(dataDir, training=False)# here we load the testing
    
    
    for p,protein in enumerate(os.listdir(dataDir)):
        print("Training:{}".format(protein))
        number_of_clip_experiments = trainingData.loadData(p)#will load in the data for protein located at index p in the directory
        cnnfc = CNN_FC(number_of_clip_experiments)
        train_model(cnnfc, trainingData, epochs=30, patience=5, verbose = True)

        optimizer = torch.optim.Adam(cnnfc.parameters(), lr = 3e-3, eps = 1e-7)

        number_of_clip_experiments = testingData.loadData(p)
        test_dataloader = torch.utils.data.DataLoader(testingData, batch_size=100)
        _, _, true_y, predicted_y = run_one_epoch(False, test_dataloader, cnnfc, optimizer)

        print("roc_auc scores is %.4f." % metrics.roc_auc_score(true_y, predicted_y))

        for i in range(len(predicted_y)):
          predicted_y[i] = int(predicted_y[i] > 0.5)

        print("accuracy score is %.4f." % metrics.accuracy_score(true_y, predicted_y))
    
    
