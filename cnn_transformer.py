import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_transformer(nn.Module):
    def get_rnn_fea(self,input_dim, num_hidden = 20):
        # this method codes dense neural nets for datasets aside from sequence
        # -input_dim: length of input layer
        # -num_hidden: length of hidden layers
        model = nn.Sequential(nn.Linear(input_dim, num_hidden),  
                                    nn.ReLU(), 
                                    nn.PReLU(),
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.3),
                                    nn.Linear(num_hidden, num_hidden),
                                    nn.ReLU(),
                                    nn.PReLU(),
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.2))
        
        return model
        
    def get_cnn_network(self):
    # this method codes the cnn network for one-hot encoded RNA sequences
        nbfilter = 64 # the default cnn input length for our case since we have 101 nucleotides windows
        model = nn.Sequential(nn.Conv1d(4,4,kernel_size = 7, padding = 3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(3),
                                    nn.Dropout(0.3),
                                    nn.Flatten(),
                                    nn.Linear(140, nbfilter),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        return model
        # not the same as original code, original code nbfilter = 102
    
    def __init__(self, number_of_clip_experiments): 
        # this method defines CNN layer, dense layer, and the fully connected layer that follows
        rg_dim = 505 # dim[1] of matrix_regionType.tab
        
        clip_dim = number_of_clip_experiments * 101 # dim[1] of matrix_Cobinding.tab
        
        rna_dim = 101 # dim[1] of matrix_RNAfold.tab

        cnn_hid = 64
        
        motif_dim = 102 # dim[1] of Motif_fea.gz
        
        seq_hid = 101 #dim[1] of Sequences.fa (here it disagrees with the 
        #not the same as original code, original code seq_hid=102
        
        super(CNN_transformer, self).__init__()
        
        self.rg_net = self.get_rnn_fea(rg_dim, 64)
        self.clip_net = self.get_rnn_fea(clip_dim, 64)
        self.rna_net = self.get_rnn_fea(rna_dim, 64)
        self.motif_net = self.get_rnn_fea(motif_dim, 102)
        self.seq_net = self.get_cnn_network()
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4, batch_first = True)
        self.transformer_net=nn.Sequential(nn.Dropout(0.2), nn.TransformerEncoder(encoder_layer, num_layers=1)) # we are putting motif to the end of the model since it has no spatial structure
        
        self.linear = nn.Sequential(nn.Linear(358, 1), nn.Sigmoid()) # 1126 = 1024 + 102 with 1024 the result of "sequencelength\times feature number" of transformer and 102 the output of motif_net
        
    def forward(self, training_data):
    # this method defines the forward function used in training and testing
        #   -training_data: a dictionary containing 5 files: "X_RG" for region type, "X_CLIP" for clip cobinding data, "X_RNA" for RNA structure data, "motif" for motif data, and "seq" for sequence data
        #                   this model assumes that training_data is preprocessed and split between training and testing
        rg_net = torch.unsqueeze(self.rg_net(training_data["X_RG"]), 2) # the unsqueeze operation gives tensors of shape (batch size, number of features, 1) shaped tensor
        clip_net = torch.unsqueeze(self.clip_net(training_data["X_CLIP"]), 2)
        rna_net = torch.unsqueeze(self.rna_net(training_data["X_RNA"]), 2)
        seq_net = torch.unsqueeze(self.seq_net(training_data["seq"]), 2)
        net = torch.cat((rg_net, clip_net, rna_net, seq_net), 2) # we want tensor of shape (batch size, sequence length, number of features), we have (batch size, sequence length)
        net = self.transformer_net(net)
        net = net.reshape(100, 256)
        motif_net = self.motif_net(training_data["motif"])
        net = torch.cat((net, motif_net), 1)
        net = self.linear(net)
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

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, eps = 1e-7, amsgrad = True)

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
        cnntransformer = CNN_transformer(number_of_clip_experiments)
        train_model(cnntransformer, trainingData, epochs=30, patience=5, verbose = True)

        # calculate roc curve
        optimizer = torch.optim.Adam(cnntransformer.parameters(), lr = 1e-3, eps = 1e-7, amsgrad = True)
        number_of_clip_experiments = testingData.loadData(p)
        test_dataloader = torch.utils.data.DataLoader(testingData, batch_size=100)
        _, _, true_y, predicted_y = run_one_epoch(False, test_dataloader, cnntransformer, optimizer)

        # fpr, tpr, threshold = metrics.roc_curve(true_y, predicted_y)

        print("roc_auc scores is %.4f." % metrics.roc_auc_score(true_y, predicted_y))

        for i in range(len(predicted_y)):
          predicted_y[i] = int(predicted_y[i] > 0.5)

        print("accuracy score is %.4f." % metrics.accuracy_score(true_y, predicted_y))
