import os
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import re

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


def train_model(model, dataset, epochs=100, patience=10, verbose = True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proteins = dataset.getProteinList()
    onehot = OneHotEncoder(handle_unknown="ignore")
    proteins = np.array(proteins).reshape(1,-1)
    onehot.fit(proteins)
    trainLen = int(0.8 * len(dataset))
    validationLen = len(dataset) - trainLen
    
    train_dataset,validation_dataset = torch.utils.data.random_split(dataset,(trainLen,validationLen))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, num_workers = 0)
    
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=100)

    optimizer = torch.optim.RMSprop(model.parameters(), lr = 3e-4, eps = 1e-7, centered = False)

    # optimizer = RMSprop and learning rate = 3e-4

    patience = patience 
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt' 
    
    import timeit
    start_time = timeit.default_timer()
    torch.set_grad_enabled(True)

    for epoch in range(epochs):
        start_time = timeit.default_timer()

        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc, _, _= run_one_epoch(False, validation_dataloader, model, optimizer, device)
        
        if val_loss < best_val_loss: 
          torch.save(model.state_dict(), check_point_filename)
          best_val_loss = val_loss
          patience_counter = patience
        else: 
          patience_counter -= 1
          if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename))
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
    for (x,y) in dataloader: 

        output = model(x) 
        output = output.squeeze()
        
        y = torch.Tensor(list(y))
        loss = lossFunc(output,y)

        if train_flag: 
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())
        if not train_flag:
            true_y = np.concatenate([true_y, y.detach().numpy()])
            predicted_y = np.concatenate([predicted_y, output.detach().numpy()])

    if train_flag:
        return( np.mean(losses), np.mean(accuracies) )

    else:
        return( np.mean(losses), np.mean(accuracies), true_y, predicted_y)

lstm_hidden_dimension=2
transformer_sequence_length=64

class CNN_LSTM_transformer(nn.Module):
    def get_rnn_fea(self,input_dim, num_hidden = 20):
        model = nn.Sequential(nn.Linear(input_dim, num_hidden),  
                                    nn.ReLU(), 
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.4))
        # Dropout rate = 0.4
        
        return model
        
    def get_cnn_network(self):
        nbfilter = transformer_sequence_length
        # transformer sequence length
        model = nn.Sequential(nn.Conv1d(4,4,kernel_size = 7, padding = 3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(3),
                                    nn.Dropout(0.4),
        # Dropout rate = 0.4
                                    nn.Flatten(),
                                    nn.Linear(140, nbfilter),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        # Dropout rate = 0.2
        return model
    
    def __init__(self, number_of_clip_experiments): 

        super(CNN_LSTM_transformer, self).__init__()
        
        self.rg_net = self.get_rnn_fea(505, transformer_sequence_length)
        # transformer sequence length
        self.clip_net = self.get_rnn_fea(number_of_clip_experiments * 101, transformer_sequence_length)
        # transformer sequence length
        self.rna_net = self.get_rnn_fea(101, transformer_sequence_length)
        # transformer sequence length
        self.motif_net = self.get_rnn_fea(102, 102)
        self.seq_net = self.get_cnn_network()
        
        self.lstm_net = nn.LSTM(1, lstm_hidden_dimension, 1, bidirectional = True)
        # LSTM hidden dimension

        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dimension * 8, nhead=4, batch_first = True)
        # LSTM hidden dimension
        self.transformer_net=nn.Sequential(nn.Dropout(0.2), nn.TransformerEncoder(encoder_layer, num_layers=1)) 
        # Dropout rate = 0.2
        self.linear = nn.Sequential(nn.Linear(transformer_sequence_length * lstm_hidden_dimension * 8 + 102, 1), nn.Sigmoid())
        # LSTM hidden dimension and transformer sequence length

    def forward(self, training_data):
        rg_net = torch.unsqueeze(self.rg_net(training_data["X_RG"]), 2)
        rg_net, _ = self.lstm_net(rg_net)

        clip_net = torch.unsqueeze(self.clip_net(training_data["X_CLIP"]), 2)
        clip_net, _ = self.lstm_net(clip_net)

        rna_net = torch.unsqueeze(self.rna_net(training_data["X_RNA"]), 2)
        rna_net, _ = self.lstm_net(rna_net)

        seq_net = torch.unsqueeze(self.seq_net(training_data["seq"]), 2)
        seq_net, _ = self.lstm_net(seq_net)

        net = torch.cat((rg_net, clip_net, rna_net, seq_net), 2) 
        net = self.transformer_net(net)
        net = net.reshape(100, transformer_sequence_length * lstm_hidden_dimension * 8)
        # LSTM hidden dimension and transformer sequence length

        motif_net = self.motif_net(training_data["motif"])
        net = torch.cat((net, motif_net), 1)
        net = self.linear(net)
        return(net)

if __name__ == '__main__':
    dataDir = "/content/clip/"
    trainingData = CombinedDataset(dataDir,training=True)
    testingData = CombinedDataset(dataDir, training=False)
    
    for p,protein in enumerate(os.listdir(dataDir)):
        print("Training:{}".format(protein))
        number_of_clip_experiments = trainingData.loadData(p)
        cnntransformer = CNN_LSTM_transformer(number_of_clip_experiments)
        train_model(cnntransformer, trainingData, epochs=30, patience=5, verbose = True)

        optimizer = torch.optim.RMSprop(cnntransformer.parameters(), lr = 3e-4, eps = 1e-7, centered = False)
        # optimizer = RMSprop and learning rate = 3e-4

        number_of_clip_experiments = testingData.loadData(p)
        test_dataloader = torch.utils.data.DataLoader(testingData, batch_size=100)
        _, _, true_y, predicted_y = run_one_epoch(False, test_dataloader, cnntransformer, optimizer)

        print("roc_auc scores is %.4f." % metrics.roc_auc_score(true_y, predicted_y))

        for i in range(len(predicted_y)):
          predicted_y[i] = int(predicted_y[i] > 0.5)

        print("accuracy score is %.4f." % metrics.accuracy_score(true_y, predicted_y))
