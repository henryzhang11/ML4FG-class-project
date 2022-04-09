#!/usr/bin/env python3
import os
from combinedDataset import CombinedDataset
import torch
from cnn_fc_model import CNN_FC
from cnn_lstm_model import CNN_LSTM
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import re
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

    # 2. Instantiates an optimizer for the model. 
    # TODO CODE

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    # 3. Run the training loop with early stopping. 
    # TODO CODE
    train_accs = []
    val_accs = []
    patience = 10 # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt' # to save the best model fit to date
    
    import timeit
    start_time = timeit.default_timer()
    torch.set_grad_enabled(True) # we'll need gradients

    for epoch in range(20):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device,onehot=onehot)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device,onehot=onehot)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
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

    for (x,y) in dataloader: # collection of tuples with iterator

        #(x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = model(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension
        
        y = torch.Tensor(list(y))
        
        loss = F.binary_cross_entropy_with_logits(output,y) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())  

    return( np.mean(losses), np.mean(accuracies) )

    train_accs = []
    val_accs = []
    patience = 10 # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt' # to save the best model fit to date
    
    import timeit
    start_time = timeit.default_timer()
    torch.set_grad_enabled(True) # we'll need gradients

    for epoch in range(20):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
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

    # 4. Return the fitted model (not strictly necessary since this happens "in place"), train and validation accuracies.
    return model, train_accs, val_accs # TODO CODE (make sure you use train_accs, val_accs in former parts of this code)


if __name__ == '__main__':
    dataDir = "/root/ML4FG/ML4FG-class-project/dataset/clip/"
 #full path to the directory holding all of the data 
    #call the combined dataset class
    trainingData = CombinedDataset(dataDir,training=True)#we want to load the training set
    for p,protein in enumerate(os.listdir(dataDir)):
        print("Training:{}".format(protein))
        trainingData.loadData(p)#will load in the data for protein located at index p in the directory
        cnnfc = CNN_FC()
        train_model(cnnfc, trainingData, epochs=100, patience=10, verbose = True)
    
