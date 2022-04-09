#!/usr/bin/env python3
import os
from combinedDataset import CombinedDataset
import torch
from cnn_fc_model import CNN_FC
from cnn_lstm_model import CNN_LSTM
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # also for plotting

def train_model(model, dataset, epochs=100, patience=10, verbose = True):
    
    # Train a 1D CNN model and record accuracy metrics.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    #dataset.to(device)
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
    train_tps = []
    train_tns = []
    train_fps = []
    train_fns = []
    train_accs = []
    
    val_tps = []
    val_tns = []
    val_fps = []
    val_fns = []
    train_accs = []
    
    patience = 10 # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt' # to save the best model fit to date
    
    import timeit
    start_time = timeit.default_timer()
    torch.set_grad_enabled(True) # we'll need gradients

    for epoch in range(20):
        start_time = timeit.default_timer()
        train_loss, train_acc, train_tp, train_tn, train_fp, train_fn = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc, val_tp, val_tn, val_fp, val_fn = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        
        train_tps.append(train_tp)
        train_tns.append(train_tn)
        train_fps.append(train_fp)
        train_fns.append(train_fn)
        train_accs.append(train_acc)
        
        val_tps.append(val_tp)
        val_tns.append(val_tn)
        val_fps.append(val_fp)
        val_fns.append(val_fn)
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

      return train_tps, train_tns, train_fps, train_fns, val_tps, val_tns, val_fps, val_fns  # TODO CODE (make sure you use train_accs, val_accs in former parts of this code)
    

def run_one_epoch(train_flag, dataloader, model, optimizer, device="cuda"):
    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    accuracies = []
    tps = []
    tns = []
    fps = []
    fns = []

    for (x,y) in dataloader: # collection of tuples with iterator

        #(x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = model(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())
        tp = torch.mean( ( (output > .5) && ( y > .5) ).float() )
        tn = torch.mean( ( (output < .5) && ( y < .5) ).float() )
        fp = torch.mean( ( (output > .5) && ( y < .5) ).float() )
        fn = torch.mean( ( (output < .5) && ( y > .5) ).float() )
        tps.append(tp.detach().cpu().numpy())
        tns.append(tn.detach().cpu().numpy())
        fps.append(fp.detach().cpu().numpy())
        fns.append(fp.detach().cpu().numpy())

    return( np.mean(losses), np.mean(accuracies), np.mean(tps), np.mean(tns), np.mean(fps), np.mean(fns) )  


if __name__ == '__main__':
    dataDir = "/root/ML4FG/ML4FG-class-project/dataset/clip/"
 #full path to the directory holding all of the data 
    #call the combined dataset class
    trainingData = CombinedDataset(dataDir,training=True)#we want to load the training set
    for p,protein in enumerate(os.listdir(dataDir)):
        print("Training:{}".format(protein))
        trainingData.loadData(p)#will load in the data for protein located at index p in the directory
        cnnfc = CNN_FC()
        train_tp, train_tn, train_fp, train_fn, val_tp, val_tn, val_fp, val_fn = train_model(cnnfc, trainingData, epochs=100, patience=10, verbose = True)

        val_acc = []
        for (item1, item2) in zip(val_tp, val_tn):
            train_acc.append(item1+item2)
        
        val_precision = []
        for (item1, item2) in zip(val_tp, val_fp):
            train_precision.append(item1/(item1+item2))
            
        val_recall = []
        for (item1, item2) in zip(val_tp, val_fn):
            train_recall.append(item1/(item1+item2))
        
        val_f1 = []
        for (item1, item2) in zip(val_precision, val_recall):
            train_precision.append(2*item1*item2/(item1+item2))
            
        plt.plot(val_f1) # plots the relationship between lambda and error calculated on validation set
        plt.xlabel("F1") # label the x axis
        plt.savefig('{}_f1.png'.format(protein))
        
        plt.plot(val_fp, val_tp)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig('{}_ROC.png'.format(protein))
        
        sns.set() # nice default plot formatting
        

        # add code here to create F_1 score charts and ROC curves for each protein
