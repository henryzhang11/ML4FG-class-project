import torch
import torch.nn as nn

class CNN_FC(nn.Module):
    def get_rnn_fea(self,input_dim, num_hidden = 128):
        # this method codes dense neural nets for datasets aside from sequence
        # -input_dim: length of input layer
        # -num_hidden: length of hidden layers
        model = nn.Sequential(nn.Linear(input_dim, num_hidden), 
                                    nn.ReLU(), 
                                    nn.PReLU(), 
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.5),
                                    nn.Linear(num_hidden, num_hidden),
                                    nn.ReLU(),
                                    nn.PReLU(),
                                    nn.BatchNorm1d(num_hidden),
                                    nn.Dropout(0.5))
        return model
        
    def get_cnn_network(self):
    # this method codes the cnn network for one-hot encoded RNA sequences
        nbfilter = 101 # the default cnn input length for our case since we have 101 nucleotides windows
        model = nn.Sequential(nn.Conv1d(4,4,kernel_size = 7, padding = 3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(3),
                                    nn.Dropout(0.5),
                                    nn.Flatten(),
                                    nn.Linear(nbfilter, nbfilter),
                                    nn.ReLU(),
                                    nn.Dropout(0.25))
        return model
        # not the same as original code, original code nbfilter = 102
    
    def __init__(self): 
        # this method defines CNN layer, dense layer, and the fully connected layer that follows
        rg_dim = 505 # dim[1] of matrix_regionType.tab
        rg_hid = 128
        
        clip_dim = 3030 # dim[1] of matrix_Cobinding.tab
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
        self.dense_net=nn.Sequential(nn.Dropout(0.5), nn.Linear(total_hid, 2), nn.Softmax(dim = 0))
        
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
