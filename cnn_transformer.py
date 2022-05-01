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
