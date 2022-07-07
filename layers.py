import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .basic_layers import sort_batch, ConvNorm, LinearNorm, Attention, tile
from .utils import get_mask_from_lengths

class SpeakerClassifier(nn.Module):
    '''
    - n layer CNN + PROJECTION
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        
        convolutions = []
        for i in range(hparams.SC_n_convolutions):
            #parse dim
            if i == 0:
                in_dim = hparams.bn_encoder_hidden_dim
                out_dim = hparams.SC_hidden_dim
            elif i == (hparams.SC_n_convolutions-1):
                in_dim = hparams.SC_hidden_dim
                out_dim = hparams.SC_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(in_dim,
                         out_dim,
                         kernel_size=hparams.SC_kernel_size, stride=1,
                         padding=int((hparams.SC_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='leaky_relu',
                         param=0.2),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.projection = LinearNorm(hparams.SC_hidden_dim, hparams.n_speakers)

    def forward(self, x,input_lengths):
        # x [B, T, dim]

        # -> [B, DIM, T]
        hidden = x.transpose(1, 2)
        for conv in self.convolutions:
            hidden = conv(hidden)
        
        # -> [B, T, dim]
        hidden = hidden.transpose(1, 2)
        logits = self.projection(hidden)

        return logits


class SpeakerEncoder(nn.Module):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling

    '''
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(hparams.n_mel_channels, int(hparams.speaker_encoder_hidden_dim / 2), 
                            num_layers=2, batch_first=True,  bidirectional=True, dropout=hparams.speaker_encoder_dropout)
        self.projection1 = LinearNorm(hparams.speaker_encoder_hidden_dim, 
                                      hparams.speaker_embedding_dim, 
                                      w_init_gain='tanh')
        self.projection2 = LinearNorm(hparams.speaker_embedding_dim, hparams.n_speakers) 
    
    def forward(self, x, input_lengths):
        '''
         x  [batch_size, mel_bins, T]

         return 
         logits [batch_size, n_speakers]
         embeddings [batch_size, embedding_dim]
        '''
        x = x.transpose(1,2)  #->[batch_size,T,mel_bins]
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
 
        outputs = torch.sum(outputs,dim=1) / sorted_lengths.unsqueeze(1).float() # mean pooling -> [batch_size, dim]

        outputs = F.tanh(self.projection1(outputs))
        outputs = outputs[initial_index]
        # L2 normalizing #
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        return logits, embeddings
    
    def inference(self, x): 
        
        x = x.transpose(1,2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs = torch.sum(outputs,dim=1) / float(outputs.size(1)) # mean pooling -> [batch_size, dim]
        outputs = F.tanh(self.projection1(outputs))
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        pid = torch.argmax(logits, dim=1)

        return pid, embeddings 

'''class BNEncoder(nn.Module):
   
    def __init__(self, hparams):
        super(BNEncoder, self).__init__()
        self.lstm1 = nn.LSTM(hparams.bn_dim, int(hparams.bn_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hparams.bn_encoder_hidden_dim*hparams.n_frames_per_step_encoder,
                             int(hparams.bn_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)

        self.concat_hidden_dim = hparams.bn_encoder_hidden_dim*hparams.n_frames_per_step_encoder
        self.n_frames_per_step = hparams.n_frames_per_step_encoder
    
    def forward(self, x, input_lengths):
        x = x.transpose(1, 2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x_packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=x.size(1)) # use total_length make sure the recovered sequence length not changed

        outputs = outputs.reshape(x.size(0), -1, self.concat_hidden_dim)

        output_lengths = torch.ceil(sorted_lengths.float() / self.n_frames_per_step).long()
        
        outputs = nn.utils.rnn.pack_padded_sequence(
            outputs, output_lengths.cpu().numpy() , batch_first=True)

        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs[initial_index], output_lengths[initial_index]
    
    def inference(self, x):
        
        x = x.transpose(1, 2)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x)
        outputs = outputs.reshape(1, -1, self.concat_hidden_dim)
        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        return outputs
'''

class BNEncoder(nn.Module):
    '''
    - Simple 2 layer bidirectional LSTM

    '''
    def __init__(self, hparams):
        super(BNEncoder, self).__init__()
        self.plstm = nn.ModuleList()
        self.plstm.append(
            nn.LSTM(hparams.bottleneck_dim, int(hparams.bn_encoder_hidden_dim / 2), 
                        num_layers=1, batch_first=True, bidirectional=True)
            )
        for i in range(1,hparams.bn_encoder_layer):
            self.plstm.append(
                nn.LSTM(hparams.bn_encoder_hidden_dim, int(hparams.bn_encoder_hidden_dim / 2), 
                        num_layers=1, batch_first=True, bidirectional=True)
            )
        self.out_projection = LinearNorm(hparams.bn_encoder_hidden_dim, hparams.bn_encoder_hidden_dim * 2)
        self.bn_encoder_hidden_dim = hparams.bn_encoder_hidden_dim
    
    def forward(self, x, input_lengths):
        '''
        x  [batch_size, mel_bins, T]

        return [batch_size, T, channels]
        '''
        total_length=x.size(1)
        
        x, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x, sorted_lengths.cpu().numpy(), batch_first=True)
        for i in range(len(self.plstm)):
            self.plstm[i].flatten_parameters()
            x,_ = self.plstm[i](x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, total_length=total_length) # use total_length make sure the recovered sequence length not changed
        
        x = self.out_projection(x)
        x = x.reshape(x.size(0), -1, self.bn_encoder_hidden_dim)

        return x[initial_index]
    
    def inference(self, x):

        for i in range(len(self.plstm)):
            self.plstm[i].flatten_parameters()
            x,_ = self.plstm[i](x)
        
        x = self.out_projection(x)
        x = x.reshape(x.size(0), -1, self.bn_encoder_hidden_dim)

        return x


class PostNet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_dim,
                             hparams.postnet_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_dim))
            )

        out_dim = hparams.n_mel_channels
        
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_dim, out_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(out_dim))
            )
        self.dropout = hparams.postnet_dropout
    
    def forward(self, input):
        # input [B, mel_bins, T]

        x = input
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), self.dropout, self.training)

        o = x + input

        return o
