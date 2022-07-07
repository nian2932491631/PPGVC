import math
import torch
import torch.nn as nn
import editdistance as ed
import torch.nn.functional as F
from collections import OrderedDict
from .basic_layers import sort_batch
import math

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
                    bidirectional=True, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        if bidirectional:
            self.rnn = rnn_type(input_size=input_size, hidden_size=int(hidden_size / 2),num_layers=1, batch_first=True,bidirectional=True)
        else:
            self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,num_layers=1, batch_first=True,bidirectional=False)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x,input_lengths):
        if self.batch_norm is not None:
            x = x.transpose(-1, -2)
            x = self.batch_norm(x)
            x = x.transpose(-1, -2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(x_packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=x.size(1))

        outputs = self.dropout(outputs)
        #self.rnn.flatten_parameters()
        return outputs[initial_index],input_lengths

class LayerCNN(nn.Module):
    """
    One CNN layer include conv2d, batchnorm, activation and maxpooling
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True, dropout=0.1):
        super(LayerCNN, self).__init__()
        if len(kernel_size) == 2:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        else:
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.batch_norm = nn.BatchNorm1d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None and len(kernel_size) == 2:
            self.pooling = nn.MaxPool2d(pooling_size)
        elif len(kernel_size) == 1:
            self.pooling = nn.MaxPool1d(pooling_size)
        else:
            self.pooling = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.dropout(x)
        return x


class ctc_ASR(nn.Module):
    def __init__(self, hparams):
        super(ctc_ASR, self).__init__()

        self.add_cnn = hparams.add_cnn
        rnn_input_size = hparams.n_mel_channels
        if self.add_cnn:
            cnns = []
            self.kernel_size = hparams.kernel_size
            self.stride = hparams.stride
            self.padding = hparams.padding
            self.cnn_layer = hparams.cnn_layer
            #print(hparams.stride[1][0])
            for n in range(hparams.cnn_layer):
                in_channel = hparams.channel[n][0]
                out_channel = hparams.channel[n][1]
                kernel_size = hparams.kernel_size[n]
                stride = hparams.stride[n]
                padding = hparams.padding[n]
                #print()

                cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, hparams.pooling, batch_norm=hparams.cnn_batchnorm, dropout=hparams.cnn_dropout)
                cnns.append(('%d' % n, cnn))
                try:
                    rnn_input_size = int(math.floor((rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)
                    #print(rnn_input_size)
                except:
                    #if using 1-d Conv
                    rnn_input_size = rnn_input_size
            rnn_input_size *= out_channel
            self.cnns = nn.Sequential(OrderedDict(cnns))

        #rnns = []
        self.rnns = nn.ModuleList()
        #print(rnn_input_size)
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hparams.rnn_hidden_size,bidirectional=hparams.bidirectional,dropout=hparams.rnn_dropout,batch_norm=False)
        #i = 0
        self.rnns.append(rnn)
        for i in range(1,hparams.rnn_layer):
            rnn = BatchRNN(input_size=hparams.rnn_hidden_size, hidden_size=hparams.rnn_hidden_size, bidirectional=hparams.bidirectional,dropout=hparams.rnn_dropout, batch_norm=hparams.rnn_batchnorm)
            self.rnns.append(rnn)
        self.bottleneck_layer = nn.Linear(hparams.rnn_hidden_size, hparams.bottleneck_dim, bias=True)
        #self.rnns = nn.Sequential(OrderedDict(rnns))
        
        #self.fc = nn.Linear(hparams.rnn_hidden_size, hparams.n_symbols, bias=True)
        #self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, mel_padded,  mel_lengths):
        '''
        bn_padded [batch_size,bn_dim,max_mel_len]
        mel_padded [batch_size, mel_bins, max_mel_len]
        mel_lengths [batch_size]

        return 
        predicted_mel [batch_size, mel_bins, T]
        predicted_gate [batch_size, T/r]
        alignment [batch_size, T/r, T/r]
        spearker_logit_from_mel [B, n_speakers]

        '''
        #print(mel_lengths[0].item())
        x = mel_padded.transpose(1,2)
        if self.add_cnn:
            x = self.cnns(x.unsqueeze(1))  # [B, cnn_hidden_dim, T/r, mel_dim/r] 
            x = x.transpose(1,2).contiguous()
            x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3))  #[B, T/r, dim]

            for i in range(self.cnn_layer):
                stride = self.stride[i]
                padding = self.padding[i]
                kernel_size = self.kernel_size[i]
                mel_lengths = torch.floor((mel_lengths.float() + 2 * padding[0] - kernel_size[0]) / stride[0] + 1 ).long()
        #print(mel_lengths[0].item())
        for i in range(len(self.rnns)):
            x,mel_lengths = self.rnns[i](x,mel_lengths)
        x = self.bottleneck_layer(x)
        #print(mel_lengths[0].item())
        #x = self.fc(x)
        #x = self.log_softmax(x)



        '''speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths) 
        bn_hidden,bn_hidden_lengths = self.bn_encoder(bn_padded,mel_lengths)     # [B,T/r,bn_encoder_hidden_dim]
        
        L = bn_hidden.size(1)
        hidden = torch.cat([bn_hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_gate, alignments = self.decoder(hidden, mel_padded, bn_hidden_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_gate, alignments, speaker_logit_from_mel, mel_lengths]'''

        return x,mel_lengths