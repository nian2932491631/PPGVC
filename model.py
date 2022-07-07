import torch
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder_Taco2AR
from .layers import BNEncoder, PostNet
from .bn_extractor import ctc_ASR
from .ECAPA_TDNN_layers import ECAPA_TDNN as SpeakerEncoder

class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        self.bn_extractor = ctc_ASR(hparams)
        self.eval_mode = hparams.eval_mode
        
        self.speaker_encoder = SpeakerEncoder()

        self.bn_encoder = BNEncoder(hparams)

        self.decoder = Decoder_Taco2AR(hparams)
        
        self.postnet = PostNet(hparams)

    def grouped_parameters(self,):

        params_group = [p for p in self.bn_encoder.parameters()]
        params_group.extend([p for p in self.speaker_encoder.parameters()])
        params_group.extend([p for p in self.decoder.parameters()])
        params_group.extend([p for p in self.postnet.parameters()])

        return params_group

    def parse_batch(self, batch):
        mel_padded, mel_lengths = batch
        mel_padded = to_gpu(mel_padded).float()
        mel_lengths = to_gpu(mel_lengths).long()

        return ((mel_padded, mel_lengths), (mel_padded,mel_lengths))


    def forward(self, mel, mel_lengths):
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
        if self.eval_mode:
            self.bn_extractor.eval()
        
        speaker_embedding = self.speaker_encoder(mel) 
        
        bn,bn_lengths = self.bn_extractor(mel,mel_lengths)
        bn = self.bn_encoder(bn.detach(),bn_lengths)
        
        L = bn.size(1)
        
        bn = torch.cat([bn, speaker_embedding.unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel = self.decoder(bn, mel, bn_lengths * 2)
        post_output = self.postnet(predicted_mel)
        
        outputs = [predicted_mel, post_output]

        return outputs

    
    def inference(self, mel,mel_lengths,mel_ref):
        '''
        decode the audio sequence from input
        inputs x
        input_text True or False
        mel_reference [1, mel_bins, T]
        '''
        
        # -> [B, n_speakers], [B, speaker_embedding_dim] 
    
    

        speaker_embedding = self.speaker_encoder(mel_ref) 

        bn,_ = self.bn_extractor(mel,mel_lengths)
        bn = self.bn_encoder.inference(bn)
        
        L = bn.size(1)
        bn = torch.cat([bn, speaker_embedding.unsqueeze(1).expand(-1, L, -1)], -1)
        predicted_mel = self.decoder.inference(bn)
        
        post_output = self.postnet(predicted_mel)

        return post_output