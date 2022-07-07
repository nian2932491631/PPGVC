import torch
from torch import nn
from .utils import get_mask_from_lengths
class ParrotLoss(nn.Module):
    def __init__(self):
        super(ParrotLoss, self).__init__()
        self.L1Loss = nn.L1Loss(reduction='none')
    
    def forward(self, model_outputs, targets, eps=1e-5):

        '''
        predicted_mel [batch_size, mel_bins, T]
        predicted_gate [batch_size, T/r]
        alignment  [batch_size, T/r, T/r]
        speaker_logit_from_mel [B, n_speakers]
        
        '''
        predicted_mel, post_output = model_outputs

        mel_target, mel_lengths  = targets

        ## get masks ##
        mel_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, mel_target.size(1), -1).float()
        
        recon_loss = torch.sum(self.L1Loss(predicted_mel, mel_target) * mel_mask) / torch.sum(mel_mask)
        recon_loss_post = (self.L1Loss(post_output, mel_target) * mel_mask).sum() / torch.sum(mel_mask)

        loss_list = [recon_loss, recon_loss_post]

        return loss_list, recon_loss + recon_loss_post

