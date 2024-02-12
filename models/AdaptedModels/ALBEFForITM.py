from models.ALBEF.models import model_pretrain

import torch
from torch import nn
import torch.nn.functional as F


class ALBEFForITM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, captions, foils):
        """ Take the image embeddings and the attention mask """
        image_embeds = self.base_model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)

        """ Take the textual embeddings for the captions and the foils """
        captions.input_ids = torch.squeeze(captions.input_ids)
        foils.input_ids = torch.squeeze(foils.input_ids)
        captions_output = self.base_model.text_encoder.bert(captions.input_ids, attention_mask=captions.attention_mask,
                                             return_dict=True, mode='text')
        captions_embeds = captions_output.last_hidden_state

        foils_output = self.base_model.text_encoder.bert(foils.input_ids, attention_mask=foils.attention_mask,
                                                 return_dict=True, mode='text')
        foils_embeds = foils_output.last_hidden_state

        """ Fuse together (the chosen mode is 'fusion' here) """
        captions_vl_embeds = self.base_model.text_encoder.bert(encoder_embeds=captions_embeds,
                                            attention_mask=captions.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            ).last_hidden_state[:,0,:]
        foils_vl_embeds = self.base_model.text_encoder.bert(encoder_embeds=foils_embeds,
                                                    attention_mask=foils.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True,
                                                    mode='fusion',
                                                    ).last_hidden_state[:,0,:]
        captions_vl_output = self.base_model.itm_head(captions_vl_embeds)
        foils_vl_output = self.base_model.itm_head(foils_vl_embeds)

        """ Each ITM head returns the probability for the caption to match the image. 
        We only take the probability for the image to match the caption """

        prob_scores = [F.softmax(captions_vl_output),F.softmax(foils_vl_output)]
        return prob_scores
