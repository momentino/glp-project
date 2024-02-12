from models.ALBEF.models import model_pretrain

import torch
from torch import nn
import torch.nn.functional as F


class XVLMForITM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, captions, foils):
        image_embeds, image_atts = self.base_model.get_vision_embeds(images)
        captions.input_ids = torch.squeeze(captions.input_ids)
        foils.input_ids = torch.squeeze(foils.input_ids)
        caption_embeds = self.base_model.get_text_embeds(captions.input_ids, captions.attention_mask)
        foil_embeds = self.base_model.get_text_embeds(foils.input_ids, foils.attention_mask)

        cross_captions = self.base_model.get_cross_embeds(image_embeds, image_atts, text_embeds=caption_embeds, text_atts=captions.attention_mask)[:, 0,
                    :]
        cross_foils = self.base_model.get_cross_embeds(image_embeds, image_atts, text_embeds=foil_embeds,
                                               text_atts=foils.attention_mask)[:, 0,
                         :]
        captions_vl_output = self.base_model.itm_head(cross_captions)
        foils_vl_output = self.base_model.itm_head(cross_foils)
        """ Each ITM head returns the probability for the caption to match the image.
        We only take the probability for the image to match the caption """

        prob_scores = [F.softmax(captions_vl_output)[:,1],F.softmax(foils_vl_output)[:,1]]
        return prob_scores
