import torch
from torch import nn
import torch.nn.functional as F


class XVLMForSimilarities(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, true_actives, foil_actives, true_passives):
        image_embeds, image_atts = self.base_model.get_vision_embeds(images)

        true_actives.input_ids = torch.squeeze(true_actives.input_ids, dim=0)
        foil_actives.input_ids = torch.squeeze(foil_actives.input_ids, dim=0)
        true_passives.input_ids = torch.squeeze(true_passives.input_ids, dim=0)

        true_actives_embeds = self.base_model.get_text_embeds(true_actives.input_ids, true_actives.attention_mask)
        foil_actives_embeds = self.base_model.get_text_embeds(foil_actives.input_ids, foil_actives.attention_mask)
        true_passives_embeds = self.base_model.get_text_embeds(true_passives.input_ids, true_passives.attention_mask)

        cross_true_actives = self.base_model.get_cross_embeds(image_embeds, image_atts, text_embeds=true_actives_embeds,
                                                              text_atts=true_actives.attention_mask)[:, 0,:]
        
        cross_foil_actives = self.base_model.get_cross_embeds(image_embeds, image_atts, text_embeds=foil_actives_embeds,
                                               text_atts=foil_actives.attention_mask)[:, 0, :]
        cross_true_passives = self.base_model.get_cross_embeds(image_embeds, image_atts, text_embeds=true_passives_embeds,
                                                              text_atts=true_passives.attention_mask)[:, 0,:]
        
        return true_actives_embeds[:,0,:], foil_actives_embeds[:,0,:], true_passives_embeds[:,0,:], cross_true_actives, cross_foil_actives, cross_true_passives
