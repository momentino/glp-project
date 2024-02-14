import torch
from torch import nn


class ALBEFForSimilarities(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, true_actives, foil_actives, true_passives):
        """ Take the image embeddings and the attention mask """
        image_embeds = self.base_model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)

        """ Take the textual embeddings for the captions and the foils """
        true_actives.input_ids = torch.squeeze(true_actives.input_ids)
        foil_actives.input_ids = torch.squeeze(foil_actives.input_ids)
        true_passives.input_ids = torch.squeeze(true_passives.input_ids)
        
        true_actives_output = self.base_model.text_encoder.bert(true_actives.input_ids, attention_mask=true_actives.attention_mask,
                                             return_dict=True, mode='text')
        true_actives_embeds = true_actives_output.last_hidden_state

        foil_actives_output = self.base_model.text_encoder.bert(foil_actives.input_ids, attention_mask=foil_actives.attention_mask,
                                             return_dict=True, mode='text')
        foil_actives_embeds = foil_actives_output.last_hidden_state

        true_passives_output = self.base_model.text_encoder.bert(true_passives.input_ids, attention_mask=true_passives.attention_mask,
                                             return_dict=True, mode='text')
        true_passives_embeds = true_passives_output.last_hidden_state


        """ Fuse together (the chosen mode is 'fusion' here) """
        true_actives_vl_embeds = self.base_model.text_encoder.bert(encoder_embeds=true_actives_embeds,
                                            attention_mask=true_actives.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            ).last_hidden_state[:,0,:]
        foil_actives_vl_embeds = self.base_model.text_encoder.bert(encoder_embeds=foil_actives_embeds,
                                                    attention_mask=foil_actives.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True,
                                                    mode='fusion',
                                                    ).last_hidden_state[:,0,:]
        true_passives_vl_embeds = self.base_model.text_encoder.bert(encoder_embeds=true_passives_embeds,
                                            attention_mask=true_passives.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            ).last_hidden_state[:,0,:]
        
        #return the three textual embeddings and the three multimodal embeddings
        return true_actives_embeds, foil_actives_embeds, true_passives_embeds, true_actives_vl_embeds, foil_actives_vl_embeds, true_passives_vl_embeds