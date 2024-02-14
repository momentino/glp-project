import torch
from torch import nn
import torch.nn.functional as F


class BLIPForITM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, true_actives, foil_actives, true_passives):
        image_embeds = self.base_model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        ta_encoder_input_ids = true_actives.input_ids.clone()
        #c_encoder_input_ids[:, 0] = self.base_model.tokenizer.enc_token_id

        fa_encoder_input_ids = foil_actives.input_ids.clone()
        #f_encoder_input_ids[:, 0] = self.base_model.tokenizer.enc_token_id

        tp_encoder_input_ids = true_passives.input_ids.clone()
        #f_encoder_input_ids[:, 0] = self.base_model.tokenizer.enc_token_id

        ta_encoder_input_ids = torch.squeeze(ta_encoder_input_ids)
        fa_encoder_input_ids = torch.squeeze(fa_encoder_input_ids)
        tp_encoder_input_ids = torch.squeeze(tp_encoder_input_ids)

        ta_output_pos = self.base_model.text_encoder(ta_encoder_input_ids,
                                       attention_mask=true_actives.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )

        fa_output_pos = self.base_model.text_encoder(fa_encoder_input_ids,
                                                attention_mask=foil_actives.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                )
        
        tp_output_pos = self.base_model.text_encoder(tp_encoder_input_ids,
                                       attention_mask=true_passives.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )

        ta_vl_embeddings = ta_output_pos.last_hidden_state[:, 0, :]
        fa_vl_embeddings = fa_output_pos.last_hidden_state[:, 0, :]
        tp_vl_embeddings = tp_output_pos.last_hidden_state[:, 0, :]
        return ta_vl_embeddings,  fa_vl_embeddings, tp_vl_embeddings
