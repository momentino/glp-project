import torch
from torch import nn
import torch.nn.functional as F


class BLIPForITM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model


    def forward(self, images, captions, foils):
        image_embeds = self.base_model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        c_encoder_input_ids = captions.input_ids.clone()
        #c_encoder_input_ids[:, 0] = self.base_model.tokenizer.enc_token_id

        f_encoder_input_ids = foils.input_ids.clone()
        #f_encoder_input_ids[:, 0] = self.base_model.tokenizer.enc_token_id

        c_encoder_input_ids = torch.squeeze(c_encoder_input_ids, dim=0)
        f_encoder_input_ids = torch.squeeze(f_encoder_input_ids, dim=0)

        captions_output_pos = self.base_model.text_encoder(c_encoder_input_ids,
                                       attention_mask=captions.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )

        foils_output_pos = self.base_model.text_encoder(f_encoder_input_ids,
                                                attention_mask=foils.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                )

        captions_vl_embeddings = captions_output_pos.last_hidden_state[:, 0, :]
        captions_vl_output = self.base_model.itm_head(captions_vl_embeddings)

        foils_vl_embeddings = foils_output_pos.last_hidden_state[:, 0, :]
        foils_vl_output = self.base_model.itm_head(foils_vl_embeddings)

        prob_scores = [F.softmax(captions_vl_output), F.softmax(foils_vl_output)]
        return prob_scores
