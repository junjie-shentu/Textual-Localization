from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from PIL import Image
import math

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0] # h = batch_size * num_heads
                #attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                #self.forward(attn[h // 2:], is_cross, place_in_unet)
                self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer = self.cur_att_layer + 1#self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers: # this is executed between two inference steps
            self.cur_att_layer = 0
            self.cur_step = self.cur_step + 1#self.cur_step += 1
            self.between_steps()
        #return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, LOW_RESOURCE=False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        self.low_resource = LOW_RESOURCE
        self.training_cross_attention_layers = []


    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
            # print("self.step_store:", self.step_store)
            # print(f"self.step_store_{key}_len:", len(self.step_store[key]))
        #return attn


    def between_steps(self):
        self.cur_step_attention = self.get_empty_store()
        for key in self.step_store:
            for i in range(len(self.step_store[key])):
                #self.cur_step_attention = self.step_store
                clone = self.step_store[key][i].clone()
                self.cur_step_attention[key].append(clone)

        if len(self.attention_store["down_cross"]) == 0:#len(self.attention_store) == 0:
            #self.attention_store = self.step_store
            for key in self.step_store:
                for i in range(len(self.step_store[key])):
                    clone = self.step_store[key][i].clone()
                    self.attention_store[key].append(clone)

        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    #self.attention_store[key][i] += self.step_store[key][i]
                    # This is additionally added to make self.attention_store support gradient backpropagation, but we still need to verify that the gradient would propagate through slicing
                    clone = self.step_store[key][i].clone()
                    self.attention_store[key][i] = self.attention_store[key][i] + clone

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    
    def get_current_attention(self):
        return self.cur_step_attention

    def get_training_attention_layers(self):
        return self.training_cross_attention_layers


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        #self.attention_store = {}
        #self.cur_step_attention = {}
        self.attention_store = self.get_empty_store()
        self.cur_step_attention = self.get_empty_store()

    def __init__(self, LOW_RESOURCE=False):
        super(AttentionStore, self).__init__(LOW_RESOURCE = LOW_RESOURCE)
        self.step_store = self.get_empty_store()
        #self.attention_store = {}
        #self.cur_step_attention = {}
        self.attention_store = self.get_empty_store()
        self.cur_step_attention = self.get_empty_store()


def aggregate_current_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_current_attention()

    # print("attention_maps:", attention_maps)

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out#.cpu()