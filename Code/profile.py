import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json, math
from typing import Tuple, Optional, Union

import train

device = torch.device('cuda:1')

with open('data/coco/COCO_train_set_not_norm.pkl', 'rb') as f:
    all_data = pickle.load(f)

print(all_data['clip_embedding'].shape, len(all_data['captions']), all_data['captions'][5], all_data['clip_embedding_text_dave'].shape)

'''
with open('others/CLIP_embeddings_centers_info.pkl', 'rb') as f:
    modality_offset = pickle.load(f)['offset_to_add_in_training']

print(modality_offset.shape)
'''

'''
with open('data/coco/verified_split_COCO_train_set_with_text_not_norm_tokens.pkl', 'rb') as f:
    all_data = pickle.load(f)

print(all_data[0])
'''