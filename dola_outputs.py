import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request
import zipfile
from extraction import get_hidden
from dola import DoLa
from datasets import load_dataset
from tqdm import tqdm

data = load_dataset("c4", "en", split="validation", streaming=True)
list_data_dict = []
idx = 0
for item in tqdm(data) :
  idx += 1
  if idx == 101 :
    break
  list_data_dict.append(item['text'])


MODEL_NAME = 'TheBloke/Llama-2-7B-fp16' # LLama-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_GPU = 1
MAX_GPU_MEMORY = 40 # GB

early_exit_layers = [i for i in range(2,34,2)]

llm = DoLa(model_name=MODEL_NAME, device=DEVICE,
           num_gpus=N_GPU, max_gpu_memory=MAX_GPU_MEMORY,
           trust_remote_code=True
           ) # Load model

mode = "dola"
mature_layer = early_exit_layers[-1]
premature_layer = None
candidate_premature_layers = early_exit_layers[:-1]
premature_layer_dist = {l:0 for l in candidate_premature_layers}

# del llm
# import gc
# gc.collect()
# torch.cuda.empty_cache()

all_data_feats = []
all_data_labels = []
all_data_dola_logits = []

for sample in tqdm(list_data_dict):
  generate_kwargs = dict(mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, \
                       candidate_premature_layers=candidate_premature_layers)
  answer_true_log_prob,premature_layers,premature_layer_feats,\
  mature_layer_feats,logits = llm.lm_score_full(sample,'',**generate_kwargs)
  input_ids = llm.tokenizer(sample, return_tensors="pt").input_ids #.to(llm.device)
  labels = input_ids[:, 1:].contiguous()
  hidden_layer_features = torch.stack([mature_layer_feats.squeeze(),premature_layer_feats],dim=0)
  all_data_feats.append(hidden_layer_features)
  all_data_labels.append(labels)
  all_data_dola_logits.append(logits)
  