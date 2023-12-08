
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# home_dir = os.getcwd()
os.chdir(os.path.expanduser("~"))

sys.path.append('DoLa') # ensures that the directory containing the script is in your Python module search path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import re

import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import argparse

import ssl
import urllib.request
import zipfile
from extraction import get_hidden
from dola import DoLa
from datasets import load_dataset
from tqdm import tqdm
NUM_SAMPLES = 600
data = load_dataset("c4", "en", split="validation", streaming=True)
list_data_dict = []
idx = 0
for item in tqdm(data) :
    idx += 1
    if idx == 10001 :
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

all_data_feats = []
all_data_labels = []
all_data_dola_logits = []

for sample in tqdm(list_data_dict[:NUM_SAMPLES]):
    len_sample = len(sample)
    sample_truncated = sample[int(0.2*len_sample) : ]
    generate_kwargs = dict(mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, \
                       candidate_premature_layers=candidate_premature_layers,post_softmax=False)
    _,premature_layers,premature_layer_feats,\
    mature_layer_feats,logits,labels,input_ids = llm.lm_score_full(sample_truncated,'',max_all_tokens=3400,**generate_kwargs)
    
    input_ids = llm.tokenizer(sample_truncated, return_tensors="pt").input_ids.cpu() #.to(llm.device)
    hidden_layer_features = torch.stack([mature_layer_feats.squeeze(),premature_layer_feats],dim=0)
    
    all_data_feats.append(hidden_layer_features)
    all_data_labels.append(labels)
    all_data_dola_logits.append(logits)

print("len feats list: ", len(all_data_feats), "    shape: ", all_data_feats[0].shape)
print("len labels list shape: ", len(all_data_labels), "    shape: ", all_data_labels[0].shape)
print("len logits list shape: ", len(all_data_dola_logits), "   shape: ", all_data_dola_logits[0].shape)

# NEED TO FIGURE OUT THE h5py WAY TO STORE A LIST OF TENSORS
torch.save(all_data_feats, f'/srv/kira-lab/share4/yali30/fall_23/cse_8803/DoLa/data/{NUM_SAMPLES}_samples/layer_features.pt')
torch.save(all_data_labels, f'/srv/kira-lab/share4/yali30/fall_23/cse_8803/DoLa/data/{NUM_SAMPLES}_samples/labels.pt')
torch.save(all_data_dola_logits, f'/srv/kira-lab/share4/yali30/fall_23/cse_8803/DoLa/data/{NUM_SAMPLES}_samples/dola_output_logits.pt')