import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

from epinet import Epinet
import jax
from jax import numpy as jnp
import functools
import haiku as hk
import dill
import yaml

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27, trust_remote_code=False):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.trust_remote_code = trust_remote_code

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, trust_remote_code=self.trust_remote_code,
            **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'dola-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                    mature_layer=mature_layer, premature_layer=premature_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                premature_layer_dist = outputs.premature_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized.cpu(), descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized.cpu(), dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized.cpu() < probs_thresh

    def lm_score(self, input_text1, input_text2="", pmi=False, max_new_tokens=256, max_all_tokens=None,
                top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, 
                candidate_premature_layers=[], mode='baseline', verbose=True, 
                remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, 
                post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            if max_all_tokens is not None:
                input_ids = input_ids[:,:max_all_tokens]
                prefix_ids = prefix_ids[:,:max_all_tokens]
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'dola-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dola':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,) -> same premature layer for a token idx across the entire batch
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer) # this is the list of selected premature layer for each token idx (across the batch)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)

    def lm_score_full(self, input_text1, input_text2, pmi=False, 
            max_new_tokens=256, max_all_tokens=None, top_p=0.95, top_k=0, 
            temperature=0.8, mature_layer=None, premature_layer=None, 
            candidate_premature_layers=[], mode='baseline', verbose=True,
            remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, 
            post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            # prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            if max_all_tokens is not None:
                input_ids = input_ids[:,:max_all_tokens]
                # prefix_ids = prefix_ids[:,:max_all_tokens]
            continue_ids = input_ids[0, :]
            

            if mode == 'dola':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range( input_ids.shape[-1]):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,) -> same premature layer for a token idx across the entire batch
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer) # this is the list of selected premature layer for each token idx (across the batch)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, :])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0,  i]
                final_logits = dict_outputs[mature_layer][0, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask.cpu(), relative_top_value, diff_logits.cpu())
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids.cpu()].sum().item()

                # extract the features here itself - both mature/premature
                mature_layer_feat = outputs['hidden_states'][mature_layer].cpu()
                dict_hidden_list = []
                for idx,layer in enumerate(premature_layers):
                    dict_hidden_list.append(outputs['hidden_states'][layer][:,idx].cpu())
        return log_probs,(premature_layers if mode == 'dola' else None),torch.cat(dict_hidden_list,0),mature_layer_feat, diff_logits

    def lm_score_full_epinet(self, input_text1, input_text2="", pmi=False, 
        max_new_tokens=256, max_all_tokens=None,top_p=0.95, top_k=0, 
        temperature=0.8, mature_layer=None, premature_layer=None, 
        candidate_premature_layers=[], mode='dola', verbose=True, 
        remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, 
        post_softmax=True, **kwargs):
        # Force mode to be dola
        mode = 'dola'
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            if max_all_tokens is not None:
                input_ids = input_ids[:,:max_all_tokens]
                prefix_ids = prefix_ids[:,:max_all_tokens]
            prefix_len = prefix_ids.shape[-1]
            continue_ids = input_ids[0, prefix_len:]
        
        _, premature_layers_indices, hidden_activations, mature_layer_feat, diff_logits = self.lm_score_full(
            input_text1, input_text2, pmi, max_new_tokens, max_all_tokens, 
            top_p, top_k, temperature, mature_layer, premature_layer, 
            candidate_premature_layers, mode, verbose, remove_stop_words, 
            relative_top, relative_top_value, post_softmax=False, **kwargs)
        
        # import pdb; pdb.set_trace()

        mature_layer_feat = mature_layer_feat[0]
        premature_layer_feat = hidden_activations

        
        # Pass the features to the epinet to get the new token distribution, but only for 
        # the answer's tokens - after the prefix
        diff_logits_after_logsoftmax = self.forward_epinet(
            mature_layer_feat[prefix_len:,:], 
            premature_layer_feat[prefix_len:,:], 
            diff_logits[prefix_len:,:])
        
        log_probs = diff_logits_after_logsoftmax[
            range(diff_logits_after_logsoftmax.shape[0]), continue_ids
        ].sum().item()
        
        return log_probs, None 
    
    # (premature_layer_dist if mode == 'dola' else None), (premature_layers if mode == 'dola' else None)


class DoLaWithEpinet(DoLa):
    def __init__(self, 
                 model_name, 
                 device, 
                 num_gpus, 
                 max_gpu_memory=27, 
                 trust_remote_code=False):
        super().__init__(model_name, device, num_gpus, max_gpu_memory, trust_remote_code)

        with open('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/dola_inference_bridge.yaml', 'rb') as f:
            file_locations = yaml.safe_load(f)
            config_path = file_locations['config_path']
            checkpoint_path = file_locations['checkpoint_path']

        with open(config_path, 'rb') as f:
            self.epinet_config = yaml.safe_load(f)

        self.epinet = Epinet(output_size=self.epinet_config['enn_output_size'],
                             feature_size=4096,
                             num_classes=self.epinet_config['num_classes'],
                             index_dim=self.epinet_config['index_dim'],
                             epinet_hiddens=self.epinet_config['epinet_hiddens'],
                             pretrained_params_file=checkpoint_path
                             )
        self.rng = hk.PRNGSequence(self.epinet_config['seed'])

    def forward_epinet(self, mature_layer_feat, premature_layer_feat, diff_logits):
        # pass the features to the epi net
        gpu_device = jax.devices()[0]
        inputs = jax.device_put(torch.cat((premature_layer_feat, mature_layer_feat), dim=1).cpu().numpy(), gpu_device)
        dola_logits = jax.device_put(diff_logits.cpu().numpy(), gpu_device)

        net_out, _ = self.epinet.apply(inputs, self.rng)
        enn_logits = net_out.preds

        enn_logits = jnp.zeros_like(enn_logits)

        final_out =  torch.tensor(jax.device_get(enn_logits + dola_logits)).to('cuda')

        # import pdb
        # pdb.set_trace()

        return final_out