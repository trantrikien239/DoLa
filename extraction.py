# Utility functions for extracting hidden states from a model
import torch

def get_hidden(input_ids, llm, layers=[32]):
    """Get hidden states from a layer of the model"""
    with torch.no_grad():
        dict_outputs, outputs = llm.model(input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
            early_exit_layers=layers
            )
        dict_logits = dict_outputs
        dict_hiddens = dict()
        for layer in layers:
            dict_hiddens[layer] = outputs['hidden_states'][layer]
        
    return dict_logits, dict_hiddens