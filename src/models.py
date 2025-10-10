import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import yaml

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    torch_dtype: str = "float16"
    device_map: str = "auto"

class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activations = defaultdict(dict)
        
    def load_model(self):
        """Load the model and tokenizer"""
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_activation_hooks(self, layer_config: Dict[str, Any]):
        """Set up hooks to capture activations from specified layers"""
        self.activations.clear()
        self._remove_hooks()
        
        # Hook MLP layers
        if layer_config.get('mlp_layers') == 'all':
            mlp_layers = list(range(len(self.model.model.layers)))
        else:
            mlp_layers = self._parse_layer_range(layer_config.get('mlp_layers', []))
            
        for layer_idx in mlp_layers:
            self._hook_mlp_layer(layer_idx)
            
        # Hook residual layers
        if layer_config.get('residual_layers') == 'all':
            residual_layers = list(range(len(self.model.model.layers)))
        else:
            residual_layers = self._parse_layer_range(layer_config.get('residual_layers', []))
            
        for layer_idx in residual_layers:
            self._hook_residual_layer(layer_idx)
    
    def _hook_mlp_layer(self, layer_idx: int):
        """Hook individual MLP layer and its components"""
        mlp_layer = self.model.model.layers[layer_idx].mlp
        
        # Hook complete MLP output
        def make_mlp_hook(l_idx):
            def mlp_hook(module, input, output):
                self.activations[f'layer_{l_idx}']['mlp_output'] = output.detach().cpu()
            return mlp_hook
        
        self.hooks.append(mlp_layer.register_forward_hook(make_mlp_hook(layer_idx)))
        
        # Hook individual FFN components
        components = ['gate_proj', 'up_proj', 'down_proj']
        for component in components:
            if hasattr(mlp_layer, component):
                def make_ffn_hook(l_idx, comp):
                    def ffn_hook(module, input, output):
                        self.activations[f'layer_{l_idx}'][f'ffn_{comp}'] = output.detach().cpu()
                    return ffn_hook
                
                self.hooks.append(getattr(mlp_layer, component).register_forward_hook(
                    make_ffn_hook(layer_idx, component)
                ))
    
    def _hook_residual_layer(self, layer_idx: int):
        """Hook residual stream at specified layer"""
        layer_module = self.model.model.layers[layer_idx]
        
        def make_residual_hook(l_idx):
            def residual_hook(module, input, output):
                if len(input) > 0:
                    self.activations[f'layer_{l_idx}']['residual_input'] = input[0].detach().cpu()
                self.activations[f'layer_{l_idx}']['residual_output'] = output.detach().cpu()
            return residual_hook
        
        self.hooks.append(layer_module.register_forward_hook(make_residual_hook(layer_idx)))
    
    def collect_activations_with_prompt(self, prompt: str, max_new_tokens: int = 20) -> Dict[str, Any]:
        """Generate text while capturing activations"""
        self.activations.clear()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'activations': {k: dict(v) for k, v in self.activations.items()},
            'generated_text': generated_text,
            'input_ids': inputs['input_ids']
        }
    
    def _parse_layer_range(self, layer_spec) -> List[int]:
        """Parse layer specification into list of indices"""
        if isinstance(layer_spec, list):
            return layer_spec
        elif isinstance(layer_spec, str) and '-' in layer_spec:
            start, end = map(int, layer_spec.split('-'))
            return list(range(start, end + 1))
        else:
            return list(range(len(self.model.model.layers)))
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def cleanup(self):
        """Clean up resources"""
        self._remove_hooks()