import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ModelConfig:
    model_name: str
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_activation_hooks(self, layers: List[str]):
        """Set up hooks for OR-Bench style layer specifications"""
        self.activations.clear()
        self._remove_hooks()
        
        for layer_spec in layers:
            if layer_spec.startswith('residuals_'):
                layer_idx = int(layer_spec.split('_')[1])
                self._hook_residual_layer(layer_idx)
            elif layer_spec.startswith('mlp_'):
                layer_idx = int(layer_spec.split('_')[1])
                self._hook_mlp_layer(layer_idx)
            elif layer_spec.startswith('attention_'):
                layer_idx = int(layer_spec.split('_')[1])
                self._hook_attention_layer(layer_idx)
    
    def _hook_residual_layer(self, layer_idx: int):
        """Hook residual stream at specified layer"""
        if layer_idx >= len(self.model.model.layers):
            return
            
        layer_module = self.model.model.layers[layer_idx]
        
        def make_residual_hook(l_idx):
            def residual_hook(module, input, output):
                if len(input) > 0:
                    self.activations[f'residuals_{l_idx}'] = input[0].detach().cpu()
            return residual_hook
        
        self.hooks.append(layer_module.register_forward_hook(make_residual_hook(layer_idx)))
    
    def _hook_mlp_layer(self, layer_idx: int):
        """Hook MLP output at specified layer"""
        if layer_idx >= len(self.model.model.layers):
            return
            
        mlp_layer = self.model.model.layers[layer_idx].mlp
        
        def make_mlp_hook(l_idx):
            def mlp_hook(module, input, output):
                self.activations[f'mlp_{l_idx}'] = output.detach().cpu()
            return mlp_hook
        
        self.hooks.append(mlp_layer.register_forward_hook(make_mlp_hook(layer_idx)))
    
    def run_with_activations(self, prompts: List[str], max_new_tokens: int = 20) -> tuple:
        """Run batch inference and return outputs with activations - OR-Bench compatible"""
        self.activations.clear()
        
        # Tokenize with padding
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                output_hidden_states=False,  # We use hooks instead
                return_dict_in_generate=True
            )
        
        # Decode generated texts
        generated_texts = []
        for i in range(len(prompts)):
            generated_seq = outputs.sequences[i][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_seq, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts, dict(self.activations)
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def cleanup(self):
        """Clean up resources"""
        self._remove_hooks()

def load_model_with_hooks(model_name: str, layers: List[str], device: str = "cuda"):
    """Factory function to load model with hooks - OR-Bench compatible"""
    config = ModelConfig(
        model_name=model_name,
        device_map=device
    )
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_model()
    model_wrapper.setup_activation_hooks(layers)
    return model_wrapper