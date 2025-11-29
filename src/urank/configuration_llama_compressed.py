"""
Custom configuration class for LlamaCompressedForCausalLM.

This file must exist separately because HuggingFace's AutoConfig and
trust_remote_code require a standalone configuration_<model_type>.py file.
"""

from transformers import LlamaConfig


class LlamaCompressedConfig(LlamaConfig):
    """
    LLaMA configuration for compressed models with FactorizedLinear layers.
    
    Note: We keep model_type="llama" for HuggingFace internal compatibility
    (RoPE, RMSNorm, cache shapes, etc.). The architectures field determines
    which class to load.
    """

    model_type = "llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_compressed = True