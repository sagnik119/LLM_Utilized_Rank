"""
Custom configuration class for GPT2CompressedLMHeadModel.

This file must exist separately because HuggingFace's AutoConfig and
trust_remote_code require a standalone configuration_<model_type>.py file.
"""

from transformers import GPT2Config


class GPT2CompressedConfig(GPT2Config):
    """
    GPT-2 configuration for compressed models with FactorizedLinear layers.
    """

    model_type = "gpt2_compressed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)