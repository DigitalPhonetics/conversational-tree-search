from enum import Enum
import torch.nn as nn
from chatbot.adviser.app.rl.layers.attention.additive_attention import AdditiveAttention

from chatbot.adviser.app.rl.layers.attention.bilinear_attention import BilinearAttention
from chatbot.adviser.app.rl.layers.attention.cosine_attention import CosineAttention
from chatbot.adviser.app.rl.layers.attention.dot_product_attention import DotProductAttention
from chatbot.adviser.app.rl.layers.attention.linear_attention import LinearAttention
from chatbot.adviser.app.rl.layers.attention.scaled_dot_product_attention import ScaledDotProductAttention


class AttentionActivationConfig(Enum):
    NONE = 'none'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    RELU = 'relu'
    SELU = 'selu'


class AttentionMechanismConfig(Enum):
    NONE = 'none'
    SCALEDDOT = 'scaleddot'
    BILINEAR = 'bilinear'
    COSINE = 'cosine'
    DOT = 'dot'
    LINEAR = 'linear'
    ADDITIVE = 'additive'



class AttentionVectorAggregation(Enum):
    SUM = "sum"
    MEAN = 'mean'
    CONCATENATE = 'concatenate'
    MAX = 'max'



def get_attention_module_instance(attention_mechanism: AttentionActivationConfig, normalize: bool = True, vector_dim: int = 512,
        matrix_dim: int = 512, activation = AttentionActivationConfig) -> nn.Module:
    """

    If activation_mechanism, will return torch.nn.Identity instance,
    otherwise an instance of an Attention module.

    Args:
        attention_mechanism (str): 'none', 'bilinear', 'cosine', 'dot', 'linear', 'scaleddot'
        activation (str): 'none', 'tanh', 'sigmoid', 'relu', 'selu'
    """
    if activation == AttentionActivationConfig.NONE:
        activation_fn = nn.Identity()
    elif activation == AttentionActivationConfig.TANH:
        activation_fn = nn.Tanh()
    elif activation_fn == AttentionActivationConfig.SIGMOID:
        activation_fn == nn.Sigmoid()
    elif activation_fn == AttentionActivationConfig.RELU:
        activation_fn = nn.ReLU()
    elif activation_fn == AttentionActivationConfig.SELU:
        activation_fn == nn.SELU()
    else:
        raise f"Activation function name unkonw : {activation}"

    if attention_mechanism == AttentionMechanismConfig.NONE:
        return nn.Identity()
    elif attention_mechanism == AttentionMechanismConfig.BILINEAR:
        return BilinearAttention(vector_dim=vector_dim, matrix_dim=matrix_dim, activation=activation_fn, normalize=normalize)
    elif attention_mechanism == AttentionMechanismConfig.COSINE:
        return CosineAttention(normalize=normalize)
    elif attention_mechanism == AttentionMechanismConfig.DOT:
        return DotProductAttention(normalize=normalize)
    elif attention_mechanism == AttentionMechanismConfig.LINEAR:
        return LinearAttention(tensor_1_dim=vector_dim, tensor_2_dim=matrix_dim, activation=activation_fn, normalize=normalize)
    elif attention_mechanism == AttentionMechanismConfig.SCALEDDOT:
        return ScaledDotProductAttention(scaling_factor=vector_dim, normalize=True)
    elif attention_mechanism == AttentionMechanismConfig.ADDITIVE:
        return AdditiveAttention(vector_dim=vector_dim, matrix_dim=matrix_dim, normalize=True)
    else:
        raise f"Attention mechanism unknown: {attention_mechanism}"
    