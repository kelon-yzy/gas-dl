from __future__ import annotations

from torch import nn

from models.cnn1d import CNN1DRegressor
from models.cnn_lstm import CNNLSTMRegressor
from models.branch_fusion import BranchFusionRegressor
from models.multimodal_fusion_v3 import MultimodalFusionV3Regressor
from models.gru import GRURegressor
from models.lstm import LSTMRegressor
from models.tcn import TCNRegressor
from models.transformer_encoder import TransformerRegressor
from models.multimodal_wrapper import (
    build_cnn1d_multimodal,
    build_gru_multimodal,
    build_lstm_multimodal,
    build_tcn_multimodal,
    build_transformer_multimodal,
    build_cnn_lstm_multimodal,
)

MODEL_REGISTRY = {
    # 纯慢变量
    "cnn1d": CNN1DRegressor,
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "tcn": TCNRegressor,
    "cnn_lstm": CNNLSTMRegressor,
    "transformer": TransformerRegressor,
    "branch_fusion": BranchFusionRegressor,
    "multimodal_fusion_v3": MultimodalFusionV3Regressor,
    # 多模态变体（MultimodalWrapper + 各纯慢变量 backbone）
    "cnn1d_multimodal": build_cnn1d_multimodal,
    "gru_multimodal": build_gru_multimodal,
    "lstm_multimodal": build_lstm_multimodal,
    "tcn_multimodal": build_tcn_multimodal,
    "transformer_multimodal": build_transformer_multimodal,
    "cnn_lstm_multimodal": build_cnn_lstm_multimodal,
}


def build_model(config: dict) -> nn.Module:
    model_config = dict(config)
    name = model_config.pop("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**model_config)
