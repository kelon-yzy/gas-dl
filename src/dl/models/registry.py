from __future__ import annotations

from torch import nn

from models.cnn1d import CNN1DRegressor
from models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor
from models.cnn1d_tcn_fusion_slow_branch import CNN1DTCNSlowBranchRegressor
from models.cnn_lstm import CNNLSTMRegressor
from models.branch_fusion import BranchFusionRegressor
from models.multimodal_fusion_v3 import MultimodalFusionV3Regressor
from models.early_fusion_film import EarlyFusionFiLMRegressor
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
    # 1DCNN 声学编码 + TCN 时序融合（专用多模态结构，独立于 MultimodalWrapper）
    "cnn1d_tcn_fusion": CNN1DTCNFusionRegressor,
    # 慢变量分支隔离实验：慢变量 MLP 编码 + target-specific heads
    "cnn1d_tcn_fusion_slow_branch": CNN1DTCNSlowBranchRegressor,
    # FiLM 调制型早期融合（E2，docs/早期融合_Early_Fusion_完整实验方案.md §7）
    "early_fusion_film": EarlyFusionFiLMRegressor,
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
