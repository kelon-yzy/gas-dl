# V2 train split z-score scaler 拟合：调用 sim_common.fit_z_score_scalers。

from sim_common.scalers import fit_z_score_scalers

from .constants import MODAL_GROUPS, SEQUENCE_CHANNELS


def fit_sequence_scalers(x_matrix, train_indexes):
    """V2 12 通道版 z-score 拟合，输出 sequence_scaler + modal_scaler。"""
    return fit_z_score_scalers(
        x_matrix,
        train_indexes,
        channel_names=SEQUENCE_CHANNELS,
        modal_groups=MODAL_GROUPS,
        transform_target="X",
        channel_axis=2,
    )
