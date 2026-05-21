import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 注意：deterministic=True 会在 configure_cudnn 中被 benchmark=True 覆盖，
    # 这里仅保证初始化阶段的确定性；训练阶段优先性能。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
