"""统一的数据容器定义。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatentDataset:
    """承载三模态特征、环境变量、标签和样本元数据。"""

    sample_ids: np.ndarray
    acoustic: np.ndarray
    optical: np.ndarray
    thermal: np.ndarray
    environment: np.ndarray
    targets: np.ndarray
    component_names: tuple[str, ...]
    metadata: pd.DataFrame
    acoustic_columns: tuple[str, ...] = field(default_factory=tuple)
    optical_columns: tuple[str, ...] = field(default_factory=tuple)
    thermal_columns: tuple[str, ...] = field(default_factory=tuple)
    environment_columns: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, object] = field(default_factory=dict)
    filter_report: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.targets.shape[0])

    def target_for(self, component_index: int) -> np.ndarray:
        """按组分索引取出一列目标值。"""

        return self.targets[:, component_index]

    def subset(self, indices: np.ndarray | list[int]) -> "PatentDataset":
        """按索引切出一个独立子数据集，避免原数据被联动修改。"""

        idx = np.asarray(indices)
        return PatentDataset(
            sample_ids=self.sample_ids[idx].copy(),
            acoustic=self.acoustic[idx].copy(),
            optical=self.optical[idx].copy(),
            thermal=self.thermal[idx].copy(),
            environment=self.environment[idx].copy(),
            targets=self.targets[idx].copy(),
            component_names=self.component_names,
            metadata=self.metadata.iloc[idx].reset_index(drop=True).copy(),
            acoustic_columns=self.acoustic_columns,
            optical_columns=self.optical_columns,
            thermal_columns=self.thermal_columns,
            environment_columns=self.environment_columns,
            provenance=dict(self.provenance),
            filter_report=dict(self.filter_report),
        )

    def with_fault_labels(self, fault_labels: pd.DataFrame) -> "PatentDataset":
        """返回带 fault_case / fault_severity 列的新 dataset，不修改原对象。"""

        required = {"fault_case", "fault_severity"}
        if not required.issubset(fault_labels.columns):
            missing = required - set(fault_labels.columns)
            raise ValueError(f"fault_labels missing required columns: {sorted(missing)}")
        new_metadata = self.metadata.copy()
        new_metadata[["fault_case", "fault_severity"]] = fault_labels[["fault_case", "fault_severity"]].to_numpy()
        return replace(self, metadata=new_metadata)
