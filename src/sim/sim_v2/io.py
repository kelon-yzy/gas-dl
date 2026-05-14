# V2 数据包路径布局；通用 IO（write_csv/write_json/build_index_rows/build_label_rows）已迁移到 sim_common。

from pathlib import Path

from sim_common.io import (
    build_index_rows as sequence_index_rows,
    build_label_rows as label_rows,
    write_csv,
    write_json,
)


def build_output_paths(output_dir):
    """构造所有输出文件的路径映射表。"""
    output_dir = Path(output_dir)
    return {
        "sequence_index": output_dir / "sequence_index.csv",
        "condition_grid_sequence": output_dir / "condition_grid_sequence.csv",
        "modal_sequence_long": output_dir / "sequences" / "modal_sequence_long.csv",
        "acoustic_derived_sequence_long": output_dir / "sequences" / "acoustic_derived_sequence_long.csv",
        "modal_sequence_npz": output_dir / "sequences" / "modal_sequence.npz",
        "sequence_labels": output_dir / "labels" / "sequence_labels.csv",
        "train_split": output_dir / "splits" / "train_sequence_ids.csv",
        "val_split": output_dir / "splits" / "val_sequence_ids.csv",
        "test_split": output_dir / "splits" / "test_sequence_ids.csv",
        "scaler_sequence": output_dir / "scalers" / "scaler_sequence.json",
        "scaler_sequence_modal": output_dir / "scalers" / "scaler_sequence_modal.json",
        "quality_summary": output_dir / "quality" / "sequence_quality_summary.json",
        "readme": output_dir / "README.md",
    }
