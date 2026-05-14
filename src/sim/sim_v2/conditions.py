# V2 序列条件采样薄壳：通用实现在 sim_common.conditions。

from sim_common.conditions import (
    build_synthetic_condition_rows as sequence_condition_rows,
    build_synthetic_condition_rows_four_component as sequence_condition_rows_four_component,
    normalize_measurement_components,
    sample_four_component_measurement,
)
