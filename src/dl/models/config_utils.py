from __future__ import annotations


def merge_model_kwargs(defaults: dict, config: dict | None = None, kwargs: dict | None = None) -> dict:
    settings = dict(defaults)
    updates = {}
    if config is not None:
        updates.update(config)
    if kwargs is not None:
        updates.update(kwargs)
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise TypeError(f"Unknown model config keys: {unknown}")
    settings.update(updates)
    return settings
