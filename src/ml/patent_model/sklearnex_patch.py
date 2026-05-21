"""Intel Extension for Scikit-learn acceleration entrypoint."""

_PATCHED = False


def patch_sklearn_for_traditional_ml() -> None:
    """Apply sklearnex before importing sklearn estimators used by traditional ML."""

    global _PATCHED
    if _PATCHED:
        return

    try:
        from sklearnex import patch_sklearn
    except ImportError as exc:
        raise RuntimeError(
            "sklearnex acceleration is enabled for traditional ML, but sklearnex could not be imported. "
            "Install scikit-learn-intelex and make sure its oneDAL native DLLs are available on PATH."
        ) from exc

    patch_sklearn()
    _PATCHED = True
