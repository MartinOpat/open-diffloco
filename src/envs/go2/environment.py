"""Variant factory for Go2 locomotion environments."""

from importlib import import_module


DEFAULT_VARIANT = "blind_nolinvel_nokinref"
VARIANTS = {
    "blind_nolinvel_nokinref": (
        "src.envs.go2.variants.blind_nolinvel_nokinref.environment"
    ),
    "blind_linvel_nokinref": (
        "src.envs.go2.variants.blind_linvel_nokinref.environment"
    ),
    "blind_linvel_kinref": "src.envs.go2.variants.blind_linvel_kinref.environment",
    "highspeed_nokinref": "src.envs.go2.variants.highspeed_nokinref.environment",
}


def get_go2_env_class(variant: str = DEFAULT_VARIANT):
    """Return the Go2Env class for a named environment variant."""
    if variant is None:
        variant = DEFAULT_VARIANT
    try:
        module_name = VARIANTS[variant]
    except KeyError as exc:
        valid = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown Go2 variant '{variant}'. Valid variants: {valid}") from exc
    return import_module(module_name).Go2Env


def Go2Env(*args, variant: str = DEFAULT_VARIANT, **kwargs):
    """Instantiate a named Go2 environment variant."""
    env_cls = get_go2_env_class(variant)
    return env_cls(*args, **kwargs)


__all__ = ["DEFAULT_VARIANT", "VARIANTS", "Go2Env", "get_go2_env_class"]
