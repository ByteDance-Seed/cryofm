from importlib import resources

import pandas as pd


def _load_csv(name: str) -> pd.DataFrame:
    with resources.files(__name__).joinpath(name).open("rb") as f:
        return pd.read_csv(f)


def load_fsc_curves() -> pd.DataFrame:
    """
    Load typical FSC curves used for synthetic/demo experiments.

    Returns
    -------
    pandas.DataFrame
        Table of resolution vs FSC values for several example curves.
    """
    return _load_csv("fsc_curves.csv")


__all__ = ["load_fsc_curves"]
