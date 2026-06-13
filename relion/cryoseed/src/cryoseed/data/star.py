from __future__ import annotations

import os

import pandas as pd
import starfile

__all__ = [
    "read_starfile",
    "parse_optics_parameters",
    "merge_optics_to_particles",
    "parse_stack_entries",
    "save_starfile",
]


def read_starfile(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a RELION-style STAR file.

    Supports STAR files with separate ``particles``/``optics`` tables (RELION 3.1+)
    and the single-table legacy format.

    Args:
        path (str): Path to the STAR file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(df_particles, df_optics)``.
    """
    df = starfile.read(path)
    try:
        df_particles = df["particles"]
        df_optics = df["optics"]
    except KeyError:
        # Fallback for single-table STAR files
        df_particles = df
        df_optics = df.iloc[[0]].copy()

    return df_particles, df_optics


def parse_optics_parameters(
    df_optics: pd.DataFrame,
) -> tuple[float | None, int | None]:
    """Parse pixel size and image size from the optics table.

    Returns ``(angpix, image_size)``. Either value can be ``None`` if the
    corresponding column is missing.

    Raises:
        ValueError: If pixel size or image size varies across optics groups.
    """
    angpix: float | None = None
    image_size: int | None = None

    if "rlnImagePixelSize" in df_optics.columns:
        pixel_sizes = df_optics["rlnImagePixelSize"].dropna().unique()
        if len(pixel_sizes) > 1:
            raise ValueError(
                f"Inconsistent pixel sizes found in optics groups: {pixel_sizes}"
            )
        if len(pixel_sizes) == 1:
            angpix = float(pixel_sizes[0])

    if "rlnImageSize" in df_optics.columns:
        image_sizes = df_optics["rlnImageSize"].dropna().unique()
        if len(image_sizes) > 1:
            raise ValueError(
                f"Inconsistent image sizes found in optics groups: {image_sizes}"
            )
        if len(image_sizes) == 1:
            image_size = int(image_sizes[0])

    return angpix, image_size


def merge_optics_to_particles(
    df_particles: pd.DataFrame,
    df_optics: pd.DataFrame,
) -> pd.DataFrame:
    """Merge optics parameters into the particles table.

    Args:
        df_particles (pd.DataFrame): Particles table.
        df_optics (pd.DataFrame): Optics table.

    Returns:
        pd.DataFrame: A particles table with optics columns merged in (when possible).
    """
    if (
        "rlnOpticsGroup" in df_particles.columns
        and "rlnOpticsGroup" in df_optics.columns
    ):
        # Ensure types match for merging
        df_particles = df_particles.copy()
        df_optics = df_optics.copy()

        df_particles["rlnOpticsGroup"] = df_particles["rlnOpticsGroup"].astype(int)
        df_optics["rlnOpticsGroup"] = df_optics["rlnOpticsGroup"].astype(int)

        # Merge
        # We use suffixes just in case, but typically optics params are unique to optics table in Relion 3.1+
        df_merged = df_particles.merge(
            df_optics, on="rlnOpticsGroup", how="left", suffixes=("", "_optics")
        )
        return df_merged
    else:
        # If no link, assume single group or old format where params might be in particles
        # In case of old format (single table), df_particles already has everything.
        return df_particles


def parse_stack_entries(
    df_particles: pd.DataFrame,
    data_prefix: str = "",
) -> tuple[pd.Series, pd.Series]:
    """Parse per-particle stack indices and stack paths.

    The STAR column ``rlnImageName`` uses the format ``"{index}@{path}"``.

    Args:
        df_particles (pd.DataFrame): Particles table.
        data_prefix (str, optional): Prefix prepended to relative paths.
            Defaults to ``""``.

    Returns:
        tuple[pd.Series, pd.Series]: ``(stack_index, stack_paths)`` where
        ``stack_index`` is 0-based.
    """
    split = df_particles["rlnImageName"].str.split("@", expand=True)
    stack_index = split[0].astype(int) - 1

    if data_prefix:
        stack_paths = split[1].apply(lambda x: os.path.join(data_prefix, x))
    else:
        stack_paths = split[1]

    return stack_index, stack_paths


def save_starfile(file_path: str, data: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a STAR file.

    Args:
        file_path (str): Output path.
        data (pd.DataFrame): DataFrame to write.
    """
    starfile.write(data, file_path, overwrite=True)