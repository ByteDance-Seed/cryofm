from typing import Union, List

import numpy as np
import numpy.linalg as LA
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path


from cryoseed.fft.fft_numpy import fft3_freq, np_real_to_ft

__all__ = [
    "calc_cc",
    "calc_fsc",
    "fsc_to_resolution",
    "get_fsc_map",
    "plot_fsc",
    "plot_fsc_multiple",
    "save_fsc_npz",
    "save_fsc_txt",
    "load_fsc_txt",
]

def _in_mask_ndi_sum(data, mask=None, labels=None, index=None):
    if mask is None:
        data_masked = data
    else:
        data_masked = np.where(mask, data, 0)
    return ndimage.sum(data_masked, labels=labels, index=index)


# if fourier mask is not None, will calculate fsc in mask only
def calc_fsc(vol1, vol2, fourier_mask=None, num_shells=None):
    assert vol1.shape == vol2.shape, f"Shape mis-match {vol1.shape} vs {vol2.shape}"

    nz, ny, nx = vol1.shape

    if num_shells is None:
        num_shells = nx // 2 + 1

    vol1 = np_real_to_ft(vol1)
    vol2 = np_real_to_ft(vol2)

    kz, ky, kx, radii = fft3_freq((nz, ny, nx))
    # odd shape has max freq less than 0.5
    max_radii = min(np.abs(kz).max(), np.abs(ky).max(), np.abs(kx).max())

    shell_radii = np.linspace(0, max_radii, num_shells, endpoint=True)

    labels = np.searchsorted(shell_radii, radii, side="left")
    # skip 0 freq
    index = np.arange(1, len(shell_radii))
    shell_count = len(index)

    numerator = _in_mask_ndi_sum(
        np.real(vol1 * np.conj(vol2)), mask=fourier_mask, labels=labels, index=index
    )
    denominator = np.sqrt(
        _in_mask_ndi_sum(
            np.abs(vol1) ** 2, mask=fourier_mask, labels=labels, index=index
        )
        * _in_mask_ndi_sum(
            np.abs(vol2) ** 2, mask=fourier_mask, labels=labels, index=index
        )
    )

    fsc_values = np.zeros(shell_count)
    for i in range(shell_count):
        if denominator[i] > 1e-3:
            fsc_values[i] = numerator[i] / denominator[i]

    fsc_freq = shell_radii[1:]

    return fsc_values, fsc_freq


# if all fsc is small than threshold, this function will output the resolution defined by the minimum frequency
def fsc_to_resolution(fsc_values, shell_freq=None, threshold=0.143, voxel_size=1.0):
    assert len(fsc_values) == len(shell_freq)

    below_threshold = fsc_values <= threshold
    cross_index = (
        np.where(below_threshold)[0][0]
        if np.any(below_threshold)
        else len(fsc_values) - 1
    )

    if shell_freq is None:
        return cross_index

    # default freq for cross_index = 0 or = len(fsc_values) - 1, fsc_values[cross_index] = threshold
    freq = shell_freq[cross_index]

    # linear interp
    if (
        cross_index > 0
        and fsc_values[cross_index - 1] > threshold > fsc_values[cross_index]
    ):
        frac = (threshold - fsc_values[cross_index]) / (
            fsc_values[cross_index - 1] - fsc_values[cross_index]
        )
        freq = (
            shell_freq[cross_index] * (1.0 - frac) + shell_freq[cross_index - 1] * frac
        )

    return voxel_size / freq


def plot_fsc(
    fsc_values,
    pix_freq,
    threshold: Union[List, float] = 0.143,
    voxel_size=1.0,
    save_path=None,
    fontname=None,
):
    if fontname is not None:
        plt.rcParams["font.family"] = fontname  # "Times New Roman"

    max_freq = max(pix_freq)

    # pre-defined ticks
    xtick_locs = [
        i
        for i in (0.01, 0.05, 0.0833, 0.125, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5)
        if i <= max_freq
    ]
    xtick_lbl = [
        "1/100",
        "1/20",
        "1/12",
        "1/8",
        "1/6",
        "1/5",
        "1/4",
        "1/3",
        "1/2.5",
        "1/2",
    ][: len(xtick_locs)]
    ytick_locs = (0.0, 0.125, 0.143, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    ytick_lbl = ("0", " ", "0.143", "0.25", " ", "0.5", " ", "0.75", " ", "1.0")

    if isinstance(threshold, float):
        threshold = [
            threshold,
        ]

    res_list = []
    for t in threshold:
        res = fsc_to_resolution(fsc_values, pix_freq, t, voxel_size)
        res_list.append(res)
        freq = voxel_size / res
        plt.annotate(
            r"{:1.2f} $\AA$".format(res),
            xy=(freq, t),
            xytext=(freq, t + 0.05),
            fontsize=12.0,
            # arrowprops={"width": 1, "shrink": .05}
        )
        plt.axhline(t, color="k", linestyle=":")

    plt.plot(pix_freq, fsc_values, linewidth=1)
    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("FSC")

    plt.xticks(xtick_locs, xtick_lbl)
    plt.yticks(ytick_locs, ytick_lbl)
    plt.title("FSC curves", fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
    plt.cla()
    plt.close()
    return res_list


def plot_fsc_multiple(
    fsc_list,
    pix_freq,
    labels=None,
    threshold: Union[List, float] = 0.143,
    voxel_size=1.0,
    save_path=None,
    fontname=None,
):

    if fontname is not None:
        plt.rcParams["font.family"] = fontname

    max_freq = max(pix_freq)

    xtick_locs = [
        i
        for i in (0.01, 0.05, 0.0833, 0.125, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5)
        if i <= max_freq
    ]
    xtick_lbl = [
        "1/100",
        "1/20",
        "1/12",
        "1/8",
        "1/6",
        "1/5",
        "1/4",
        "1/3",
        "1/2.5",
        "1/2",
    ][: len(xtick_locs)]
    ytick_locs = (0.0, 0.125, 0.143, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    ytick_lbl = ("0", " ", "0.143", "0.25", " ", "0.5", " ", "0.75", " ", "1.0")

    if isinstance(threshold, float):
        threshold = [threshold]

    # for t in threshold:
    #     plt.axhline(t, color='k', linestyle=":", label=f"Threshold {t}")

    for i, fsc in enumerate(fsc_list):
        label = labels[i] if labels is not None else f"Curve {i}"
        plt.plot(pix_freq, fsc, linewidth=1.2, label=label)

    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("FSC")
    plt.xticks(xtick_locs, xtick_lbl)
    plt.yticks(ytick_locs, ytick_lbl)
    plt.title("FSC Curves")
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.cla()
    plt.close()


def calc_cc(vol1, vol2, eps=1e-7):
    assert vol1.shape == vol2.shape, f"Shape mis-match {vol1.shape} vs {vol2.shape}"

    vol1 = vol1.ravel()
    vol2 = vol2.ravel()

    norm1 = LA.norm(vol1, 2)
    norm2 = LA.norm(vol2, 2)

    return np.sum(vol1 * vol2) / (norm1 * norm2 + eps)


def get_fsc_map(vol1, vol2, mask):
    """
    Adapted from spisonet.
    """
    h1 = vol1 * mask
    h2 = vol2 * mask
    f1 = np_real_to_ft(h1)
    f2 = np_real_to_ft(h2)
    ret = np.real(np.multiply(f1, np.conj(f2)))
    n1 = np.real(np.multiply(f1, np.conj(f1)))
    n2 = np.real(np.multiply(f2, np.conj(f2)))
    fsc_map = ret / np.sqrt(n1 * n2)
    return fsc_map

def save_fsc_npz(
    path,
    freq,
    fsc,
    *,
    iter_=None,
    resolution=None,
    resolution_05=None,
    fsc_gt=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "freq": np.asarray(freq, dtype=np.float32),
        "fsc": np.asarray(fsc, dtype=np.float32),
    }

    if iter_ is not None:
        data["iter"] = int(iter_)
    if resolution is not None:
        data["resolution_0143"] = float(resolution)
    if resolution_05 is not None:
        data["resolution_05"] = float(resolution_05)
    if fsc_gt is not None:
        data["fsc_gt"] = np.asarray(fsc_gt, dtype=np.float32)

    np.savez(path, **data)


def load_fsc_npz(path):
    path = Path(path)

    data = np.load(path)

    freq = data["freq"]
    fsc = data["fsc"]

    result = {
        "freq": freq,
        "fsc": fsc,
    }

    # optional fields
    if "iter" in data:
        result["iter"] = int(data["iter"])
    if "resolution_0143" in data:
        result["resolution"] = float(data["resolution_0143"])
    if "resolution_05" in data:
        result["resolution_05"] = float(data["resolution_05"])
    if "fsc_gt" in data:
        result["fsc_gt"] = data["fsc_gt"]

    return result


def save_fsc_txt(
    path,
    freq,
    fsc,
    *,
    iter_=None,
    resolution=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    freq = list(freq)
    fsc = list(fsc)

    if len(freq) != len(fsc):
        raise ValueError("freq and fsc must have the same length")

    with open(path, "w") as f:
        # ===== header =====
        f.write("# CryoSeed FSC\n")
        if iter_ is not None:
            f.write(f"# iter: {iter_}\n")
        if resolution is not None:
            f.write(f"# resolution(0.143): {resolution:.2f} Angstrom\n")

        f.write("# columns: freq fsc\n")

        # ===== data =====
        for x, y in zip(freq, fsc):
            f.write(f"{x:10.6f} {y:8.6f}\n")


def load_fsc_txt(path):
    path = Path(path)

    freq = []
    fsc = []

    meta = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # ===== header =====
            if line.startswith("#"):
                if "iter:" in line:
                    meta["iter"] = int(line.split(":")[1])
                elif "resolution" in line:
                    val = line.split(":")[1].replace("Angstrom", "").strip()
                    meta["resolution"] = float(val)
                continue

            # ===== data =====
            parts = line.split()
            if len(parts) >= 2:
                freq.append(float(parts[0]))
                fsc.append(float(parts[1]))

    return {
        "freq": np.array(freq, dtype=np.float32),
        "fsc": np.array(fsc, dtype=np.float32),
        **meta,
    }