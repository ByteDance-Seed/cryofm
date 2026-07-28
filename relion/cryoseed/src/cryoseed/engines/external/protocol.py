from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from cryoseed.engines.external.manager import ExternalReconstructJob


@dataclass(frozen=True)
class ExternalReconstructLayout:
    """Filesystem layout for one external reconstruction job.

    The ``result`` path is part of the primary handoff back to the main
    process. The external tool may overwrite it, but a clean command exit with
    the original result left untouched is still an acceptable identity-style
    outcome. ``result_star`` is optional auxiliary output.
    """

    work_dir: str
    data_real: str
    data_imag: str
    weight: str
    result: str
    result_star: str
    star: str


def build_external_reconstruct_layout(
    *,
    output_root: str,
    epoch: int,
    half_index: int,
) -> ExternalReconstructLayout:
    """Build the per-half file layout for one external reconstruction round."""
    work_dir = os.path.abspath(os.path.join(output_root, f"epoch_{int(epoch):03d}"))
    epoch_tag = os.path.basename(work_dir)
    stem = f"{epoch_tag}_half{int(half_index)}_class000_external_reconstruct"
    prefix = os.path.join(work_dir, stem)
    return ExternalReconstructLayout(
        work_dir=work_dir,
        data_real=f"{prefix}_data_real.mrc",
        data_imag=f"{prefix}_data_imag.mrc",
        weight=f"{prefix}_weight.mrc",
        result=f"{prefix}.mrc",
        result_star=f"{prefix}_out.star",
        star=f"{prefix}.star",
    )


def write_external_reconstruct_metadata(
    *,
    star_path: str,
    data_real_path: str,
    data_imag_path: str,
    weight_path: str,
    result_path: str,
    result_star_path: str,
    original_image_size: int,
    current_image_size: int,
    pixel_size: float,
    particle_diameter: float,
    prior_variance: torch.Tensor,
    fsc: torch.Tensor,
) -> None:
    """Write the STAR metadata file consumed by the external reconstruction program."""
    tau2 = torch.as_tensor(prior_variance, dtype=torch.float32).reshape(-1).cpu()
    fsc = torch.as_tensor(fsc, dtype=torch.float32).reshape(-1).cpu()
    work_dir = os.path.dirname(star_path)
    data_real_relpath = os.path.relpath(os.path.abspath(data_real_path), start=work_dir)
    data_imag_relpath = os.path.relpath(os.path.abspath(data_imag_path), start=work_dir)
    weight_relpath = os.path.relpath(os.path.abspath(weight_path), start=work_dir)
    result_relpath = os.path.relpath(os.path.abspath(result_path), start=work_dir)
    result_star_relpath = os.path.relpath(os.path.abspath(result_star_path), start=work_dir)

    with open(star_path, "w") as f:
        f.write("# version 50001\n\n")
        f.write("data_external_reconstruct_general\n\n")
        f.write(f"_rlnExtReconsDataReal {data_real_relpath}\n")
        f.write(f"_rlnExtReconsDataImag {data_imag_relpath}\n")
        f.write(f"_rlnExtReconsWeight   {weight_relpath}\n")
        f.write(f"_rlnExtReconsResult   {result_relpath}\n")
        f.write(f"_rlnExtReconsResultStarfile {result_star_relpath}\n")
        f.write("_rlnTau2FudgeFactor 1.000000\n")
        f.write("_rlnPaddingFactor 1.000000\n")
        f.write("_rlnReferenceDimensionality 3\n")
        f.write(f"_rlnOriginalImageSize {int(original_image_size)}\n")
        f.write(f"_rlnCurrentImageSize {int(current_image_size)}\n")
        f.write(f"_rlnPixelSize {float(pixel_size):.6f}\n")
        f.write(f"_rlnParticleDiameter {float(particle_diameter):.6f}\n\n")
        f.write("# version 50001\n\n")
        f.write("data_external_reconstruct_tau2\n\n")
        f.write("loop_\n")
        f.write("_rlnSpectralIndex #1\n")
        f.write("_rlnReferenceTau2 #2\n")
        f.write("_rlnGoldStandardFsc #3\n\n")
        for spectral_index, (tau2_value, fsc_value) in enumerate(zip(tau2.tolist(), fsc.tolist())):
            f.write(f"{spectral_index}  {tau2_value:.6e}  {fsc_value:.6f}\n")


def build_external_reconstruct_job(
    *,
    name: str,
    layout: ExternalReconstructLayout,
) -> ExternalReconstructJob:
    """Convert one external reconstruction request layout into a manager job.

    The manager treats this job as successful when the external command exits
    cleanly. It does not require proof that the command produced a modified
    reconstruction, because some tools may legitimately behave like an
    identity transform and leave ``layout.result`` unchanged.
    """
    return ExternalReconstructJob(
        name=name,
        work_dir=layout.work_dir,
        request_path=layout.star,
        command_extra_args=(
            "--work-dir",
            layout.work_dir,
            "--star-file",
            os.path.basename(layout.star),
        ),
    )