# Working with Cryo-EM Software

This guide compiles practical tips and insights we've accumulated from working with cryo-EM software during our research. It covers essential software tools commonly used in our work, including ChimeraX, RELION, EMAN2, and CryoSPARC.

---

## ChimeraX

ChimeraX provides many useful commands for reference. For a comprehensive collection of recipes and examples, visit [ChimeraX Recipes](https://rbvi.github.io/chimerax-recipes/).

#### Flipping a Volume

To mirror-flip a volume in ChimeraX:

```
vol flip #1
```

#### Displaying Coordinate Axes

To visualize coordinate axes, create an `XYZ-axes.bild` file with the following content and open it in ChimeraX:

```text
.comment -- This file shows X,Y,Z axes as red, yellow, blue arrows --
.comment -- Edit "translate" and "scale" values to adjust offset and size --
.translate 0.0 0.0 0.0
.scale 5
.sphere 0 0 0 0.5
.color 1 0 0
.arrow 0 0 0 96 0 0 0.7
.color 1 1 0
.arrow 0 0 0 0 96 0 0.7
.color 0 0 1
.arrow 0 0 0 0 0 96 0.7
```

#### Viewing Multiple Models or Maps

- **Viewing atomic models**: Use the `mseries` command. See the [official documentation](https://www.cgl.ucsf.edu/chimerax/docs/user/commands/mseries.html) for details.
- **Viewing density maps (volumes)**: Use the `vseries` command. See the [official documentation](https://www.cgl.ucsf.edu/chimerax/docs/user/commands/vseries.html) for details.

---

## EMAN2

#### Calculating and Plotting FSC Curves

EMAN2 provides tools for calculating FSC (Fourier Shell Correlation) curves:

```bash
# Activate the EMAN2 conda environment
e2proc3d.py a.mrc output.txt --calcfsc=b.mrc
```

To visualize the results, use the separate `plotfsc.py` script. The official example script is available at [plotfsc.py](https://github.com/cryoem/eman2/blob/master/examples/plotfsc.py). Run:

```bash
python plotfsc.py output.txt
```

---

## RELION

#### Low-pass Filtering a Volume

To apply a low-pass filter to a volume:

```bash
relion_image_handler --angpix xxx --lowpass 20 --i input.mrc --o output.mrc
```

#### Down-sampling (Binning) a Volume

To down-sample (bin) a volume:

```bash
relion_image_handler --angpix xxx --rescale_angpix xxx --new_box xxx --i input.mrc --o output.mrc
```

#### Simulating the Projection Process

To simulate projections, you need a STAR file containing pose information:

```bash
relion_project --i input.mrc --o rln_sim --ctf --angpix xxx --ang example.star

# Add noise (0.1 is high enough for most cases)
relion_project --i input.mrc --o rln_sim --ctf --angpix xxx --add_noise --white_noise 0.01 --ang test.star
```

#### Reconstruction

To perform reconstruction from particle STAR files:

```bash
relion_reconstruct --i test.star --o relion.mrc --ctf --sym C1 --angpix xxx --mask_diameter xxx --fsc

# Using MPI for parallel processing
mpirun -n 20 `which relion_reconstruct_mpi` --i test.star --o relion.mrc --ctf --sym C1 --angpix xxx --mask_diameter xxx --fsc
```
