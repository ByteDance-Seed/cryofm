# Installation

We strongly recommend **using a dedicated environment** to install and use CryoFM. This helps avoid dependency conflicts and makes environment management much easier. We suggest using [conda](https://docs.conda.io/) to create and manage your environments.

---

## 1. Recommended: Install in a Clean Conda Environment

Whether you are a regular user or a developer, **always create a new, separate environment** for CryoFM. Here is the recommended installation process:

```bash
# Create a new conda environment for CryoFM (recommended)
conda create -n cryofm python=3.9 -y
conda activate cryofm

# Install CryoFM via pip
pip install cryofm
```

This will install CryoFM along with all of its required dependencies in an isolated environment.

---

## 2. Developer Installation (For Contributors)

If you plan to modify or contribute to the CryoFM source code, it's best to set up a separate environment for development:

```bash
# Clone the repository
git clone https://github.com/ByteDance-Seed/cryofm.git
cd cryofm

# Create and activate a new conda environment for development
conda create -n cryofm-dev python=3.9 -y
conda activate cryofm-dev

# Install required dependencies
pip install -r requirements.txt

# Install CryoFM in editable (development) mode
pip install -e .
```

Using a dedicated environment ensures that your development work does not interfere with your main Python setup or other projects.

---

## 3. Additional Notes

- Certain optional features may require extra dependencies. Please see the [documentation](https://github.com/ByteDance-Seed/cryofm) for details.
- For advanced installation options (including GPU support), refer to the project README.
- If you encounter installation or environment issues, see the [Troubleshooting](../troubleshooting/common-issues.md) section.

---

Happy analyzing with CryoFM!

