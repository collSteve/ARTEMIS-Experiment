# ARTEMIS Project Setup Troubleshooting Log

This document records the extensive troubleshooting process undertaken to set up the ARTEMIS project on a Linux machine with an NVIDIA GPU (Driver 550.120, CUDA 12.4 support).

## Objective

To create a stable Conda environment to run the project's training scripts on a GPU.

---

### Attempt 1: Initial Setup via `requirements.txt`

-   **Action:** Analyzed `requirements.txt` to create a Conda environment and install packages via `pip`.
-   **Problem:** Quickly identified that this approach is too brittle for the PyTorch Geometric (PyG) stack, as `torch-scatter` and `torch-sparse` require binaries that are precisely matched to the PyTorch and CUDA versions.

---

### Attempt 2: Refined `pip` Install with Specific CUDA Wheels (PyTorch 2.4.0)

-   **Action:**
    1.  `conda create -n artemis-env python=3.10 -y`
    2.  `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121`
    3.  `pip install torch_scatter==2.1.2 torch_sparse==0.6.18 torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html`
-   **Result:** `RuntimeError: No CUDA version supported` when running the script.
-   **Conclusion:** The pre-compiled binaries from the PyG channel were incompatible with the host system's driver/runtime environment.

---

### Attempt 3: Recompiling PyG from Source

-   **Action:** To solve the binary incompatibility, we attempted to compile the PyG libraries from source.
    1.  `pip uninstall torch-scatter torch-sparse torch-geometric -y`
    2.  `pip install --no-binary :all: torch_scatter==2.1.2 ...`
-   **Result:** Build failed with `OSError: .../lib/libstdc++.so.6: version 'GLIBCXX_3.4.32' not found`.
-   **Conclusion:** The C++ compiler (`g++`) available to `pip` was newer than the C++ standard library available inside the base Conda environment, causing a linker error.

---

### Attempt 4: Upgrading Conda's C++ Toolchain

-   **Action:** To fix the `GLIBCXX` error, we installed a newer C++ toolchain directly into the Conda environment.
    1.  `conda install -c conda-forge gxx_linux-64`
-   **Result:** The same `OSError: ... GLIBCXX_3.4.32' not found` persisted when running the script.
-   **Conclusion:** The PyG libraries were already compiled and linked against the old toolchain. They needed to be recompiled *after* the new toolchain was in place.

---

### Attempt 5: Recompiling PyG After C++ Toolchain Upgrade

-   **Action:** We re-ran the compilation after installing the newer C++ toolchain.
    1.  `pip uninstall ...`
    2.  `pip install --no-cache-dir --no-binary :all: ...`
-   **Result:** Build failed with `error: #error -- unsupported GNU version! gcc versions later than 13 are not supported!`.
-   **Conclusion:** The `gxx_linux-64` we installed from `conda-forge` was too new (v15) for the system's NVIDIA CUDA compiler (`nvcc`), which only supported `gcc` up to v13.

---

### Attempt 6: Full `conda` Installation (Multiple Attempts)

-   **Action 1:** `conda create -c pyg ... pyg ...`
-   **Result 1:** Failed with `LibMambaUnsatisfiableError: nothing provides libcublas`.
-   **Action 2:** `conda create -c pytorch -c nvidia -c pyg ... pyg ...`
-   **Result 2:** Succeeded, but running the script gave `ModuleNotFoundError: No module named 'torch_sparse'`.
-   **Action 3:** `conda create -c pytorch -c nvidia -c pyg ... torch-geometric torch-scatter torch-sparse ...`
-   **Result 3:** Failed with `PackagesNotFoundError: ... torch-sparse`.
-   **Conclusion:** A pure `conda` installation was not a viable path, as the required packages were not available on the channels for this specific system configuration.

---

### Attempt 7: CPU-Only Environment as a Sanity Check

-   **Action:** To verify the script's integrity, we created a CPU-only environment using `pip`.
    1.  `conda create -n artemis-cpu-env python=3.10 -y`
    2.  `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu`
    3.  `pip install torch_scatter torch_sparse torch_geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html`
-   **Result:** **Success.** The script ran without errors on the CPU.
-   **Conclusion:** The Python code is correct. The problem is confirmed to be entirely within the GPU hardware/software stack.

---

### Attempt 8: GPU Environment with PyTorch 2.1

-   **Action:** Hypothesizing that PyTorch 2.4 was the issue, we tried an older version, PyTorch 2.1.
    1.  `conda create -n artemis-gpu-env -c conda-forge python=3.10 gxx_linux-64=11 ... -y` (to ensure a compatible compiler)
    2.  `pip install torch==2.1.2 ... --index-url .../cu121`
    3.  `pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html` (using pre-compiled binaries)
-   **Result:** `RuntimeError: Could not infer dtype of numpy.bool`.
-   **Conclusion:** This was a known incompatibility between PyTorch 2.1 and NumPy 2.x.

---

### Attempt 9: Downgrading NumPy

-   **Action:** To fix the NumPy error, we downgraded it.
    1.  `pip install "numpy<2"`
-   **Result:** The script failed again with the original `RuntimeError: No CUDA version supported`.
-   **Final Conclusion:** This was the final piece of evidence. Even with a compatible compiler and NumPy version, the pre-compiled binaries for PyTorch Geometric 2.5.3 and PyTorch 2.1.2 are fundamentally incompatible with the host system's driver and CUDA runtime. All attempts to install pre-compiled binaries or compile from source have failed due to a series of intractable environment conflicts. The only remaining robust solution is to sidestep the host environment entirely, for example by using a Docker container.
