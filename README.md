# MGF2PyTorch: Efficient Storage and Querying of MS/MS Spectra for Machine Learning

This repository contains a Python implementation for parsing MGF (Mascot Generic Format) files and converting them into a compact, queryable HDF5 format optimized for downstream machine learning applications using PyTorch.

## Overview

Mass spectrometry-based proteomics produces large volumes of tandem MS/MS spectra. To efficiently use this data for ML tasks, we provide a tool that:

- Parses MGF files manually or using `pyteomics`
- Stores spectra in a flat HDF5 layout with fast indexed access
- Provides PyTorch-compatible `Dataset` and `DataLoader` support
- Supports filtering by charge and m/z
- Includes a Jupyter notebook demonstrating usage
