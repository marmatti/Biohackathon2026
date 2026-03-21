# Filament Simulator

A lightweight Python simulator for generating **rule-based synthetic fluorescence microscopy movies** of **yeast-like cells** with optional **intracellular filament formation**. The simulator is intended for a rapid prototyping of synthetic datasets and stress-testing computer vision methods under different image quality conditions

---

## Overview

The simulator generates movies of fluorescent cell-like objects together with filament events that can:

- appear and disappear over time
- grow, persist, and shrink
- rotate and translate within a host cell
- optionally bend / curve
- be rendered with optical blur and imaging noise

Each simulation run produces:
- a **clean movie** before background and detector noise
- a **noisy movie**
- ordered **PNG frames**
- **movie-level labels**
- **frame-level labels**

---

## Main components

### `configs/base.yaml`
Contains the base configuration for the simulator.

This file controls:
- movie dimensions and frame count
- cell number, size, brightness, spacing, and ellipticity
- filament event timing and probability
- filament length, thickness, motion, and curvature
- optics and defocus
- background and detector noise
- output filenames and directories

This is the main file to edit when tuning the simulation.

### `simulator/runner.py`
This is the main batch execution entry point.

It does the following:
1. loads a base YAML config
2. optionally applies user-specified parameter overrides
3. determines the seed(s) to use
4. runs one or multiple simulation jobs
5. creates one output directory per run
6. saves movies, PNG frames, and labels for each run

### `run_sim.ipynb`
Example notebook showing how to use `runner.py` to generate simulation outputs.

### `movie.ipynb`
Example notebook for visualizing generated movies.

---

## Typical workflow

1. edit `configs/base.yaml` or create a new yaml
2. optionally decide on parameter overrides
3. run the simulator using `runner.py` or `run_sim.ipynb`
4. inspect outputs in the generated run directory
5. visualize the results in `movie.ipynb`