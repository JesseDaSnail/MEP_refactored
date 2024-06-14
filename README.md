# Modeling Parametric Source Arrays using Finite-Difference Time-Domain Methods

## About

This repository contains code for modeling parametric source arrays using the finite-difference time-domain (FDTD) method. It is part of the thesis "Modeling Parametric Source Arrays using Finite-Difference Time-Domain Methods" by Jesse Buijs at Delft University of Technology.

## Installation
The code uses the Nvidia CUDA toolkit to perform computations in parallel on the GPU. An Nvidia graphics card is therefore required.

Installation steps:
1. Install the CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
2. Clone the repository
3. Install the dependencies (see [requirements.txt](requirements.txt)). Preferably in a virtual environment.

## Code structure

To run the simulation, I prefer using a jupyter notebook. This makes it possible to adjust plotting/analysis parameters without needing to run/load a new simulation. It can be found in [main.ipynb](main.ipynb).

[model.py](model.py) contains the Model object that describes the domain's dimensions, discretization, medium parameters and PML parameters.

[sources.py](sources.py) contains the different types of sources that inherit from the parent class "Source". These objects describe the pulse amplitude, width of the source, pulse time window length, and focussing.

[simulation.py](simulation.py) contains the main simulation loop. The FDTD_sim class sets up the simulation by accepting a (Model) that describes the domain and can add one or more (Source) objects. Currently only the first source is supported.

[cuda_kernels_2D.py](cuda_kernels_2D.py) contains the CUDA code that will run on the GPU. These kernels are called from FDTD_sim in [simulation.py](simulation.py).

[analysis.py](analysis.py) contains the "SimulationResult" object that stores the result from one run of the simulation. The "SimulationResultAnalyser" takes a SimulationResult as input and contains analysis functions.

[filtering.py](filtering.py) contains functions to filter the time axis of a pressure field. Currently only a simple lowpass filter is implemented that sets fourier coefficients above a certain frequency to zero.

[io_util.py](io_util.py) contains functions to save and load "SimulationResult" objects using pickle. This prevents the need to rerun long simulations.

[main_domain_extender.ipynb](main_domain_extender.ipynb) is a jupyter notebook that can be used to perform domain extension on a SimulationResult obtained from the main simulation.


