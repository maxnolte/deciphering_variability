## deciphering_variability

This is the analysis code for a revised version of the preprint
["Cortical reliability amid noise and chaos"](https://doi.org/10.1101/304121).
The study is fully based on a previously published neocortical
microcircuit model and associated methods (https://doi.org/10.1016/j.cell.2015.09.029). [NEURON](https://www.neuron.yale.edu/neuron/) models and detailed
 information about the microcircuit can be found at the Blue Brain Project portal (https://bbp.epfl.ch/nmc-portal/downloads).

The scripts and modules in this repo were used to analyze ~32 terabytes of new simulation output, generated with the previously published microcircuit model.
The simulation output and libraries to read it can be shared upon reasonable request. Please contact Max Nolte at
[max.nolte@epfl.ch]().

### Structure

Main analysis and figure creation:
* initial_analysis_final.py - Main divergence analysis (Figs. 1-5)
* analysis_exp_25_decoupled.py - Reliablity & decoupled reliability analysis (Figs. 6-7)

Additional analyses:
* analysis_poisson.py - Fano factor analysis I
* reliability mechanisms.py - In-degree analysis
* analysis_stimulus.py - Preliminary stimulus divergence analysis

Validations:
* stim_amplitude_scan.py - Validation of noise stimulus
* ca_scan.py - MVR/Det. synapses validation
* auto_correlation.py - Correlation validation
* analysis_spike_shift.py - Spike injection validation
* analysis_spike_injection_FR.py - Spike injection validation
* analysis_noise_stim.py - Noise stimulus validations

Modules:
* correlations.py - Functions to compute r_V(t) and RMSD_V(t)
* magicspike/distances.py - Functions to compute reliability

Jupyter notebooks:
* plot_voltage_traces_exp_25.ipynb - Plot example voltage traces
* stimulus_analysis.ipynb - Plot normal and decoupled r
* raster_long_exp_25.ipynb - Plot long raster plot to stimulus

Revision analyses:
* revision/data_access_shuffling.py - Data for calcium and jitter simulations
* revision/criticality_analysis.py - Criticality and jitter (Fig. 8)
* revision/generate_jittered_inputs.py - Script to generate jittered and variable inputs
* revision/jitter_flick_analysis.py - Jitter and variable input analysis

### Getting started

To generate figures, update simulation output paths to new data location, and then run figure generation scripts in Python 2.7, e.g.:
```
python intitial_analysis_final.py
```

## License

The code in this repository is licensed under the MIT license.
