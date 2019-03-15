## variability_analysis

This is the analysis code for the preprint
["Cortical reliability amid noise and chaos"](https://doi.org/10.1101/304121).
The study is fully based on a previously published neocortical
microcircuit model and associated methods (https://doi.org/10.1016/j.cell.2015.09.029). [NEURON](https://www.neuron.yale.edu/neuron/) models and detailed
 information about the microcircuit can be found at the Blue Brain Project portal (https://bbp.epfl.ch/nmc-portal/downloads).

This library was used to analyze 32 terabytes of new simulation output, generated with the previously published microcircuit model.
The simulation output and a library to read it can be shared upon reasonable request. Please contact Max Nolte at
[max.nolte@epfl.ch]().



### Structure

Main analysis and figure creation:
* initial_analysis_final.py
* analysis_exp_25_decoupled.py
* reliability mechanisms.py
* analysis_stimulus.py

Additional analyses:
* stim_amplitude_scan.py
* ca_scan.py
* auto_correlation.py
* analysis_spike_shift.py
* analysis_spike_injection_FR.py
* analysis_spatial_structure.py
* analysis_poisson.py
* analysis_noise_stim.py

Libraries:
* correlations.py
* RELIABILITY!

Jupyter notebooks:
* plot_voltage_traces_exp_25.ipynb
* stimulus_analysis.ipynb
* raster_long_exp_25.ipynb


### Getting started

To generate figures, update simulation output paths to new data location, and then run figure generation scripts in Python 2.7, e.g.:
```
python intitial_analysis_final.py
```

## License

The code in this repository is licensed under the MIT license.
