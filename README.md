# EoL4Chem

EoL4Chem are Python scripts written to track chemical waste flows and identify recycling, energy recovery, treatment & disposal facilities (RETDFs) located across the United States of America, using publicly-available databases. EoL4Chem uses the U.S. Environmental Protection Agency (EPA)'s [Chemical Data Reporting (CDR)](https://www.epa.gov/chemical-data-reporting) to gather information about potential post-recycling scenarios for recycled chemicals (e.g., fuel or fuel agent). EoL4Chem integrates the [PAU4Chem](https://github.com/jodhernandezbe/PAU4Chem) repository, which transforms data for performing chemical flow analysis for pollution abatement units (PAUs) or on-site end-of-life (EoL) operations (e.g., distillation). Also, EoL4Chem incorporates the physical properties (e.g., boiling point) from [another repository](https://github.com/jodhernandezbe/Properties_Scraper) that retrieves the properties using web scraping and automatization. These properties support chemical flow analysis performance.

<p align="center">
  <img src=https://github.com/jodhernandezbe/EoL4Chem/blob/main/EoL4Chem.svg width="100%">
</p>

## Requirements:

This code was written using Python 3.x, Anaconda 3, and operating system Ubuntu 18.04. The following Python libraries are required for running the code:

1. [pandas](https://anaconda.org/anaconda/pandas)
2. [numpy](https://anaconda.org/conda-forge/numpy)
3. [argparse](https://anaconda.org/conda-forge/argparse)
4. [plotly](https://anaconda.org/plotly/plotly)
5. [plotly-orca](https://anaconda.org/plotly/plotly-orca)
6. [psutil](https://pypi.org/project/psutil/)
7. [holoviews](https://anaconda.org/conda-forge/holoviews)
8. [matplotlib](https://anaconda.org/conda-forge/matplotlib)
9. [seaborn](https://anaconda.org/anaconda/seaborn)

The case study needs the files obtained by data engineering, which are in [transform](https://github.com/jodhernandezbe/EoL4Chem/tree/main/transform). The [extract](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract) folder contains all the files got by web scraping and automatization, which together [ancillary](https://github.com/jodhernandezbe/EoL4Chem/tree/main/ancillary) folder files (e.g., fuzzy analytic hierarchy process (FAHP)) support the data engineering process.

## How to use

To run the Python script for the n-hexane case study, you need to navigate to the directory containing circular.py, i.e., [circular](https://github.com/jodhernandezbe/EoL4Chem/tree/main/transform/circular) folder whose full path is */EoL4Chem/transform/circular/*. Then, you execute the following command either on Windows CMD or Unix terminal:

```
python circular.py -CAS 110543 -N_cycles 100
```
or by default running conditions

```
python circular.py
```

## Disclaimer

The views expressed in this article are those of the authors and do not necessarily represent the views or policies of
the U.S. Environmental Protection Agency. Any mention of trade names, products, or services does not imply an endorsement by the U.S.
Government or the U.S. Environmental Protection Agency. The U.S. Environmental Protection Agency does not endorse any commercial products, service, or enterprises.

## Acknowledgement

This research was supported in by an appointment for Jose D. Hernandez-Betancur to the Research Participation
Program at the Center for Environmental Solutions and Emergency Response, Office of Research and Development,
U.S. Environmental Protection Agency, administered by the Oak Ridge Institute for Science and Education through an Interagency Agreement No. DW-89-92433001 between the U.S. Department of Energy and the U.S. Environmental Protection Agency.

