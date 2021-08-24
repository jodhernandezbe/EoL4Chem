# Overview

EoL4Chem are Python scripts written to track chemical waste flows and identify recycling, energy recovery, treatment & disposal facilities (RETDFs) located across the United States of America, using publicly-available databases. EoL4Chem uses the U.S. Environmental Protection Agency (EPA)'s [Chemical Data Reporting (CDR)](https://www.epa.gov/chemical-data-reporting) to gather information about potential post-recycling scenarios for recycled chemicals (e.g., fuel or fuel agent). EoL4Chem integrates the [PAU4Chem](https://github.com/jodhernandezbe/PAU4Chem) repository, which transforms data for performing chemical flow analysis for pollution abatement units (PAUs) or on-site end-of-life (EoL) operations (e.g., distillation). Also, EoL4Chem incorporates the physical properties (e.g., boiling point) from [another repository](https://github.com/jodhernandezbe/Properties_Scraper) that retrieves the properties using web scraping and automatization. These properties support chemical flow analysis performance.

<p align="center">
  <img src=https://github.com/jodhernandezbe/EoL4Chem/blob/main/EoL4Chem.svg width="100%">
</p>

## EoL4Chem folder structure (tree)

The following is the project structure for the EoL4Chem folder. Only the most important files/folders are shown

```
.
├── ancillary
│   ├── cdr
│   ├── fahp
│   ├── normalizing_naics
│   ├── others
│   ├── pau_flow_allocation
│   ├── plots
│   ├── rcrainfo
│   ├── releases
│   └── tri
├── extract
│   ├── frs
│       └── frs_scraper.py
│   ├── gps
│   ├── properties
│   │   ├── cameo
│   │   ├── comptox
│   │   ├── ifa
│   │   ├── nist
│   │   ├── osha
│   │   └── main.py
│   ├── rcrainfo
│   │   └── rcrainfo_scraper.py
│   └── tri
│       └── tri_scraper.py
└── transform
    ├── cdr
    │   └── cdr_transformer.py
    ├── circular
    │   └── circular.py
    ├── pau4chem
    │   └── building_pau_db.py
    ├── source_reduction
    │   └── building_source_reduction.py
    ├── tri
    │   └── tri_transformer.py
    └── waste_tracking
        ├── disposal_activities.py
        ├── off_tracker_transformer.py
        └── on_tracker_transformer.py
```


# Requirements

## Using Git Large File Storage (git-lfs)

You must use git lfs in order to retrieve all the files and save them in your computer. Take a look at the [.gitattributes](https://github.com/jodhernandezbe/EoL4Chem/blob/main/.gitattributes) file to see the list of files pushed using git-lfs command.

## Creating conda virtual environment

This code was written using Python 3.x, Anaconda 3, and operating system Ubuntu 18.04. You can create a conda environment called EoL4Chem_env by running the following command after installing [Anaconda](https://www.anaconda.com/) in your computer:

```
conda env create -f environment.yml
```

The case study needs the files obtained by data engineering, which are in [transform](https://github.com/jodhernandezbe/EoL4Chem/tree/main/transform). The [extract](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract) folder contains all the files retrieved by web scraping and automatization, which together [ancillary](https://github.com/jodhernandezbe/EoL4Chem/tree/main/ancillary) folder files (e.g., fuzzy analytic hierarchy process (FAHP)) support the data engineering process.

## Ovoiding ModuleNotFoundError and ImportError

If you are working as a Python programmer, you should avoid both ```ModuleNotFoundError``` and ```ImportError```. Thus, follow the steps below to solve the above mentioned problems:

<ol>
   <li>
    Open the .bashrc file with the text editor of your preference (e.g., Visual Studio Code)
        
    code ~/.bashrc
   </li>
   <li>
    Scroll to the bottom of the file and add the following lines
       
    export PACKAGE=$(locate -br '^EoL4Chem$')
    export PYTHONPATH="${PYTHONPATH}:$PACKAGE"
   </li>
   <li>
    Save the file with the changes
   </li>
   <li>
    You can open another terminal to verify that the variable has been successfully saved by running the following command
    
    echo $PYTHONPATH
   </li>
</ol>

# How to use (circular life cycle analysis)

To run the Python script for the n-hexane case study, you need to navigate to the directory containing circular.py, i.e., [circular](https://github.com/jodhernandezbe/EoL4Chem/tree/main/transform/circular) folder whose full path is */EoL4Chem/transform/circular/*. Then, you execute the following command on Unix terminal:

```
python circular.py -CAS 110543 -N_cycles 100
```
or

```
python circular.py
```

## Outputs

After running the Python script you obtain the following output files in the [circular](https://github.com/jodhernandezbe/EoL4Chem/tree/main/transform/circular) folder:

| File name | Description |
| ------------- | ------------- |
| Sankey_1.pdf | Sankey diagram in Figure 4A |
| Sankey_flows_1.csv | Flows between nodes in Figure 4A |
| Sankey_2.pdf | Sankey diagram in Figure 4B |
| Sankey_flows_2.csv | Flows between nodes in Figure 4B |
| stacked_barplot.pdf | Stacked bar plot or Figure 5 |
| chord_waste.pdf | Chord diagram or Figure 6 |
| chord_waste.csv | Labels for industry sectors in Figure 6 |
| chord_recyled.pdf | Chord diagram or Figure 7 |
| chord_recyled.csv | Labels for industry sectors in Figure 7 |
| seaborn_heatmap_non_industrial.pdf | Plot for chemical cross-contamination or Figure 8 |

# Disclaimer

The views expressed in this article are those of the authors and do not necessarily represent the views or policies of
the U.S. Environmental Protection Agency. Any mention of trade names, products, or services does not imply an endorsement by the U.S.
Government or the U.S. Environmental Protection Agency. The U.S. Environmental Protection Agency does not endorse any commercial products, service, or enterprises.

# Acknowledgement

This research was supported in by an appointment for Jose D. Hernandez-Betancur to the Research Participation
Program at the Center for Environmental Solutions and Emergency Response, Office of Research and Development,
U.S. Environmental Protection Agency, administered by the Oak Ridge Institute for Science and Education through an Interagency Agreement No. DW-89-92433001 between the U.S. Department of Energy and the U.S. Environmental Protection Agency.

