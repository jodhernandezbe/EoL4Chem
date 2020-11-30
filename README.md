# EoL4Chem

EoL4Chem are Python scripts written to track chemical waste flows and identify recycling, energy recovery, treatment & disposal facilities (RETDFs) located across the United States of America, using publicly-available databases. EoL4Chem uses the U.S. Environmental Protection Agency (EPA)'s [Chemical Data Reporting (CDR)](https://www.epa.gov/chemical-data-reporting) to gather information about potential post-recycling scenarios for recycled chemicals (e.g., fuel or fuel agent). EoL4Chem integrates the [PAU4Chem](https://github.com/jodhernandezbe/PAU4Chem) repository, which transforms data for performing chemical flow analysis for pollution abatement units (PAUs) or on-site end-of-life (EoL) operations (e.g., distillation). Also, EoL4Chem incorporates the physical properties (e.g., boiling point) from [another repository](https://github.com/jodhernandezbe/Properties_Scraper) that retrieves the properties using web scraping and automatization. These properties support chemical flow analysis performance.

<p align="center">
  <img src=https://github.com/jodhernandezbe/EoL4Chem/blob/main/EoL4Chem.svg width="100%">
</p>

## 1. Requirements:

This code was written using Python 3.x, Anaconda 3, and operating system Ubuntu 18.04. The following Python libraries are required for running the code:

1. [chardet](https://anaconda.org/anaconda/chardet)
2. [pandas](https://anaconda.org/anaconda/pandas)
3. [numpy](https://anaconda.org/conda-forge/numpy)
4. [pyyaml](https://anaconda.org/anaconda/pyyaml/)
5. [selenium](https://anaconda.org/conda-forge/selenium)
6. [webdriver-manager](https://pypi.org/project/webdriver-manager/)
7. [regex](https://anaconda.org/conda-forge/regex)
8. [beautifulsoup](https://anaconda.org/anaconda/beautifulsoup4)
9. [requests](https://anaconda.org/anaconda/requests)
10. [argparse](https://anaconda.org/conda-forge/argparse)
11. [fake-useragent](https://anaconda.org/auto/fake-useragent)
12. [nltk](https://anaconda.org/anaconda/nltk)
13. [plotly](https://anaconda.org/plotly/plotly)
14. [plotly-orca](https://anaconda.org/plotly/plotly-orca)
15. [psutil](https://pypi.org/project/psutil/)
16. [holoviews](https://anaconda.org/conda-forge/holoviews)
17. [matplotlib](https://anaconda.org/conda-forge/matplotlib)
18. [seaborn](https://anaconda.org/anaconda/seaborn)

## 2. How to use

### 2.1. Web scraping module

EoL4Chem is a modular framework that uses web scraping to extract the information from the web and organize it before data engineering. To run the web scraping module, navigate to the folder [extract](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract). This folder contains five subfolders [frs](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/frs), [gps](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/gps), [properties](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/properties), [rcrainfo](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/rcrainfo), and [tri](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/tri). The [gps](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/gps) folder has the scripts for using [Nominatim](https://nominatim.org/)'s API and [OSRM](http://project-osrm.org/.)'s, which are called by the data engineering process. [properties](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/properties) is used as explained in this GitHub [repository](https://github.com/jodhernandezbe/Properties_Scraper).

#### 2.1.1. frs folder

This folder contains the information from the EPA's [Facility Registry Service (FRS)](https://www.epa.gov/frs). To retrieve the FRS information, navigate to the folder [frs](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/frs). Then, you execute the following command either on Windows CMD or Unix terminal:

```
python frs_scraper.py
```

#### 2.1.2. rcrainfo folder

This folder contains the information from the EPA's [Resource Conservation and Recovery Act Information (RCRAInfo)](https://www.epa.gov/enviro/rcrainfo-overview). To retrieve the RCRAInfo information, navigate to the folder [rcrainfo](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/rcrainfo). Execute the following command either on Windows CMD or Unix terminal:

```
python rcrainfo_scraper.py --help
```
You could see the following menu:

```
usage: rcrainfo_scraper.py [-h] [-Y YEAR [YEAR ...]] Option

positional arguments:
  Option                What do you want to do: [A] Extract information from RCRAInfo.
                                                [B] Organize files for csv.

optional arguments:
  -h, --help            show this help message and exit
  -Y YEAR [YEAR ...], --Year YEAR [YEAR ...]
                        What Bienniarl report do you want?. Currently up to
                        2017
```
 
Execute each step starting with A and finishing with B. Use flag -Y for choosing the odd years from 2001 to 2017 (the most recent RCRA biennial report).The flag -Y can work for retrieving many years at the same time, for example:

```
python rcrainfo_scraper.py A -Y 2001 2003
```

#### 2.1.3. tri folder

This folder contains the information from the EPA's [Toxics Release Inventory (TRI)](https://www.epa.gov/toxics-release-inventory-tri-program). To retrieve the TRI information, navigate to the folder [tri](https://github.com/jodhernandezbe/EoL4Chem/tree/main/extract/tri). Execute the following command either on Windows CMD or Unix terminal:

```
python tri_web_scraper.py -Y TRI_Year -F TRI_File
```

The flag -Y represents the TRI reporting year that you would like to get, while -F is the file from the TRI for retrieving the information (e.g., File 1a). Check [TRI Basic Plus Data Files Guides
](https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-plus-data-files-guides) for knowing more about the TRI files. EoL4Chem requires the files 1a, 1b, 1b, 2a, 2b, 3a, 3b, and 3c to run the data engineering. The above command can work for retrieving many years and files at the same time, for example:

```
python tri_web_scraper.py -Y 2001 2002 2003 -F 1a 1b 2b
```

### 2.2. Data engineering module

## Disclaimer

The views expressed in this article are those of the authors and do not necessarily represent the views or policies of
the U.S. Environmental Protection Agency. Any mention of trade names, products, or services does not imply an endorsement by the U.S.
Government or the U.S. Environmental Protection Agency. The U.S. Environmental Protection Agency does not endorse any commercial products, service, or enterprises.

## Acknowledgement

This research was supported in by an appointment for Jose D. Hernandez-Betancur to the Research Participation
Program at the Center for Environmental Solutions and Emergency Response, Office of Research and Development,
U.S. Environmental Protection Agency, administered by the Oak Ridge Institute for Science and Education through an Interagency Agreement No. DW-89-92433001 between the U.S. Department of Energy and the U.S. Environmental Protection Agency.

