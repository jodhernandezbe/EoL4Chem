# EoL4Chem

EoL4Chem are Python scripts written to track chemical waste flows across different locations in the United States of America, using publicly-available databases. EoL4Chem uses the U.S. Environmental Protection Agency's [Chemical Data Reporting (CDR)](https://www.epa.gov/chemical-data-reporting) to gather information about potential post-recycling scenarios for recycled chemicals (e.g., fuel or fuel agent). EoL4Chem integrates [PAU4Chem](https://github.com/jodhernandezbe/PAU4Chem) repository which transforms data for performing chemical flow analysis for pollution abatement units (PAUs) or on-site end-of-life (EoL) operations (e.g., distillation). In addition, EoL4Chem incorporates the physical properties (e.g., boiling point) from [another repository] (https://github.com/jodhernandezbe/Properties_Scraper) that retrieves the properties using web scraping and automatization.

<sup>[1](#myfootnote1)</sup>

<p align="center">
  <img src=https://github.com/jodhernandezbe/EoL4Chem/blob/main/EoL4Chem.svg width="100%">
</p>

# Requirements:

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

# Disclaimer

The views expressed in this article are those of the authors and do not necessarily represent the views or policies of
the U.S. Environmental Protection Agency. Any mention of trade names, products, or services does not imply an endorsement by the U.S.
Government or the U.S. Environmental Protection Agency. The U.S. Environmental Protection Agency does not endorse any commercial products, service, or enterprises.

# Acknowledgement

This research was supported in by an appointment for Jose D. Hernandez-Betancur to the Research Participation
Program at the Center for Environmental Solutions and Emergency Response, Office of Research and Development,
U.S. Environmental Protection Agency, administered by the Oak Ridge Institute for Science and Education through an Interagency Agreement No. DW-89-92433001 between the U.S. Department of Energy and the U.S. Environmental Protection Agency.

-----------------------------------------------------------------------------------------------------------------------------

<a name="myfootnote1">1</a>: Recycling, energy recovery, treatment & disposal facility (RETDF). Pollution abatement unit (PAU).
