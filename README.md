tecantaloupe
============

Python-based pipeline for parsing and manipulating output from plate readers. For the moment the code is focused on 96-well plate format output from Tecan plate readers, but it is sufficiently modular to accomodate any brand's output so long as you are willing to write a parser for their output format.

# Current Features 

* Ability to specify per-well labels and conditional information (media, induction).
* Parsing of Excel output from iContol.
* Built-in blanking and data-smoothing. 
* Aggregation of data by well labels for plotting and analysis.

# Planned features 

* Calculation of growth rates and yields for microbial growth curves. 

# Dependencies

* numpy
* pandas
* matplotlib (for plotting scripts)
* seaborn (for plotting scripts)

You can install all dependencies using pip as follows

```bash
pip install numpy pandas matplotlib seaborn
```