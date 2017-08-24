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
$ pip install numpy pandas matplotlib seaborn
```

# Examples

Run plot_timecourse.py to plot your timecourse data. Execute it from the top-level directory of this project so that the imports will work. This script will blank and smooth your data (using default parameters), take the mean of all your replicates (as defined by your plate specification CSV file) and then plot them against time. 

```bash
$ python growth/scripts/plot_timecourse.py -p growth/plate_specs/example_plate_spec.csv growth/data/example_data.xlsx
```

The above example uses data where only one thing (absorbance at 600 nm) was measured. But you will often have output that measures more than one thing (e.g. aborbance and GFP fluorescence). You can use the same script to plot a single measurement label (e.g. 'GFP') from the file against time.

```bash
$ python growth/scripts/plot_timecourse.py -p growth/plate_specs/example_plate_spec_multimeasurement.csv growth/data/example_data_multimeasurement.xlsx -m GFP
```

You can also find more detailed example code in this [iPython notebook](https://github.com/flamholz/tecantaloupe/blob/master/notebooks/Example_MultiMeasurement.ipynb). 