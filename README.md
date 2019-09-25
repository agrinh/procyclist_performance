# Summary
This package contains the official implementation of [Towards Machine Learning on data from Professional Cyclists](https://arxiv.org/abs/1808.00198). It is built to predict the heart rate of professional cyclists given time series of measurements collected by bike computers like srm or pioneer devices. The current configuration gives the algorithm feedback of the heart rate from 30 time steps ago to allow it to  model the cyclists physiological response to work. 

# Installation
The recommended method of installation is to first clone the repository and install it in edit mode, to allow modifying the configuration to your usage:

```shell
git clone git@github.com:agrinh/procyclist_performance.git
pip install -e procyclist_performance
```

# Usage
Run a training session with:

```shell
python -m procyclist_performance.train
```

Modify which parameters to extract in [dataset.py](procyclist/dataset.py). The first training will produce a cache of the preprocessed data which is re-used on subsequent runs, until the `load` function is modified. The input and target paramters are chosen in `main` of [train.py](procyclist/train.py).

# Data

## Concepts
The package reads collections of data or `Sessions`. This makes it possible to use it separately for e.g. different groups of cyclists in the team. Each collection of `Sessions` may contain multiple riders, each with a number of training or racing sessions. Furthermore, since data is often collected by different devices, each configured device should have a glob pattern to match data produced by that particular device.

## Configuration
The package is configured by setting the configuration files in (procyclist/config)[procyclist/config]. Configuring:

- Collection of `Sessions` : [sessions.cfg](procyclist/config/sessions.cfg)
- Devices used to collect data: [devices.cfg](procyclist/config/devices.cfg)

### Collection of Sessions
Follow the examples in [sessions.cfg](procyclist/config/sessions.cfg) to add new collections. The `DEFAULT` section (which applies to all collections) defines:

- Name of matrix in each `<cyclist name>.mat` file containing metadata about cyclist (`meta_matrix`)
- Relative path from package to file specifying names for each column in the metadata matrix (`meta_parameters`)
- Path to use for storing preprocessed data (`cache_path`)

By default the package has the following data collections (`Sessions`) configured:
- example: /data/example
- men: /data/men
- women: /data/women

### Devices
The [devices.cfg](procyclist/config/devices.cfg) comes with settings for SRM and Pioneer devices (though these must be altered to match how you handle your data). Specified for each device is:

- Name of matrix in each `<session>.mat` file with session information (`matrix`)
- Relative path from package to file specifying names for each column in the matrix (`parameters`)
- A [filename pattern](https://docs.python.org/3/library/glob.html) which should match all filenames produced by the device ('filename')


## Structure

In the root directory of a collection of `Sessions` there should be directories corresponding to cyclist names. Each should contain a metadata `.mat` file with the riders name, and a data directory with `.mat` files for each cycling session. These cycling session files may carry any name, but should have a name matching the device pattern string.

### Example
Below is an example directory structure compatible with this package and the `Session` collection `example` (defined in [sessions.cfg](procyclist/config/sessions.cfg)) and device `pioneer` (defined in [devices.cfg](procyclist/config/devices.cfg)).

```
/data/example
├── cyclist_1
│   ├── cyclist_1.mat
│   └── data
│         ├── session_1_pioneer.mat
│         ├── session_2_pioneer.mat
│         └── < ... >
├── cyclist_2
│   ├── cyclist_2.mat
│   └── data
│         ├── session_1_pioneer.mat
│         ├── session_2_pioneer.mat
│         └── < ... >
.
.
.
```
This configuration expects the cyclist metadata `.mat` files (`cyclist_n.mat`) to contain a matrix named `alles`, with columns specified by [config/parameters_alles.csv](procyclist/config/parameters_alles.csv). Individual sessions (`session_n_pioneer.mat`) should contain a cell array named `data_pioneer`, with element {0}{0} being a matrix where rows are timesteps and columns are as specified in [config/parameters_alles.csv](procyclist/config/parameters_pioneer.csv).

# Citations
If you found this package useful in your research please use the following citation for the paper (Towards Machine Learning on data from Professional Cyclists)[https://arxiv.org/abs/1808.00198]:

```
@inproceedings{hilmkilcycling,
  title={Towards machine learning on data from professional cyclists},
  author={Hilmkil, Agrin and Ivarsson, Oscar and Johansson, Moa and Kuylenstierna, Dan and van Erp, Teun},
  booktitle={Proceedings of the 12th World Congress on Performance Analysis of Sports},
  publisher={Faculty of Kinesiology, University of Zagreb},
  pages={168--176}
  address={Opatija, Croatia},
  year={2018}
}
```
