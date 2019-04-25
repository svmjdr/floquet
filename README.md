Simulations of Josephson circuits using Floquet-Markov theory
=============================================================

This repository contains code to run Floquet-Markov simulations on Josephson
circuits such as a transmon coupled to a cavity.


## Getting started

```bash
# First, clone this repository
git clone https://gitlab.inria.fr/lverney/floquet-markov-for-josephson-circuits
# Then, install all required Python modules
cd floquet-markov-for-josephson-circuits
pip install -r requirements.txt
```

You can now use this simulation framework. Typically, running
```bash
python -m simus --help
```
will list all the available commands and expected parameters.


## Usage

The two main command lines are:

**Note:** Be careful that of the position of the `n_bar` parameter is not the
same in `single` and `run` commands.

1. `python -m simus single n_bar N_max_a N_max_b PATH_TO_PARAMS_FILE`
to run a single simulation using the parameters described in the parameter
file `PATH_TO_PARAMS_FILE`, troncatures `N_max_a` (cavity) and `N_max_b`
(transmon) and a drive strength such that the cavity is centered around
`n_bar` photons.

2. `python -m simus run N_max_a N_max_b n_bar_min n_bar_max n_bar_step PATH_TO_PARAMS_FILE`
to tun a batch of simulations using the same parameters as before and sweeping
`n_bar` between `n_bar_min` and `n_bar_max` with a step of `n_bar_step`.
Simulations are run in parallel, as much as possible.

Additionnally, there is a `run.sh` script in the root of this repository which
runs all the simulations required to plot the figures from the paper, for easy
reproducibility.


When you run any of the above mentionned commands, the program will compute
steadystate of the system, using Floquet framework. Computed data will be
dumped as Python `pickle` files, along with graphs of initial state
decomposition and steadystate decompisition, under the `out/` folder. All
computed data are dumped, refer to the end of the
`simus.run_simus.run_single_simulation` function to have a complete list of
the dumped data.


Then, there are a couple of scripts available under the `scripts` folder, to
post-process the dumped data and plot additional graphs, such as real
populations graphs and Stark-shift computation. This two-pass architecture
lets you simulate one (which is the longer phase) and then generate graphs
on the fly from your dumped data.


## Reproducing simulations from the paper

There is a `run.sh` shell script at the root of this repository. Running it
should run all the required simulations and plot figures from the article.


## How to specify parameters?

You should specify the parameters to use with the current model in a `JSON`
file. For convenience, they are grouped in a `params` folder at the moment.

The provided ones are:
* `unshunted/super_easy_params.json` a set of parameters for super quick and
  easy calculations, mainly operating in the Cooper Pair Box regime.
* `unshunted/easy_params.json` another set of parameters for quick and easy
  calculations, mainly operating in the Cooper Pair Box regime. Useful to
  compare with `shunted/easy_params.json` case.
* `unshunted/exp_params.json` a complete set of parameters matching
  experimental parameters from Zaki.
* `shunted/easy_params.json` a set of parameters for quick and easy
  calculations of a shunted transmon coupled to a cavity. This one can be
  compared easily to the behavior of `unshunted/easy_params.json`.

Each parameter is a JSON dictionary with three fields:
* `value` being the actual value of the parameter.
* `unit` being the unit of the parameter. At the moment, only `'GHz'` (handles
  multiplication by `2 pi`) and `''` (no unit) are supported. The reference
  unit is the `GHz` for frequencies.
* `comment` being a text string to explain what is the parameter used for.


## License

This code is licensed under [MIT](https://opensource.org/licenses/MIT)
license. You are free to use this software, with or without modification,
provided that the conditions listed in the LICENSE.txt file are satisfied.

``simus/floquet.py`` file is a modified version of the
[QuTiP](https://github.com/qutip/qutip) Python package, licensed under BSD
license.
