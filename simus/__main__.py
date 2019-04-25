#!/usr/bin/env python
# coding: utf-8
"""
Floquet simulations of a transmon coupled to cavity.
"""
from __future__ import absolute_import, print_function, unicode_literals

import argparse
import logging
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

from simus import f_energies
from simus import run_simus
from simus import tools

# Init logging
logging.basicConfig(
    format='%(name)s - %(asctime)s - %(message)s'
)

# Define and parse CLI arguments
parser = argparse.ArgumentParser(
    prog='simus',
    description='Run Floquet simulations of a transmon coupled to a cavity'
)
subparsers = parser.add_subparsers(title='available subcommands',
                                   dest='subcommand',
                                   help='subcommand')
subparsers.required = True

# Single subcommand, to run a single simulation with a given set of params.
single_parser = subparsers.add_parser(
    'single',
    help='Run a single simulation, dumping data and plotting steadystate.'
)
single_parser.add_argument('n_bar', type=float,
                           help='Mean number of photon in the pump drive.')
single_parser.add_argument('N_max_a', type=int,
                           help='Truncature of the Fock space of the cavity.')
single_parser.add_argument('N_max_b', type=int,
                           help='Truncature of the transmon eigenspace.')
single_parser.add_argument('params_file',
                           help='File to load parameters from.')

# Run subcommand, to run a batch of simulations
run_parser = subparsers.add_parser(
    'run',
    help=('Run a batch of simulations in parallel, dumping data and '
          'plotting steadystate.')
)
run_parser.add_argument('N_max_a', type=int,
                        help='Truncature of the Fock space of the cavity.')
run_parser.add_argument('N_max_b', type=int,
                        help='Truncature of the transmon eigenspace.')
run_parser.add_argument('n_bar_min', type=float,
                        help='Lower mean number of photon in the pump drive.')
run_parser.add_argument('n_bar_max', type=float,
                        help='Upper mean number of photon in the pump drive.')
run_parser.add_argument('n_bar_step', type=float,
                        help='Step on mean number of photon in the pump drive.')
run_parser.add_argument('params_file',
                        help='File to load parameters from.')

# TODO: Integrate transmon alone model

# f_energies subcommand, to plot Floquet quasi-energies as a function of n_bar
f_energies_parser = subparsers.add_parser(
    'f_energies',
    help='Compute Floquet quasi-energies on a range of n_bar values.'
)
f_energies_parser.add_argument('N_max_a', type=int,
                        help='Truncature of the Fock space of the cavity.')
f_energies_parser.add_argument('N_max_b', type=int,
                        help='Truncature of the transmon eigenspace.')
f_energies_parser.add_argument('n_bar_min', type=float,
                        help='Lower mean number of photon in the pump drive.')
f_energies_parser.add_argument('n_bar_max', type=float,
                        help='Upper mean number of photon in the pump drive.')
f_energies_parser.add_argument('n_bar_step', type=float,
                        help='Step on mean number of photon in the pump drive.')
f_energies_parser.add_argument('params_file',
                        help='File to load parameters from.')

# Parse arguments
parsed_args = parser.parse_args()

# Load parameters from file
params = tools.load_params(parsed_args.params_file)
out_directory = os.path.join(
    'out',
    os.path.splitext(
        os.path.basename(parsed_args.params_file)
    )[0],
    '%d_%d' % (parsed_args.N_max_a, parsed_args.N_max_b)
)

# Handle the correct subcommand call
if parsed_args.subcommand == 'single':
    run_simus.run_single_simulation(
        parsed_args.n_bar, parsed_args.N_max_a, parsed_args.N_max_b, params,
        out_directory=out_directory
    )
elif parsed_args.subcommand == 'run':
    n_bar_list = np.arange(parsed_args.n_bar_min,
                           parsed_args.n_bar_max,
                           parsed_args.n_bar_step)
    run_simus.run_simulations(
        n_bar_list, parsed_args.N_max_a, parsed_args.N_max_b, params,
        out_directory=out_directory
    )
elif parsed_args.subcommand == 'f_energies':
    n_bar_list = np.arange(parsed_args.n_bar_min,
                           parsed_args.n_bar_max,
                           parsed_args.n_bar_step)
    f_energies.plot_f_energies(
        n_bar_list, parsed_args.N_max_a, parsed_args.N_max_b, params
    )
