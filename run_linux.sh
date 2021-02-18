#!/bin/sh

##################
# Unshunted case #
##################

# ng=0
python -m simus run 10 40 1 102 5 params/unshunted/unshunted_final_params.json
python -m simus run 10 40 1 702 10 params/unshunted/unshunted_final_params.json
python -m simus run 10 50 401 502 10 params/unshunted/unshunted_final_params.json
python -m simus run 10 50 601 702 10 params/unshunted/unshunted_final_params.json
# Plot
python scripts/fig2.py out/unshunted_final_params/


################
# Shunted case #
################

# python -m simus run 10 20 1 202 10 params/shunted/shunted_final_params.json
# python -m simus run 10 20 1 10000 100 params/shunted/shunted_final_params.json
# Plot
# python scripts/fig3.py out/shunted_final_params/
# python scripts/fig4.py out/shunted_final_params/
