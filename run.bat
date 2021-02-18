@ECHO OFF
:: Unshunted case


python -m simus run 10 40 1 100 5 params/unshunted/unshunted_final_params.json
python -m simus run 10 40 101 200 5 params/unshunted/unshunted_final_params.json



