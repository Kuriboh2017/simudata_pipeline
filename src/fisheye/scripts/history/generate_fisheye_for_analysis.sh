#!/bin/bash

run_fisheye_sim_dataset() {
    local sim_dataset=$1
    
    random_seed=$(date +%s)

    rm -rf "${sim_dataset}/group1/data.json"
    find "${sim_dataset}/group1/" -type d \( -name '*_3to1*' -o -name '*_fisheye*' \) -prune -exec rm -r {} \;
    
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -roll 0.1 -pitch 0.1 -yaw 0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -roll -0.1 -pitch -0.1 -yaw -0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -roll 0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -roll -0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -pitch 0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -pitch -0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -yaw 0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}" -yaw -0.1 -noisy
    time run_fisheye_sim_for_analysis.py -i "${sim_dataset}" -c 100 -seed "${random_seed}"
}

run_fisheye_sim_dataset $1
