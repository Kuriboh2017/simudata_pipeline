#!/bin/bash

for sim_dataset_path in "$@"; do
    script -c "time generate_fisheye_for_analysis.sh ${sim_dataset_path}" ${sim_dataset_path}.log
done

