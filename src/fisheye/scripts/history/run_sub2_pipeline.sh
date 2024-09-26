#!/bin/bash

function run_sub2_pipelines() {
    base_path=$1

    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            for sub2dir in ${subdir}/*; do
                if [ -d "$sub2dir" ]; then
                log_filename="$(basename $sub2dir).log"
                script -c "time run_fisheye_pipelines.py -i $sub2dir" "$log_filename"
                echo "Finished processing $sub2dir"
                fi
            done
        fi
    done
}

run_sub2_pipelines /mnt/113-data/samba-share/simulation/train/2x_0920

