#!/bin/bash

function filter_unmatched_images() {
    base_path=$1

    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            for sub2dir in ${subdir}/*; do
                if [ -d "$sub2dir" ]; then
                log_filename="$(basename $sub2dir).log"
                script -c "time filter_out_unmatched_images.py -i $sub2dir" "$log_filename"
                echo "Finished processing $sub2dir"
                fi
            done
        fi
    done
}

filter_unmatched_images /mnt/113-data/samba-share/simulation/train/2x_tiny_1018

