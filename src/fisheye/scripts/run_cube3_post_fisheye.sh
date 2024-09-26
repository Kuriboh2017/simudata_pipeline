#!/bin/bash

function cube3_post_fisheye_process() {
    base_path=$1

    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            log_filename="$(basename $subdir).log"
            script -c "time cube3_post_fisheye_process.py -i $subdir" "$log_filename"
            echo "Finished processing $subdir"
            echo ""
        fi
    done
}

# cube3_post_fisheye_process /mnt/113-data/samba-share/simulation/train/startfly/15mm_out
# cube3_post_fisheye_process /mnt/113-data/samba-share/simulation/train/startfly/28mm_out

cube3_post_fisheye_process /mnt/113-data/samba-share/simulation/train/2x_tiny_1018_out
