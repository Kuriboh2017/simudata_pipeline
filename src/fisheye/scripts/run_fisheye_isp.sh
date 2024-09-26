#!/bin/bash


run_each_subfolder() {
    base_path=$1
    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            log_filename="$(basename $subdir).log"
            script -c "time run_fisheye_isp_combo_2x_by_folder_2fisheye.py -i $subdir -o ${base_path}_out --remapping-folder /mnt/119-data/S22017/remapping_tables/2x/" "$log_filename"
            echo "Finished processing $subdir"
            echo ""
        fi
    done
}

run_each_subfolder "/mnt/113-data/samba-share/simulation/train/startfly/15mm"

run_each_subfolder "/mnt/113-data/samba-share/simulation/train/startfly/28mm"

