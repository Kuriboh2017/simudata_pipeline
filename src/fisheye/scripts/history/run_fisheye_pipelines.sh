#!/bin/bash

function panorama_disp_seg() {
    base_path=$1

    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            for sub2dir in ${subdir}/*; do
                if [ -d "$sub2dir" ]; then
                    echo "Processing $sub2dir"
                    disparity_log_filename="$(basename $sub2dir)_disparity.log"
                    seg_log_filename="$(basename $sub2dir)_seg.log"
                    script -c "time run_panorama_disparity_from_depth.py -i $sub2dir/cube_front/CubeDepth" "$disparity_log_filename"
                    script -c "time run_seg_gray_from_color.py -i $sub2dir/cube_front/CubeSegmentation" "$seg_log_filename"
                    echo "Finished processing $sub2dir"
                    echo ""
                fi
            done
            echo "Finished processing $subdir"
        fi
    done
}

panorama_disp_seg /mnt/113-data/samba-share/simulation/train/2x_tiny



#!/bin/bash

function fisheye_isp_combo_2x() {
    base_path=$1

    for subdir in ${base_path}/*; do
        if [ -d "$subdir" ]; then
            echo "Processing $subdir"
            for sub2dir in ${subdir}/*; do
                if [ -d "$sub2dir" ]; then
                    echo "Processing $sub2dir"
        		    leaf_dir="$(basename $sub2dir)"
                    log_filename="${leaf_dir}.log"
                    script -c "time run_fisheye_isp_combo_2x_by_folder.py -i $sub2dir -o /mnt/113-data/samba-share/simulation/train/2x_tiny_out/${leaf_dir} --remapping-folder /mnt/119-data/S22017/remapping_tables/2x/" "$log_filename"
                    echo "Finished processing $sub2dir"
                    echo ""
                fi
            done
            echo "Finished processing $subdir"
        fi
    done
}

fisheye_isp_combo_2x /mnt/113-data/samba-share/simulation/train/2x_tiny


