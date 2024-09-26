
parent_dir="/mnt/113-data/samba-share/3cubes_renamed"
for dir in $(find "$parent_dir" -maxdepth 1 -mindepth 1 -type d); do
    echo "Processing directory: $dir"
    run_disparity_from_depth.py -i "$dir"/group0/cam0_0/Depth -o "$dir"/group0/cam0_0/Disparity --baseline 0.105
    run_disparity_from_depth.py -i "$dir"/group1/cam1_1/Depth -o "$dir"/group1/cam1_1/Disparity --baseline 0.090
done


parent_dir="/mnt/113-data/samba-share/3cubes_renamed"
for dir in $(find "$parent_dir" -maxdepth 1 -mindepth 1 -type d); do
    echo "Processing directory: $dir"
    ./post_fisheye_rot_180.py -i "$dir"/group1/
done



parent_dir="/mnt/113-data/samba-share/evaluation_fisheye_renamed"
for dir in $(find "$parent_dir" -maxdepth 1 -mindepth 1 -type d); do
    echo "Processing directory: $dir"
    ls "$dir"/group0/cam0_0/Image_3to1/*.webp | wc -l
    ls "$dir"/group0/cam0_1/Image_3to1/*.webp | wc -l
    ls "$dir"/group1/cam1_0/Image_3to1/*.webp | wc -l
    ls "$dir"/group1/cam1_1/Image_3to1/*.webp | wc -l
done


dir=/mnt/113-data/samba-share/evaluation_fisheye/Autel_Japanese_Street_2023-08-09-19-08-11_out




parent_dir="/mnt/119-data/S22017/fisheye/full_resolution_20230815"
for dir in $(find "$parent_dir" -maxdepth 1 -mindepth 1 -type d); do
    echo "Processing directory: $dir"
    script -c "time ./post_fisheye_process.py -i $dir" "$dir".log
done



