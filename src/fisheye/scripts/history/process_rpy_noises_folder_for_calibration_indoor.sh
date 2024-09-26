post_sim_process.py -i height_7m_2023-09-15-15-00-40 -o height_7m
post_sim_process.py -i height_6m_2023-09-15-15-00-08 -o height_6m
post_sim_process.py -i height_5m_2023-09-15-14-59-41 -o height_5m
post_sim_process.py -i height_4m_2023-09-15-14-58-59 -o height_4m
post_sim_process.py -i height_3m_2023-09-15-14-58-40 -o height_3m
post_sim_process.py -i height_2m_2023-09-15-14-58-12 -o height_2m



mkdir process
mv height_7m process/
mv height_6m process/
mv height_5m process/
mv height_4m process/
mv height_3m process/
mv height_2m process/

cd process

process_directory() {
    dir="$1"

    front_rgb_dir="${dir}_out/front/rgb"
    front_depth_dir="${dir}_out/front/depth"
    rear_rgb_dir="${dir}_out/rear/rgb"
    rear_depth_dir="${dir}_out/rear/depth"
    
    mkdir -p "${front_rgb_dir}"
    mkdir -p "${front_depth_dir}"
    mkdir -p "${rear_rgb_dir}"
    mkdir -p "${rear_depth_dir}"
    
    rgb_file=$(ls "$dir/cube_front/CubeScene/" | head -n 1)
    depth_file=$(ls "$dir/cube_front/CubeDepth/" | head -n 1)
    
    cp "$dir/cube_front/CubeScene/${rgb_file}" "${front_rgb_dir}/original_panorama.webp"
    cp "$dir/cube_front/CubeDepth/${depth_file}" "${front_depth_dir}/original_panorama.lz4"
    cp "$dir/cube_rear/CubeScene/${rgb_file}" "${rear_rgb_dir}/original_panorama.webp"
    cp "$dir/cube_rear/CubeDepth/${depth_file}" "${rear_depth_dir}/original_panorama.lz4"
}

dirs=(height_7m height_6m height_5m height_4m height_3m height_2m)

for dir in "${dirs[@]}"; do
    process_directory "$dir"
done

mkdir fisheye_conversion
mv height_7m_out fisheye_conversion/
mv height_6m_out fisheye_conversion/
mv height_5m_out fisheye_conversion/
mv height_4m_out fisheye_conversion/
mv height_3m_out fisheye_conversion/
mv height_2m_out fisheye_conversion/


dirs=(height_7m_out height_6m_out height_5m_out height_4m_out height_3m_out height_2m_out)
rgb_subdir=(front/rgb rear/rgb)
depth_subdir=(front/depth rear/depth)
combined_rgb_dirs=()
combined_depth_dirs=()
for dir in "${dirs[@]}"; do
    for subdir in "${rgb_subdir[@]}"; do
        combined_rgb_dirs+=("$dir/$subdir")
    done
    for subdir in "${depth_subdir[@]}"; do
        combined_depth_dirs+=("$dir/$subdir")
    done
done

fisheye_conversion_rgb() {
    local dir="$1"
    pushd "$dir"
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_noiseless.png -o1 up_noiseless.png --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_roll_0.1.png -o1 up_roll_0.1.png --roll 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_roll_0.2.png -o1 up_roll_0.2.png --roll 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_roll_negative_0.1.png -o1 up_roll_negative_0.1.png --roll -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_roll_negative_0.2.png -o1 up_roll_negative_0.2.png --roll -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_pitch_0.1.png -o1 up_pitch_0.1.png --pitch 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_pitch_0.2.png -o1 up_pitch_0.2.png --pitch 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_pitch_negative_0.1.png -o1 up_pitch_negative_0.1.png --pitch -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_pitch_negative_0.2.png -o1 up_pitch_negative_0.2.png --pitch -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_yaw_0.1.png -o1 up_yaw_0.1.png --yaw 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_yaw_0.2.png -o1 up_yaw_0.2.png --yaw 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_yaw_negative_0.1.png -o1 up_yaw_negative_0.1.png --yaw -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_yaw_negative_0.2.png -o1 up_yaw_negative_0.2.png --yaw -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_rpy_0.1.png -o1 up_rpy_0.1.png --roll 0.1 --pitch 0.1 --yaw 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_rpy_0.2.png -o1 up_rpy_0.2.png --roll 0.2 --pitch 0.2 --yaw 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_rpy_negative_0.1.png -o1 up_rpy_negative_0.1.png  --roll -0.1 --pitch -0.1 --yaw -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.webp -o0 down_rpy_negative_0.2.png -o1 up_rpy_negative_0.2.png  --roll -0.2 --pitch -0.2 --yaw -0.2 --recalculate
    rm remapping_table_panorama2fisheye.npz
    popd
}
for dir in "${combined_rgb_dirs[@]}"; do
    fisheye_conversion_rgb "$dir"
done

fisheye_conversion_depth() {
    local dir="$1"
    pushd "$dir"
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_noiseless.npz --depth -o1 up_noiseless.npz --depth --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_roll_0.1.npz --depth -o1 up_roll_0.1.npz --depth --roll 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_roll_0.2.npz --depth -o1 up_roll_0.2.npz --depth --roll 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_roll_negative_0.1.npz --depth -o1 up_roll_negative_0.1.npz --depth --roll -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_roll_negative_0.2.npz --depth -o1 up_roll_negative_0.2.npz --depth --roll -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_pitch_0.1.npz --depth -o1 up_pitch_0.1.npz --depth --pitch 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_pitch_0.2.npz --depth -o1 up_pitch_0.2.npz --depth --pitch 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_pitch_negative_0.1.npz --depth -o1 up_pitch_negative_0.1.npz --depth --pitch -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_pitch_negative_0.2.npz --depth -o1 up_pitch_negative_0.2.npz --depth --pitch -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_yaw_0.1.npz --depth -o1 up_yaw_0.1.npz --depth --yaw 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_yaw_0.2.npz --depth -o1 up_yaw_0.2.npz --depth --yaw 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_yaw_negative_0.1.npz --depth -o1 up_yaw_negative_0.1.npz --depth --yaw -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_yaw_negative_0.2.npz --depth -o1 up_yaw_negative_0.2.npz --depth --yaw -0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_rpy_0.1.npz --depth -o1 up_rpy_0.1.npz --depth --roll 0.1 --pitch 0.1 --yaw 0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_rpy_0.2.npz --depth -o1 up_rpy_0.2.npz --depth --roll 0.2 --pitch 0.2 --yaw 0.2 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_rpy_negative_0.1.npz --depth -o1 up_rpy_negative_0.1.npz --depth  --roll -0.1 --pitch -0.1 --yaw -0.1 --recalculate
    panorama_to_fisheye.py -i original_panorama.lz4 -o0 down_rpy_negative_0.2.npz --depth -o1 up_rpy_negative_0.2.npz --depth  --roll -0.2 --pitch -0.2 --yaw -0.2 --recalculate
    rm remapping_table_panorama2fisheye.npz
    popd
}
for dir in "${combined_depth_dirs[@]}"; do
    fisheye_conversion_depth "$dir"
done

