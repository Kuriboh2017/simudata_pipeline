3pinholes_to_panorama.py -i 3pinholes_origin.png -o panorama.png
3pinholes_to_panorama.py -i 3pinholes_origin.png --float32 -o panorama.npz
panorama_to_3pinholes.py -i panorama.npz -o0 mapback.npz --float32
panorama_to_3pinholes.py -i panorama.npz -o0 roll_0.05.npz --float32 --roll 0.05 --recalculate
panorama_to_3pinholes.py -i panorama.npz -o0 pitch_0.05.npz --float32 --pitch 0.05 --recalculate
panorama_to_3pinholes.py -i panorama.npz -o0 yaw_0.05.npz --float32 --yaw 0.05 --recalculate
panorama_to_3pinholes.py -i panorama.npz -o0 rpy_0.05.npz --float32 --roll 0.05 --pitch 0.05 --yaw 0.05 --recalculate

rm panorama_3pinholes_1.png
rm panorama_3pinholes_1.npz

check_fisheye_mapback.py -b 3pinholes_origin.png -e mapback.npz --error-visual-range 0.2
check_fisheye_mapback.py -b 3pinholes_origin.png -e roll_0.05.npz --error-visual-range 0.5
check_fisheye_mapback.py -b 3pinholes_origin.png -e pitch_0.05.npz --error-visual-range 1.25
check_fisheye_mapback.py -b 3pinholes_origin.png -e yaw_0.05.npz --error-visual-range 1.25
check_fisheye_mapback.py -b 3pinholes_origin.png -e rpy_0.05.npz --error-visual-range 2.25

rm remapping_table*.npz
mv *_error.npz ../error_data/

mv roll_0.05_error_0_color.png rotate_around_x_0.05_error_0_color.png
mv roll_0.05_error_0_gray_scale_50.png rotate_around_x_0.05_error_0_gray_scale_50.png
mv roll_0.05_error_1_color.png rotate_around_x_0.05_error_1_color.png
mv roll_0.05_error_1_gray_scale_50.png rotate_around_x_0.05_error_1_gray_scale_50.png
mv roll_0.05_error_2_color.png rotate_around_x_0.05_error_2_color.png
mv roll_0.05_error_2_gray_scale_50.png rotate_around_x_0.05_error_2_gray_scale_50.png

mv pitch_0.05_error_0_color.png rotate_around_y_0.05_error_0_color.png
mv pitch_0.05_error_0_gray_scale_50.png rotate_around_y_0.05_error_0_gray_scale_50.png
mv pitch_0.05_error_1_color.png rotate_around_y_0.05_error_1_color.png
mv pitch_0.05_error_1_gray_scale_50.png rotate_around_y_0.05_error_1_gray_scale_50.png
mv pitch_0.05_error_2_color.png rotate_around_y_0.05_error_2_color.png
mv pitch_0.05_error_2_gray_scale_50.png rotate_around_y_0.05_error_2_gray_scale_50.png

mv yaw_0.05_error_0_color.png rotate_around_z_0.05_error_0_color.png
mv yaw_0.05_error_0_gray_scale_50.png rotate_around_z_0.05_error_0_gray_scale_50.png
mv yaw_0.05_error_1_color.png rotate_around_z_0.05_error_1_color.png
mv yaw_0.05_error_1_gray_scale_50.png rotate_around_z_0.05_error_1_gray_scale_50.png
mv yaw_0.05_error_2_color.png rotate_around_z_0.05_error_2_color.png
mv yaw_0.05_error_2_gray_scale_50.png rotate_around_z_0.05_error_2_gray_scale_50.png

mv rpy_0.05_error_0_color.png rotate_around_xyz_0.05_error_0_color.png
mv rpy_0.05_error_0_gray_scale_50.png rotate_around_xyz_0.05_error_0_gray_scale_50.png
mv rpy_0.05_error_1_color.png rotate_around_xyz_0.05_error_1_color.png
mv rpy_0.05_error_1_gray_scale_50.png rotate_around_xyz_0.05_error_1_gray_scale_50.png
mv rpy_0.05_error_2_color.png rotate_around_xyz_0.05_error_2_color.png
mv rpy_0.05_error_2_gray_scale_50.png rotate_around_xyz_0.05_error_2_gray_scale_50.png

mv *_error_*.png ../error_map/
