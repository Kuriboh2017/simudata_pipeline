#!/bin/bash

script -c 'time run_fisheye_low_altitude.py -i lowfly_out/snowtown_2023-07-10-13-11-46  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' snowtown_2023-07-10-13-11-46.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/wintertown_2023-07-10-13-30-05  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' wintertown_2023-07-10-13-30-05.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/mountain_2023-07-10-12-57-59  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' mountain_2023-07-10-12-57-59.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/jungle_2023-07-10-14-25-49  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' jungle_2023-07-10-14-25-49.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/parking_2023-07-10-12-18-26  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' parking_2023-07-10-12-18-26.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/forest_2023-07-07-14-10-17  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' forest_2023-07-07-14-10-17.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/abandonfactory_2023-07-07-14-57-21  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' abandonfactory_2023-07-07-14-57-21.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/mediter_2023-07-10-11-06-17  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' mediter_2023-07-10-11-06-17.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/garage_2023-07-07-20-01-03  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' garage_2023-07-07-20-01-03.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/grass_2023-07-10-09-22-22  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' grass_2023-07-10-09-22-22.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/deserttown_2023-07-07-18-06-00  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' deserttown_2023-07-07-18-06-00.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/ocean_2023-07-10-11-43-07  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' ocean_2023-07-10-11-43-07.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/winter_2023-07-07-17-05-30  -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' winter_2023-07-07-17-05-30.log

script -c 'time run_fisheye_low_altitude.py -i lowfly_out/dekogongym_2023-07-07-17-38-27 -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' dekogongym_2023-07-07-17-38-27.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/japanstreet_2023-07-11-09-09-59 -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' japanstreet_2023-07-11-09-09-59.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_out/japanstreet_2023-07-11-09-20-23 -o low_altitude/ --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' japanstreet_2023-07-11-09-20-23.log

# 0713
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Winter_Demo_02_2023-07-13-12-57-33_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Winter_Demo_02_2023-07-13-12-57-33_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Japanese_Street_2023-07-13-20-35-04_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Japanese_Street_2023-07-13-20-35-04_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Garage_02_2023-07-13-20-01-10_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Garage_02_2023-07-13-20-01-10_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Forest_Environment_Set_Map_2023-07-13-18-31-13_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Forest_Environment_Set_Map_2023-07-13-18-31-13_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Forest_2023-07-13-19-16-47_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Forest_2023-07-13-19-16-47_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Evening_Highway_2023-07-13-17-29-56_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Evening_Highway_2023-07-13-17-29-56_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Deserttown_2023-07-13-16-48-18_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Deserttown_2023-07-13-16-48-18_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_Demo_Map_2023-07-13-21-16-05_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_Demo_Map_2023-07-13-21-16-05_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_DekogonGym_Sunset_2023-07-13-16-06-05_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_DekogonGym_Sunset_2023-07-13-16-06-05_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_abandonedfactory_2023-07-13-14-55-03_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_abandonedfactory_2023-07-13-14-55-03_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_abandonedfactory_2023-07-13-14-14-04_out -o low_altitude_0713/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_abandonedfactory_2023-07-13-14-14-04_out.log

# 0714
script -c 'time run_fisheye_low_altitude.py -i lowfly_0714/deserttown_2023-07-14-10-49-52_out -o low_altitude_0714/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' deserttown_2023-07-14-10-49-52_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0714/mediterraneanisland_2023-07-14-11-35-33_out -o low_altitude_0714/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' mediterraneanisland_2023-07-14-11-35-33_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0714/mountaingrassland_2023-07-14-09-58-06_out -o low_altitude_0714/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' mountaingrassland_2023-07-14-09-58-06_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0714/mountaingrassland_2023-07-14-12-26-40_out -o low_altitude_0714/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' mountaingrassland_2023-07-14-12-26-40_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0714/openlandGrass_2023-07-14-13-48-40_out -o low_altitude_0714/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' openlandGrass_2023-07-14-13-48-40_out.log

# 0715
script -c 'time run_fisheye_low_altitude.py -i lowfly_0715/parkingGarage_2023-07-14-14-23-32_out -o low_altitude_0715/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' parkingGarage_2023-07-14-14-23-32_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0715/rocky_pine_forest_2023-07-14-15-30-26_out -o low_altitude_0715/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' rocky_pine_forest_2023-07-14-15-30-26_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0715/tropical_jungle_pack_2023-07-14-16-47-17_out -o low_altitude_0715/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' tropical_jungle_pack_2023-07-14-16-47-17_out.log
script -c 'time run_fisheye_low_altitude.py -i lowfly_0715/tropicaloceantool_2023-07-14-16-23-38_out -o low_altitude_0715/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' tropicaloceantool_2023-07-14-16-23-38_out.log

# 0718

script -c 'time run_fisheye_low_altitude.py -i lowfly_0713/Autel_abandonedfactory_2023-07-13-14-14-04_out -o low_altitude_0715/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' Autel_abandonedfactory_2023-07-13-14-14-04_fix_to_0715.log

script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-11-49-50_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-11-49-50_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-12-25-00_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-12-25-00_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-13-02-35_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-13-02-35_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-13-37-39_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-13-37-39_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-14-13-22_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-14-13-22_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-14-48-42_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-14-48-42_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-15-26-15_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-15-26-15_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-15-58-32_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-15-58-32_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-16-39-32_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-16-39-32_out.log
script -c 'time run_fisheye_low_altitude.py -i cloud_0718/cloud_2023-07-18-17-16-04_out -o high_altitude_cloud_0718/  --remapping-folder /mnt/115-rbd01/fisheye_dataset/train/low_altitude/remapping_table' cloud_2023-07-18-17-16-04_out.log
