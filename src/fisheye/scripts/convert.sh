

SIM_DATA_PATH=/mnt/113-data/R10198/samba-share/sytn_data_out_indoor/abandonedfactory_0519/abandonedfactory_2023-05-19-17-55-04/

mkdir -p sim_dataset/group1/cam1_0/
mkdir -p sim_dataset/group1/cam1_1/
rsync -avz perception:${SIM_DATA_PATH}/group1/cam1_0/Depth sim_dataset/group1/cam1_0/
rsync -avz perception:${SIM_DATA_PATH}/group1/cam1_0/Image sim_dataset/group1/cam1_0/
rsync -avz perception:${SIM_DATA_PATH}/group1/cam1_0/Segmentation sim_dataset/group1/cam1_0/

rsync -avz perception:${SIM_DATA_PATH}/group1/cam1_1/Depth sim_dataset/group1/cam1_1/
rsync -avz perception:${SIM_DATA_PATH}/group1/cam1_1/Image sim_dataset/group1/cam1_1/

