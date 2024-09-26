#!/usr/bin/env python3
from pathlib import Path
import argparse
import distutils.spawn
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from subprocess import run
blur_only_folders = ['/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-14-16-32',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-14-37-09',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-14-57-43',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-15-18-20',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-15-38-53',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-15-59-46',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-16-20-45',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-16-41-52',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-17-02-44',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-17-23-22',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-17-44-25',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_mixed_conner_case_0526/mediterraneanisland_2023-05-26-18-06-03',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-19-49-44',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-20-12-54',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-20-37-14',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-20-59-04',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-21-20-27',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-21-41-42',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-22-03-10',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-22-24-21',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-22-45-13',
                     '/mnt/113-data/R10198/samba-share/normal/mediterraneanisland_wire_false_alarm_0525/mediterraneanisland_2023-05-24-23-06-03',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-09-08-13',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-10-01-06',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-10-55-39',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-11-49-21',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-12-43-00',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-13-34-18',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-14-25-12',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-15-18-23',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-11-16-11-24',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/eval_verify/Evening_Highway_wire_2022-11-14-20-31-28',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221109/DesertTown_wire_2022-11-05-17-08-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221109/DesertTown_wire_2022-11-06-17-17-30',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221110_new/OldTown_wire_2022-11-10-17-33-57',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221110_new/OldTown_wire_2022-11-10-19-57-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221110_new/OldTown_wire_2022-11-10-21-46-05',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-05-18-40-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-05-19-59-02',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-05-21-18-44',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-05-22-38-21',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-05-23-57-27',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-06-01-16-48',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/Abandoned_City_wire_2022-11-06-02-36-06',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221112/TrainStation_wire_2022-11-11-09-56-03',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221113/ModularNeighborhoodPack_wire_2022-11-09-15-38-58',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221113/ModularNeighborhoodPack_wire_2022-11-09-23-56-03',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221113/ModularNeighborhoodPack_wire_2022-11-10-02-00-01',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221113/ModularNeighborhoodPack_wire_2022-11-10-08-56-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221113/ModularNeighborhoodPack_wire_2022-11-10-10-53-54',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-09-21-33-43',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-10-15-59-42',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-10-18-42-32',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-10-21-07-00',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-10-23-50-33',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-11-02-33-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-11-05-19-12',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/DesertTown_wire_2022-11-11-08-06-54',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-17-41-29',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-18-32-19',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-19-20-17',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-20-06-41',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-20-54-47',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221114/OldTown_wire_2022-11-10-21-41-54',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/ModularNeighborhoodPack_wire_2022-11-13-02-50-29',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/ModularNeighborhoodPack_wire_2022-11-13-05-00-39',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/ModularNeighborhoodPack_wire_2022-11-13-07-17-55',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/ModularNeighborhoodPack_wire_2022-11-13-09-31-24',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-11-13-38-15',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-11-16-46-25',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-11-19-49-19',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-11-23-06-30',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-12-02-26-31',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-12-05-40-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221115/UrbanCity_wire_2022-11-12-08-54-23',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-11-10-41',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-12-12-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-13-12-01',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-14-17-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-15-20-44',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-16-23-53',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-17-25-01',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-18-27-40',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-12-19-30-56',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/IndustrialCity_wire_2022-11-14-21-54-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-02-02-49',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-02-50-42',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-03-40-26',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-04-29-46',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-05-19-31',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-06-08-46',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-06-58-34',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-07-48-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-13-08-38-16',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221116/Japanese_Street_wire_2022-11-14-23-13-17',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/Trainstation_wire_2022-11-16-10-49-40',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/Trainstation_wire_2022-11-16-12-19-04',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/Trainstation_wire_2022-11-16-15-01-06',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/Trainstation_wire_2022-11-16-18-51-00',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/Trainstation_wire_2022-11-16-21-21-05',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/UrbanCity_wire_2022-11-14-04-35-22',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/UrbanCity_wire_2022-11-14-08-58-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/UrbanCity_wire_2022-11-14-12-39-36',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221117/UrbanCity_wire_2022-11-14-16-11-21',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-16-22-39-20',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-16-23-24-30',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-00-10-42',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-00-56-50',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-01-42-28',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-09-03-42',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-10-27-40',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-11-33-39',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-12-09-11',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-12-40-03',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-13-24-47',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-14-09-17',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Abandoned_City_wire_2022-11-17-14-55-02',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-10-53-35',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-11-50-21',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-12-45-40',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-13-57-01',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221118/Trainstation_wire_2022-11-16-14-50-42',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Evening_Highway_wire_2022-11-18-15-29-15',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Evening_Highway_wire_2022-11-18-16-16-55',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Evening_Highway_wire_2022-11-18-17-04-47',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Evening_Highway_wire_2022-11-18-17-51-59',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Evening_Highway_wire_2022-11-18-18-39-14',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Steampunkenvironment01_wire_2022-11-16-09-52-29',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Steampunkenvironment01_wire_2022-11-16-13-46-48',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Steampunkenvironment01_wire_2022-11-16-17-20-39',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Steampunkenvironment01_wire_2022-11-16-20-52-47',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221119/Steampunkenvironment01_wire_2022-11-17-00-24-07',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/evening_highway_wire_2022-11-18-15-52-47',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/evening_highway_wire_2022-11-18-17-37-30',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/Steampunkenvironment01_wire_2022-11-15-09-00-41',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/Steampunkenvironment01_wire_2022-11-15-12-13-48',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/Steampunkenvironment01_wire_2022-11-15-16-02-58',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/Steampunkenvironment01_wire_2022-11-15-22-02-11',
                     '/mnt/cephfs-partitions/perception/New_Simulation_tools_origin/wire/training_large_new_wire/20221121/Steampunkenvironment01_wire_2022-11-16-03-00-12',]

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)
kRootPath = '/mnt/115-rbd01/fisheye_dataset/train/data'


def _process_folder(origin_folder_path, args):
    origin_path = Path(origin_folder_path)
    src_dir = origin_path / 'group1' / 'cam1_0' / 'Segmentation' / 'Graymap'
    dst_dir = Path(kRootPath) / origin_path.name / 'group1' / \
        'cam1_0' / 'Segmentation' / 'Graymap_3to1'

    run_panorama_to_3pinholes_exe = distutils.spawn.find_executable(
        "run_panorama_to_3pinholes.py")
    assert run_panorama_to_3pinholes_exe, 'Error: python executable `run_panorama_to_3pinholes.py` is not available!'
    cmds = [
        run_panorama_to_3pinholes_exe,
        f'--input-dir={src_dir}',
        f'--output-dir={dst_dir}',
        '--delete-odd-rows',
        '--segmentation-graymap'
    ]
    _logger.info(f'Executing command:\n{" ".join(cmds)}')
    if not args.dry_run:
        run(cmds)


def _main(args):
    with Pool(3) as p:
        p.map(partial(_process_folder, args=args), blur_only_folders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run pipeline: panorama -> 3pinholes')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='whether to do a dry run')
    args = parser.parse_args()
    _main(args)
