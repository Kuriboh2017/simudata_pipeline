import os
import re
import numpy as np


filelist='/mnt/112-data/R23024/data/junwu/data/filelist/synt_new_scene_syn_train_lz4y.csv'

UNIQUE='_unique'

def remove_datetime_string(input_str):
    return re.sub(r'_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}$', '', input_str) + UNIQUE


def get_folder_names(filelist):
    with open(filelist, 'r') as file:
        text = file.read()
    matches = re.findall(r'\b[\w_]+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', text)
    folder_names = np.unique(matches)
    print(f'Find {len(folder_names)} unique matched folder names')
    return folder_names

folder_names = get_folder_names(filelist)


scene_names = np.unique([remove_datetime_string(val) for val in folder_names])
print(f'Find {len(scene_names)} unique scene names')


def count_lines_with_substrings(filename, substrings):
    counts = {substring: 0 for substring in substrings}
    
    with open(filename, 'r') as file:
        for line in file:
            for substring in substrings:
                target = substring.replace(UNIQUE, '_202')
                if target in line:
                    counts[substring] += 1
                    
    return counts

scene_names_count = count_lines_with_substrings(filelist, scene_names)


# 'Abandoned_City_wire': 86240,
# 'Abandoned_City': 1600,
# 'abandonfactory': 500,
# 'angry_mesh': 6002,
# 'Autel_abandonedfactory': 2030,
# 'Autel_DekogonGym_Sunset': 2000,
# 'Autel_Demo_Map': 2827,
# 'Autel_Evening_Highway': 1599,
# 'Autel_Forest_Environment_Set_Map': 2836,
# 'Autel_Forest': 2083,
# 'Autel_Garage_02': 2000,
# 'Autel_Japanese_Street': 2000,
# 'Autel_Winter_Demo_02': 4000,
# 'Brushify_beach': 24490,
# 'dekogongym': 500,
# 'deserttown_wire': 2935,
# 'DesertTown_wire': 40404,
# 'DesertTown': 17600,
# 'deserttown': 2500,
# 'Evening_Highway_wire': 58600,
# 'evening_highway_wire': 6494,
# 'forest_along_road': 10004,
# 'forest_environment_set_no_white_box': 20004,
# 'forest': 9753,
# 'garage': 500,
# 'grass': 615,
# 'highway': 500,
# 'IndustrialCity_wire': 38936,
# 'Japanese_Street_wire': 38929,
# 'japanstreet': 17,
# 'jungle': 1479,
# 'lowpolysnowforest': 6002,
# 'mediter': 654,
# 'mediterraneanisland': 2688,
# 'ModularNeighborhoodPack_wire': 35113,
# 'mountain': 1329,
# 'mountaingrassland': 4038,
# 'ocean': 503,
# 'OldTown_wire': 35115,
# 'OpenLand': 5022,
# 'openlandGrass': 2413,
# 'parking': 1000,
# 'parkingGarage': 3068,
# 'rocky_pine_forest': 3252,
# 'snowtown': 662,
# 'Steampunkenvironment01_wire': 39221,
# 'Trainstation_wire': 30705,
# 'TrainStation_wire': 4837,
# 'tropical_jungle_pack_wire': 28004,
# 'tropical_jungle_pack': 2823,
# 'TropicalOceanTool': 6218,
# 'tropicaloceantool': 866,
# 'UrbanCity_wire': 63030,
# 'winter': 523,
# 'wintertown': 1766,




# Manual organized:
# 'abandond_factory': 2530,
# 'Abandoned_City': 87840,
# 'angry_mesh': 6002,
# 'beach': 24490,
# 'DekogonGym': 2500,
# 'DesertTown': 63439,
# 'Evening_Highway': 66693,
# 'Forest_Environment_Set': 22840,
# 'Forest': 21840,
# 'garage': 2500,
# 'highway': 500,
# 'IndustrialCity': 38936,
# 'Japanese_Street': 40946,
# 'lowpolysnowforest': 6002,
# 'mediterranean_island': 6169,
# 'ModularNeighborhoodPack': 35113,
# 'mountaingrassland': 5367,
# 'ocean': 503,
# 'OldTown': 35115,
# 'openlandGrass': 8050,
# 'parkingGarage': 4068,
# 'rocky_pine_forest': 3252,
# 'snow_town': 662,
# 'Steampunkenvironment01': 39221,
# 'Trainstation': 35542,
# 'tropical_jungle_pack': 32306,
# 'TropicalOceanTool': 7084,
# 'UrbanCity': 63030,
# 'Winter_town': 6289,



folder_names = get_folder_names('/mnt/112-data/R23024/data/junwu/data/filelist/close_to_ground/train_lz4y.csv')
# {'Autel_DekogonGym_Sunset_unique': 2000, 'Autel_Demo_Map_unique': 2827, 'Autel_Evening_Highway_unique': 1599, 'Autel_Forest_Environment_Set_Map_unique': 2836, 'Autel_Forest_unique': 2083, 'Autel_Garage_02_unique': 2000, 'Autel_Japanese_Street_unique': 2000, 'Autel_abandonedfactory_unique': 2030, 'abandonfactory_unique': 500, 'dekogongym_unique': 500, 'deserttown_unique': 2500, 'forest_unique': 9753, 'garage_unique': 500, 'grass_unique': 615, 'highway_unique': 500, 'japanstreet_unique': 17, 'jungle_unique': 1479, 'mediter_unique': 654, 'mediterraneanisland_unique': 2688, 'mountain_unique': 1329, 'mountaingrassland_unique': 4038, 'ocean_unique': 503, 'openlandGrass_unique': 2413, 'parkingGarage_unique': 3068, 'parking_unique': 1000, 'rocky_pine_forest_unique': 3252, 'snowtown_unique': 662, 'tropical_jungle_pack_unique': 2823, 'tropicaloceantool_unique': 866, 'winter_unique': 523, 'wintertown_unique': 1766}

folder_names = get_folder_names('/mnt/112-data/R23024/data/junwu/data/filelist/close_to_ground/test_lz4y.csv')
