
LEFT_RGB_DIR=/mnt/119-data/S22017/fisheye/isp_20230824_rename/mediterraneanisland_2023-05-24-21-31-08_renamed/group0/cam0_0/Image
RIGHT_RGB_DIR=/mnt/119-data/S22017/fisheye/isp_20230824_rename/mediterraneanisland_2023-05-24-21-31-08_renamed/group0/cam0_1/Image

scp perception:${LEFT_RGB_DIR}/1684928970815_dark_isp.webp left_dark_isp.webp
scp perception:${LEFT_RGB_DIR}/1684928970815_demosaicing.webp left_demosaicing.webp
scp perception:${LEFT_RGB_DIR}/1684928970815_fixed_isp.webp left_fixed_isp.webp
scp perception:${LEFT_RGB_DIR}/1684928970815_random_isp.webp left_random_isp.webp
scp perception:${LEFT_RGB_DIR}/1684928970815_vignette.webp left_vignette.webp
scp perception:${LEFT_RGB_DIR}/1684928970815.webp left.webp


scp perception:${RIGHT_RGB_DIR}/1684928970815_dark_isp.webp right_dark_isp.webp
scp perception:${RIGHT_RGB_DIR}/1684928970815_demosaicing.webp right_demosaicing.webp
scp perception:${RIGHT_RGB_DIR}/1684928970815_fixed_isp.webp right_fixed_isp.webp
scp perception:${RIGHT_RGB_DIR}/1684928970815_random_isp.webp right_random_isp.webp
scp perception:${RIGHT_RGB_DIR}/1684928970815_vignette.webp right_vignette.webp
scp perception:${RIGHT_RGB_DIR}/1684928970815.webp right.webp



REAL_LEFT_DIR=/mnt/113-data/samba-share/3d-reconstruction/preprocess_data/2023_06_29/Autonomy405/sensors_data_2023.06.27-10.22.35_fisheye/group0/cam0_0/rect123_high
REAL_RIGHT_DIR=/mnt/113-data/samba-share/3d-reconstruction/preprocess_data/2023_06_29/Autonomy405/sensors_data_2023.06.27-10.22.35_fisheye/group0/cam0_1/rect123_high

scp perception:${REAL_LEFT_DIR}/100309478939.png left_0.png
scp perception:${REAL_LEFT_DIR}/100810293149.png left_1.png
scp perception:${REAL_LEFT_DIR}/101311031326.png left_2.png
scp perception:${REAL_LEFT_DIR}/101811792732.png left_3.png
scp perception:${REAL_LEFT_DIR}/102312455285.png left_4.png
scp perception:${REAL_LEFT_DIR}/102813250813.png left_5.png
scp perception:${REAL_LEFT_DIR}/103313948091.png left_6.png
scp perception:${REAL_LEFT_DIR}/103824856161.png left_7.png
scp perception:${REAL_LEFT_DIR}/104315402190.png left_8.png
scp perception:${REAL_LEFT_DIR}/104824817619.png left_9.png

scp perception:${REAL_RIGHT_DIR}/100309478939.png right_0.png
scp perception:${REAL_RIGHT_DIR}/100810293149.png right_1.png
scp perception:${REAL_RIGHT_DIR}/101311031326.png right_2.png
scp perception:${REAL_RIGHT_DIR}/101811792732.png right_3.png
scp perception:${REAL_RIGHT_DIR}/102312455285.png right_4.png
scp perception:${REAL_RIGHT_DIR}/102813250813.png right_5.png
scp perception:${REAL_RIGHT_DIR}/103313948091.png right_6.png
scp perception:${REAL_RIGHT_DIR}/103824856161.png right_7.png
scp perception:${REAL_RIGHT_DIR}/104315402190.png right_8.png
scp perception:${REAL_RIGHT_DIR}/104824817619.png right_9.png


REAL_LEFT_DIR=/mnt/113-data/samba-share/3d-reconstruction/preprocess_data/2023_07_03/Autonomy499/sensors_data_2023.07.03-15.34.53_fisheye/group0/cam0_0/rect123_high/
REAL_RIGHT_DIR=/mnt/113-data/samba-share/3d-reconstruction/preprocess_data/2023_07_03/Autonomy499/sensors_data_2023.07.03-15.34.53_fisheye/group0/cam0_1/rect123_high/

scp perception:${REAL_LEFT_DIR}/100309721713.png left_0.png
scp perception:${REAL_LEFT_DIR}/100810482234.png left_1.png
scp perception:${REAL_LEFT_DIR}/101311142474.png left_2.png
scp perception:${REAL_LEFT_DIR}/101811902995.png left_3.png
scp perception:${REAL_LEFT_DIR}/102312569401.png left_4.png
scp perception:${REAL_LEFT_DIR}/102813329922.png left_5.png
scp perception:${REAL_LEFT_DIR}/103313995662.png left_6.png
scp perception:${REAL_LEFT_DIR}/103814756183.png left_7.png
scp perception:${REAL_LEFT_DIR}/104315422382.png left_8.png
scp perception:${REAL_LEFT_DIR}/104816182902.png left_9.png

scp perception:${REAL_RIGHT_DIR}/100309721713.png right_0.png
scp perception:${REAL_RIGHT_DIR}/100810482234.png right_1.png
scp perception:${REAL_RIGHT_DIR}/101311142474.png right_2.png
scp perception:${REAL_RIGHT_DIR}/101811902995.png right_3.png
scp perception:${REAL_RIGHT_DIR}/102312569401.png right_4.png
scp perception:${REAL_RIGHT_DIR}/102813329922.png right_5.png
scp perception:${REAL_RIGHT_DIR}/103313995662.png right_6.png
scp perception:${REAL_RIGHT_DIR}/103814756183.png right_7.png
scp perception:${REAL_RIGHT_DIR}/104315422382.png right_8.png
scp perception:${REAL_RIGHT_DIR}/104816182902.png right_9.png


# check_img_similarity.py -l left.webp -r right.webp 
# left and right similarity: 0.7511219713656387

add_sharpen.py -i left.webp 
add_sharpen.py -i right.webp
check_img_similarity.py -l left_sharpened.webp -r right_sharpened.webp 


add_isp.py -i left.webp 
add_isp.py -i right.webp
check_img_similarity.py -l left_isp_fixed.webp -r right_isp_fixed.webp 

add_demosaicing.py -i left.webp 
add_demosaicing.py -i right.webp
check_img_similarity.py -l left_demosaiced.webp -r right_demosaiced.webp 


add_vignette.py -i left.webp 
add_vignette.py -i right.webp
check_img_similarity.py -l left_vignette.webp -r right_vignette.webp 

add_clache.py -i left.webp 
add_clache.py -i right.webp
check_img_similarity.py -l left_clache.webp -r right_clache.webp 


check_img_similarity.py -l left.webp -r right.webp 

add_demosaicing.py -i left.webp -o l1.webp
add_demosaicing.py -i right.webp -o r1.webp
check_img_similarity.py -l l1.webp -r r1.webp 

add_sharpen.py -i l1.webp -o l2.webp
add_sharpen.py -i r1.webp -o r2.webp
check_img_similarity.py -l l2.webp -r r2.webp 

add_vignette.py -i l2.webp -o l3.webp
add_vignette.py -i r2.webp -o r3.webp
check_img_similarity.py -l l3.webp -r r3.webp 

add_isp.py -i l3.webp -o l4.webp
add_isp.py -i r3.webp -o r4.webp
check_img_similarity.py -l l4.webp -r r4.webp 






check_img_similarity.py -l left.webp -r right.webp 

add_vignette.py -i left.webp -o l1.webp
add_vignette.py -i right.webp -o r1.webp
check_img_similarity.py -l l1.webp -r r1.webp 

add_demosaicing.py -i l1.webp -o l2.webp
add_demosaicing.py -i r1.webp -o r2.webp
check_img_similarity.py -l l2.webp -r r2.webp 

add_isp.py -i l2.webp -o l3.webp
add_isp.py -i r2.webp -o r3.webp
check_img_similarity.py -l l3.webp -r r3.webp 

add_clache.py -i l3.webp -o l4.webp
add_clache.py -i r3.webp -o r4.webp
check_img_similarity.py -l l4.webp -r r4.webp 

add_sharpen.py -i l4.webp -o l5.webp
add_sharpen.py -i r4.webp -o r5.webp
check_img_similarity.py -l l5.webp -r r5.webp 








LEFT_DIR=./sample_isp_out/sample_src/group1/cam1_0/Image_3to1
RIGHT_DIR=./sample_isp_out/sample_src/group1/cam1_1/Image_3to1
SUFFIX0=down.webp
SUFFIX1=down_0_vignette_1_demosaicing_2_fixed_isp_3_clache_4_sharpen.webp
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1684928970815_${SUFFIX0}" -r "${RIGHT_DIR}/1684928970815_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1684928970815_${SUFFIX1}" -r "${RIGHT_DIR}/1684928970815_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1685080554145_${SUFFIX0}" -r "${RIGHT_DIR}/1685080554145_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1685080554145_${SUFFIX1}" -r "${RIGHT_DIR}/1685080554145_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683279199705_${SUFFIX0}" -r "${RIGHT_DIR}/1683279199705_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683279199705_${SUFFIX1}" -r "${RIGHT_DIR}/1683279199705_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1682568006500_${SUFFIX0}" -r "${RIGHT_DIR}/1682568006500_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1682568006500_${SUFFIX1}" -r "${RIGHT_DIR}/1682568006500_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683180209861_${SUFFIX0}" -r "${RIGHT_DIR}/1683180209861_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683180209861_${SUFFIX1}" -r "${RIGHT_DIR}/1683180209861_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683547511925_${SUFFIX0}" -r "${RIGHT_DIR}/1683547511925_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683547511925_${SUFFIX1}" -r "${RIGHT_DIR}/1683547511925_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683367022135_${SUFFIX0}" -r "${RIGHT_DIR}/1683367022135_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683367022135_${SUFFIX1}" -r "${RIGHT_DIR}/1683367022135_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1682346168902_${SUFFIX0}" -r "${RIGHT_DIR}/1682346168902_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1682346168902_${SUFFIX1}" -r "${RIGHT_DIR}/1682346168902_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683180369861_${SUFFIX0}" -r "${RIGHT_DIR}/1683180369861_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683180369861_${SUFFIX1}" -r "${RIGHT_DIR}/1683180369861_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1682567926500_${SUFFIX0}" -r "${RIGHT_DIR}/1682567926500_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1682567926500_${SUFFIX1}" -r "${RIGHT_DIR}/1682567926500_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1682346088902_${SUFFIX0}" -r "${RIGHT_DIR}/1682346088902_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1682346088902_${SUFFIX1}" -r "${RIGHT_DIR}/1682346088902_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683285281728_${SUFFIX0}" -r "${RIGHT_DIR}/1683285281728_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683285281728_${SUFFIX1}" -r "${RIGHT_DIR}/1683285281728_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683535411977_${SUFFIX0}" -r "${RIGHT_DIR}/1683535411977_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683535411977_${SUFFIX1}" -r "${RIGHT_DIR}/1683535411977_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683167646135_${SUFFIX0}" -r "${RIGHT_DIR}/1683167646135_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683167646135_${SUFFIX1}" -r "${RIGHT_DIR}/1683167646135_${SUFFIX1}"
echo " "
check_img_similarity.py -l "${LEFT_DIR}/1683290263160_${SUFFIX0}" -r "${RIGHT_DIR}/1683290263160_${SUFFIX0}"
check_img_similarity.py -l "${LEFT_DIR}/1683290263160_${SUFFIX1}" -r "${RIGHT_DIR}/1683290263160_${SUFFIX1}"
