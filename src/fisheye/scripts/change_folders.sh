mkdir depth_3pinholes
mkdir disparity_3pinholes
mv Depth_3to1/*_depth.npz depth_3pinholes
mv Depth_3to1/*.npz disparity_3pinholes
mv Depth_3to1/intermediate/fisheye depth_fisheye
rm -rf Depth_3to1

ls disparity_3pinholes | wc -l
ls depth_3pinholes | wc -l
ls depth_fisheye | wc -l


mkdir image_3pinholes
mv Image_3to1/*.webp image_3pinholes
mv Image_3to1/intermediate/fisheye image_fisheye
rm -rf Image_3to1

ls image_3pinholes | wc -l
ls image_fisheye | wc -l

