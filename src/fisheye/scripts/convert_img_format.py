import os
from PIL import Image
from pathlib import Path

def convert_png_to_jpeg(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'Segmentation' not in root and file.lower().endswith('.png'):
                png_path = Path(root) / file
                jpeg_path = png_path.with_suffix('.jpg')
                # Convert PNG to JPEG
                with Image.open(png_path) as img:
                    rgb_im = img.convert('RGB')
                    rgb_im.save(jpeg_path, 'JPEG')
                # Delete the original PNG file
                os.remove(png_path)

# Replace 'your_directory_path' with the path to your target directory
convert_png_to_jpeg('.')
