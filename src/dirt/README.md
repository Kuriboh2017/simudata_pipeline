## Lens dirt effects

### Shader's algorithm
The shader processes the image in two major steps:
1. Apply radial blur to the original image by mixing each pixel with the surrounding pixels.
2. Add a dirt texture overlay to the blurred image.

### QuickStart

After the [installation](../../README.md#quickstart),

Run a cpp sample
```bash
screen_dirt --input-image-path='samples/input/305212385187.png'
```

Run the python code for batch processing
```bash
python3 batchfiles/add_dirt.py -i $PWD/samples/input -o $PWD/samples/output
```

### Cpp optional arguments

#### Arguments for the blur algorithm
- `--blur-intensity`:
A positive value to specify the dirt intensity.

- `--blur-kernel-size`:
A positive value to specify the blur range per pixel.

- `--blur-pseudo-overexposure`:
A positive value to fake the excessive exposure in the blur algorithm.

#### Arguments for the dirt texture
- `--dirt-texture-id`:
The reference dirt texture id in [src/textures folder](src/textures).

- `--dirt-texture-ratio`:
The mixed ratio of the dirt texture.

- `--dirt-texture-offset-x`, `--dirt-texture-offset-y`:
Offset the dirt pixel texture in x/y direction.

- `--dirt-texture-rotation`:
Rotate the dirt texture in radians.

- `--dirt-texture-scale`:
Scale the size of dirt texture.

- `--dirt-texture-red/green/blue-scale`:
Scale the red/green/blue color of dirt texture.

### Python arguments

```text
$ python3 batchfiles/add_dirt.py -h

usage: add_dirt.py [-h] -i INPUT_DIR -o OUTPUT_DIR

Add screen dirt to images.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory of the input images
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory of the output images
```

Inside the python script, cpp arguments are randomly generated in a range, which can be modified in the beginning of `batchfiles/add_dirt.py`.