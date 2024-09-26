## Lens flare effects

### Shader's algorithm
The shader generates the lens flare effect in 3 steps:
1. Draw a bright solid circle as the Sun.
2. Draw the diffraction spikes around the Sun. The spikes are randomly genrated
   using an arctangent function and a noise function.
3. Draw the lens flares one by one repeatedly in the opposite direction of the
   Sun. Each flare's exact location and intensity are randomly generated.

After that, the generated lens flare is added as an overlay on top of the original base image.

### Shader reference
The original shader code to generate lens flare effects:
https://www.shadertoy.com/view/lsBGDK

### QuickStart

After the [installation](../../README.md#quickstart),

Run a cpp sample
```bash
lens_flare --input-image-path='samples/input/305212385187.png'
```

Run the python code for batch processing
```bash
python3 batchfiles/add_flare.py -i $PWD/samples/input -o $PWD/samples/output
```

### Cpp optional arguments

- `--sun-size`:
Size of the generated Sun. Default: 0.15

- `--sun-location-x`, `--sun-location-y`:
The location of the generated Sun in the image between -1 and 1. The top left of the image is
(-1,-1), and the bottom right of the image is (1,1).

- `--diffraction-spikes-intensity`:
Overall intensity of the diffraction spikes around the Sun.

- `--lens-flares-count`:
The maximum number of generated lens flares.

- `--lens-flares-intensity`:
Overall intensity of the lens flares.

- `--lens-flares-seed `:
Random seed of the lens flares.

### Python arguments

```text
$ python3 batchfiles/add_flare.py -h

usage: add_flare.py [-h] -i INPUT_DIR -o OUTPUT_DIR

Add screen flare to images.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory of the input images
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory of the output images
```

Inside the python script, cpp arguments are randomly generated in a range, which can be modified in the beginning of `batchfiles/add_flare.py`.