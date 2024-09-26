## Lens strong flare effects

### Background
There are two types of lens flare effects:
- Lens flare
- Strong lens flare

While the original [lens flares](../flare/README.md) effects aim to provide "usual" lens flares to augment the image data for perception model training. In that case, the perception algorithm was supposed to work as usual. On the contrary, this **strong lens flares** effects aim to "crash" the perception algorithm. The perception algorithm is expected to recognize that the lens flares effects are too strong and then alert other autonomy modules.

### Shader's algorithm
The shader generates the lens flare effect in 3 steps:
1. Draw a bright solid circle as the Sun.
2. Draw the diffraction spikes around the Sun.
3. Draw the lens flares components.

After that, the generated lens flare is added as an overlay on top of the original base image.

### Shader reference
The original shader code to generate lens flare effects:
https://www.shadertoy.com/view/4sX3Rs

### QuickStart

After the [installation](../../README.md#quickstart),

Run a cpp sample
```bash
strong_flare --input-image-path='samples/input/305212385187.png'
```

Run the python code for batch processing
```bash
python3 batchfiles/add_strong_flare.py -i $PWD/samples/input -o $PWD/samples/output
```

### Cpp optional arguments

- `--sun-size`:
Size of the generated Sun.

- `--sun-location-x`, `--sun-location-y`:
The location of the generated Sun in the image between -1 and 1. The top left of the image is
(-1,-1), and the bottom right of the image is (1,1).

- `--diffraction-spikes-intensity`:
Overall intensity of the diffraction spikes around the Sun.

- `--lens-flares-intensity`:
Overall intensity of the lens flares.

- `--lens-flares-red/green/blue-scale`:
Scale the red/green/blue color of lens flares.


### Python arguments

```text
$ python3 batchfiles/add_strong_flare.py -h

usage: add_strong_flare.py [-h] -i INPUT_DIR -o OUTPUT_DIR

Add screen flare to images.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory of the input images
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory of the output images
```

Inside the python script, cpp arguments are randomly generated in a range, which can be modified in the beginning of `batchfiles/add_strong_flare.py`.