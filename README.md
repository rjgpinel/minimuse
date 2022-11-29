# RecVis 22/23 - Multimodal - MUjoco Simulation Environments

![](img/minimuse.png)

## Run expert trajectories

Run
```
python -m muse.run --env Push-v0
```
Use `--render` to generate mp4 videos from cameras.

## Installation

### Install MuJoCo

Download mujoco 2.1.0 [binaries](https://mujoco.org/download) and extract them in `~/.mujoco/mujoco210`.<br/>

Then setup the environment variable by adding to your `.bashrc` the following:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

### Conda Environment

Install python dependencies by creating an environment with `anaconda`:
```
conda env create -f environment.yml
```

Then install `minimuse`:
```
pip install -e .
```

## GUI Rendering

To render the scene in a GUI, add to your `.bashrc`:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
To render images from a camera, run `unset LD_PRELOAD`. Currently it is not possible to both have a GUI window and render camera images.


## Dataset collection
To collect a dataset run:

```
python collect.py --output-dir output_dataset/ --episodes 1000 --num-workers 25
```
with `output_dataset/` the directory where the dataset will be stored.


