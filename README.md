# Tennis Shot-Placement Tracking

![Tennis shot-placement tracking demo](./rally_demo_ahmani.gif)

- **What:** Convert match video into player positions, ball locations, speeds, and shot-placement coordinates.
- **Why:** Help coaches move from raw film to visual training decisions without expensive court-installed tracking hardware.

Public computer-vision prototype built with Python. [See the visual-analysis library](https://github.com/jerryshi042003/Tennis-Viz).

## Credits

The detection and tracking core (court detection, ball tracking network, player detection, pose and stroke recognition) is adapted from [avivcaspi/TennisProject](https://github.com/avivcaspi/TennisProject).

The shot-placement and analysis layer includes `analyzing.py`, `player_shot_side_functions.py` and `statistics_module.py`, which process tracked positions into placement coordinates, shot side and match statistics for coaches. `player_shot_side_functions.py` credits Vivian Yee in its source header.

## Running the research code

Install the Python dependencies with `python -m pip install -r requirements.txt`.
The scripts also require video files, model checkpoints under `saved states/`,
and intermediate arrays under `output/`; these are not included in this repository.
Paths in the script entry points must match those local inputs.

The committed GIF and video show an earlier output. They do not establish that
a fresh clone can reproduce it or that every predicted coordinate is accurate.
This repository does not include an end-to-end evaluation or a trained model release.
