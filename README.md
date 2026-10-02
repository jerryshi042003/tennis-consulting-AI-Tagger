# Tennis Shot-Placement Tracking

![Tennis shot-placement tracking demo](./rally_demo_ahmani.gif)

- **What:** Convert match video into player positions, ball locations, speeds, and shot-placement coordinates.
- **Why:** Help coaches move from raw film to visual training decisions without expensive court-installed tracking hardware.

Public computer-vision prototype built with Python. [See the visual-analysis library](https://github.com/jerryshi042003/Tennis-Viz).

## Credits

The detection and tracking core (court detection, ball tracking network, player detection, pose and stroke recognition) is adapted from [avivcaspi/TennisProject](https://github.com/avivcaspi/TennisProject).

My additions are the shot-placement and analysis layer: `analyzing.py`, `player_shot_side_functions.py` and `statistics_module.py`, which turn the tracked positions into per-shot placement coordinates, shot side and match statistics for coaches.
