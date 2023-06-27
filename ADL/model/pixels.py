"""Module for presets for HEALPix pixels."""
default = {
    "val": [9, 38, 41],
    "test": [6],
    "train": [x for x in range(48) if x not in [9, 38, 41, 6]],
}
east_val = [38, 6]
west_val = [9, 41]
east_north = [6]
east_south = [38]
