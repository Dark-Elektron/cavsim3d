"""Default visualization and solver settings."""

WEBGUI_SETTINGS = {
    'Objects': {'Clipping Plane': True, 'Vectors': True},
    'Colormap': {'ncolors': 125},
    'Clipping': {'enable': True, 'x': 1, 'y': 0, 'z': 0},
    'Vectors': {'grid_size': 200},
    'Complex': {'animate': True}
}

DEFAULT_SOLVER_SETTINGS = {
    'order': 3,
    'maxh': 0.05,
    'curve_order': 3,
    'pinvit_maxit': 20,
    'pinvit_num': 10,
}