""" Outline import from SVG files, using svgelements for parsing.

The outline is the first shape in the file (a path, polygon, rect, ...), which must be a single closed contour. Absolute
units of the root's width and height (mm, in, ...) give the drawing's size; without them, one user unit is one mm.
"""

import math

import numpy as np
import svgelements as se


def _flatten(segment, tolerance):
    """ Points along a curve or arc within tolerance of it, without its start point """
    n = 1
    while True:
        points = np.array(segment.npoint(np.linspace(0, 1, 2*n + 1)))
        chord_midpoints = (points[:-2:2] + points[2::2]) / 2
        if np.hypot(*(points[1::2] - chord_midpoints).T).max() <= tolerance:
            return points[2::2].tolist()
        n *= 2


def read_outline(filename, tolerance=0.01):
    """ Points of the outline in mm """
    svg = se.SVG.parse(str(filename), ppi=25.4) # (one px is one mm)
    shape = next((el for el in svg.elements() if isinstance(el, se.Shape)), None)
    if shape is None:
        raise ValueError(f'{filename} contains no path')
    points = []
    for segment in abs(se.Path(shape)): # (abs applies the transforms)
        if isinstance(segment, se.Move):
            if points:
                raise ValueError(f'The outline in {filename} has more than one contour. Join it into one closed path.')
            points.append(segment.end)
        elif isinstance(segment, se.Linear):
            points.append(segment.end)
        else:
            points += _flatten(segment, tolerance)
    points = [(float(x), float(y)) for x, y in points]
    if len(points) < 4 or math.dist(points[0], points[-1]) > tolerance:
        raise ValueError(f'The outline in {filename} is not closed')
    return points[:-1]
