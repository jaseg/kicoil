# Copyright 2025 Jan Sebastian Götte <code@jaseg.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
import logging
import itertools
from dataclasses import dataclass, field
import sys
from math import *

import numpy as np
import shapely

from gerbonara.cad.kicad.footprints import Footprint
from gerbonara.cad.kicad.primitives import Zone, Hatch, ZoneKeepout, ZonePolygon, XYCoord

from . import kicad, skeletonator


mu_0 = 1.25663706127e-06 # from scipy.constants

DEFAULT_CIRCLE_SEGMENTS = 360 # points per revolution of circular coils in the footprint
SPIRAL_CHORD_ERROR = 0.0005 # mm, chord error of the circular spiral center lines used for measurements
PIECE_STEP = 0.025 # revolutions, length of the center line pieces that the turn spacing is measured on
VIA_OFFSET_STEPS = 10 # steps of the automatic via offset search
KEEPOUT_MITER_LIMIT = 2.0 # keepout corners are cut off at this multiple of the margin from the outline's corner


def point_line_distance(p, l1, l2):
    x0, y0 = p
    x1, y1 = l1
    x2, y2 = l2
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return ((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / sqrt((x2-x1)**2 + (y2-y1)**2)


# https://en.wikipedia.org/wiki/Farey_sequence#Next_term
def farey_sequence(n: int, descending: bool = False) -> None:
    """Print the n'th Farey sequence. Allow for either ascending or descending."""
    a, b, c, d = 0, 1, 1, n
    if descending:
        a, c = 1, n - 1
    yield a, b

    while c <= n and not descending or a > 0 and descending:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        yield a, b


def divisors(n, max_b=10):
    for a, b in farey_sequence(n):
        if a != 0 and a == n and b < max_b:
            yield b
        if b != 1 and b == n and a < max_b:
            yield a


def arc_approximate(points, trace_width, layer, tolerance=0.02, level=0):
    """ Approximate spiral arm using circular arcs. This results in a smoother output using less segments than if we
    approximate the arc using straight line segments.

    The input to this function is a list of points of a straight line segment approximation, and it returns a list of
    gerbonara arc objects approximating the input. """
    if len(points) < 3:
        raise ValueError('Need at least three points to approximate')

    i_mid = len(points)//2

    x0, y0 = points[0]
    x1, y1 = points[i_mid]
    x2, y2 = points[-1]

    if len(points) < 5:
        yield kicad.make_arc(x0, y0, x2, y2, x1, y1, trace_width, layer)
        return

    # https://stackoverflow.com/questions/56224824/how-do-i-find-the-circumcenter-of-the-triangle-using-python-without-external-lib
    d = 2 * (x0 * (y2 - y1) + x2 * (y1 - y0) + x1 * (y0 - y2))
    if abs(d) > 1e-12:
        cx = ((x0 * x0 + y0 * y0) * (y2 - y1) + (x2 * x2 + y2 * y2) * (y1 - y0) + (x1 * x1 + y1 * y1) * (y0 - y2)) / d
        cy = ((x0 * x0 + y0 * y0) * (x1 - x2) + (x2 * x2 + y2 * y2) * (x0 - x1) + (x1 * x1 + y1 * y1) * (x2 - x0)) / d
        r = dist((cx, cy), (x1, y1))
    if abs(d) <= 1e-12 or any(abs(dist((px, py), (cx, cy)) - r) > tolerance for px, py in points):
        # (Collinear points are split until the pieces are short enough to be emitted as nearly straight arcs)
        yield from arc_approximate(points[:i_mid+1], trace_width, layer, tolerance, level+1)
        yield from arc_approximate(points[i_mid:], trace_width, layer, tolerance, level+1)

    else:
        if point_line_distance((x1, y1), (x0, y0), (x2, y2)) > 0:
            yield kicad.make_arc(x0, y0, x2, y2, x1, y1, trace_width, layer)
        else:
            yield kicad.make_arc(x2, y2, x0, y0, x1, y1, trace_width, layer)


def center_line_pieces(arms, step):
    """ Cut the center lines of the arms where their angle passes a multiple of `step` revolutions. `arms` holds a
    (points, angles in revolutions) tuple per arm. Returns the pieces as LineStrings, the arm of each piece, and the
    lowest and highest angle of each piece. """
    pieces, arm, lo, hi = [], [], [], []
    for k, (points, angles) in enumerate(arms):
        points, angles = np.array(points), np.array(angles)
        if angles[-1] < angles[0]:
            points, angles = points[::-1], angles[::-1]
        # Insert the cut points into the center line, then split it there
        cuts = np.arange(floor(angles[0] / step) + 1, ceil(angles[-1] / step)) * step
        at = np.searchsorted(angles, cuts, side="right")
        cut_points = np.column_stack([np.interp(cuts, angles, points[:, 0]), np.interp(cuts, angles, points[:, 1])])
        points, angles = np.insert(points, at, cut_points, axis=0), np.insert(angles, at, cuts)
        ends = [0, *(at + np.arange(len(at))), len(points) - 1]
        for a, b in itertools.pairwise(ends):
            pieces.append(shapely.LineString(points[a:b+1]))
            arm.append(k)
            lo.append(angles[a])
            hi.append(angles[b])
    return np.array(pieces), np.array(arm), np.array(lo), np.array(hi)


def min_turn_spacing(pieces, search_radius, min_turn_separation=0.5):
    """ Smallest distance between center line pieces of different turns, and a point where it occurs. Returns
    (search_radius, None) if there are none closer than search_radius. """
    geoms, arm, lo, hi = pieces

    # All pairs of pieces closer than the search radius, each pair once
    i, j = shapely.STRtree(geoms).query(geoms, predicate='dwithin', distance=search_radius)
    keep = i < j
    i, j = i[keep], j[keep]

    # Pieces of different arms are always on different turns. Pieces of the same arm are on different turns when they
    # are at least min_turn_separation revolutions apart.
    angle_gap = np.maximum(lo[j] - hi[i], lo[i] - hi[j])
    keep = (arm[i] != arm[j]) | (angle_gap >= min_turn_separation)
    i, j = i[keep], j[keep]
    if not len(i):
        return search_radius, None

    d = shapely.distance(geoms[i], geoms[j])
    k = np.argmin(d)
    if d[k] >= search_radius:
        return search_radius, None
    x, y = shapely.shortest_line(geoms[i[k]], geoms[j[k]]).coords[0]
    return float(d[k]), (x, y)


class Shape:
    # How far the inner via ring may be moved inwards from the innermost turn
    max_via_offset = inf


@dataclass
class CircleShape(Shape):
    outer_diameter: float
    inner_diameter: float


    def __post_init__(self):
        self.outer_radius = self.outer_diameter / 2
        self.inner_radius = self.inner_diameter / 2
        # The inner via ring must not pass the center (it would end up on the opposite side)
        self.max_via_offset = self.inner_radius


    @property
    def slug(self):
        return f'circle_{self.outer_diameter:.2f}x{self.inner_diameter:.2f}'


    @property
    def desc(self):
        return f'{self.outer_diameter:.2f} mm OD, {self.inner_diameter:.2f} mm ID circular'


    def compute_spiral(self, a1, a2, fn=None, debug=False):
        """ Points of the spiral from angle a1 on the outer radius to angle a2 on the inner radius, with fn points per
        revolution, and the angle of each point in revolutions. """
        r1, r2 = self.outer_radius, self.inner_radius
        if fn is None:
            fn = ceil(pi / acos(1 - SPIRAL_CHORD_ERROR/self.outer_radius))
        fn = max(1, ceil(fn * abs(a2-a1)/(2*pi)))
        a = a1 + (a2 - a1) * np.arange(fn + 1) / fn
        r = r1 + (r2 - r1) * np.arange(fn + 1) / fn
        return list(zip((np.cos(a) * r).tolist(), (np.sin(a) * r).tolist())), (a / (2*pi)).tolist()


    def project_point(self, r, a):
        return cos(a) * r, sin(a) * r


    def offset_exterior(self, margin):
        r = self.outer_radius + margin
        tol = 0.05 # mm
        n = ceil(pi / acos(1 - tol/r))
        return [(r*cos(a*2*pi/n), r*sin(a*2*pi/n)) for a in range(n)]


class OffsetShape(Shape):
    def __post_init__(self):
        self.sk = skeletonator.Skeletonator(self.polygon)
        self.polygon = list(self.sk.outline) # normalized: counter-clockwise, without repeated points
        if self.annular_width >= self.sk.max_depth:
            raise ValueError(f'Annular width ({self.annular_width:.2f}) is too large. Must be less than {self.sk.max_depth:.2f}')
        if self.annular_width <= 0:
            raise ValueError('Annular width must be positive')
        self.sk.set_angle_reference(self.annular_width)
        self.max_via_offset = self.sk.max_depth - self.annular_width
        self.outer_radius = self.sk.radius
        self.inner_radius = self.sk.radius - self.annular_width

    @property
    def slug(self):
        return f'polygonal_{len(self.polygon)}pt_a{self.annular_width:.2f}mm'


    @property
    def desc(self):
        return f'polygonal (n={len(self.polygon)} point, {self.annular_width:.2f} mm annular width)'


    def compute_spiral(self, a1, a2, fn=None, debug=False):
        # Skeletonator uses a t coordinate from 0 - 1 per revolution instead of a radian angle.
        return self.sk.spiral(a1/(2*pi), a2/(2*pi), 0, self.annular_width, debug=debug)


    def project_point(self, r, a):
        # Skeletonator uses a t coordinate from 0 - 1 per revolution instead of a radian angle.
        return self.sk.project_point(a/(2*pi) % 1, self.sk.radius - r)


    def offset_exterior(self, margin):
        """ Outline offset outwards by margin (inwards for a negative margin), e.g. for the keepout zone. """
        if margin > 0:
            offset = shapely.Polygon(self.polygon).buffer(margin, join_style='mitre', mitre_limit=KEEPOUT_MITER_LIMIT)
            return list(shapely.geometry.polygon.orient(offset).exterior.coords)[:-1]
        if margin == 0:
            return list(self.polygon)
        return self.sk.spiral(0, 1, -margin, -margin)[0][:-1]


@dataclass
class TrapezoidShape(OffsetShape):
    width: float
    height: float
    offset: float
    annular_width: float
    arc_tolerance: float = 0.05 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        w, h, d = self.width, self.height, self.offset
        self.polygon = [(w/2-d, -h/2), (w/2, h/2), (-w/2, h/2), (-w/2+d, -h/2)]
        super().__post_init__()

    @property
    def slug(self):
        return f'trapezoid_{self.width:.2f}-2*{self.offset:.2f}x{self.height:.2f}'

    @property
    def desc(self):
        return f'{self.width:.2f} x {self.height:.2f} mm, {self.offset:.2f} mm offset isosceles trapezoidal'


@dataclass
class RectangleShape(OffsetShape):
    width: float
    height: float
    annular_width: float
    arc_tolerance: float = 0.05 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        w, h = self.width, self.height
        self.polygon = [(w/2, -h/2), (w/2, h/2), (-w/2, h/2), (-w/2, -h/2)]
        super().__post_init__()

    @property
    def slug(self):
        return f'rectangle_{self.width:.2f}x{self.height:.2f}'

    @property
    def desc(self):
        return f'{self.width:.2f} x {self.height:.2f} mm rectangle'

@dataclass
class SectorShape(OffsetShape):
    inner_diameter: float
    outer_diameter: float
    angle: float
    annular_width: float
    arc_tolerance: float = 0.05 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        # Careful: The inner/outer radius properties are relative to the polygon center and are very different from these!
        r1, r2 = self.inner_diameter / 2, self.outer_diameter/2
        n1 = ceil(pi / acos(1 - self.arc_tolerance/self.inner_diameter) * self.angle / (2*pi))
        n2 = ceil(pi / acos(1 - self.arc_tolerance/self.outer_diameter) * self.angle / (2*pi))
        # center on y axis
        pt = lambda r, a: (r*sin(a), r*cos(a))
        self.polygon  = [pt(r2, self.angle/2 - i/n2 * self.angle) for i in range(n2+1)]
        self.polygon += [pt(r1, i/n1 * self.angle - self.angle/2) for i in range(n1+1)]
        super().__post_init__()

    @property
    def slug(self):
        return f'sector_{self.outer_diameter:.2f}x{self.inner_diameter:.2f}_{degrees(self.angle):.0f}deg'

    @property
    def desc(self):
        return f'{self.outer_diameter:.2f} x {self.inner_diameter:.2f} mm {degrees(self.angle):.0f} deg sector'


@dataclass
class StarShape(OffsetShape):
    inner_diameter: float
    outer_diameter: float
    annular_width: float
    points: int = 5
    arc_tolerance: float = 0.05 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        # center on y axis
        pt = lambda r, a: (-r*sin(a), r*cos(a))
        circle = lambda r, n, phase: [pt(r, (i + phase)*2*pi/n) for i in range(n)]
        self.polygon  = [x for pair in zip(circle(self.outer_diameter/2, self.points, 0), circle(self.inner_diameter/2, self.points, 0.5)) for x in pair]
        super().__post_init__()

    @property
    def slug(self):
        return f'star_{self.outer_diameter:.2f}x{self.inner_diameter:.2f}'

    @property
    def desc(self):
        purpose = ', for demonic purposes' if self.points == 5 else ''
        return f'{self.outer_diameter:.2f} x {self.inner_diameter:.2f} mm star shape{purpose}'


@dataclass
class RegularPolygonShape(OffsetShape):
    diameter: float
    annular_width: float
    corners: int = 8
    arc_tolerance: float = 0.05 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        # center on y axis
        pt = lambda r, a: (-r*sin(a), r*cos(a))
        circle = lambda r, n, phase: [pt(r, (i + phase)*2*pi/n) for i in range(n)]
        self.polygon  = list(circle(self.diameter/2, self.corners, 0))
        super().__post_init__()

    @property
    def slug(self):
        return f'regular_{self.corners}gon_{self.diameter:.2f}'

    @property
    def desc(self):
        return f'{self.diameter:.2f} mm diameter {self.corners} corner regular polygon'


@dataclass
class SVGShape(OffsetShape):
    """ Outline from an SVG file, see svg_import.read_outline. """
    filename: str
    annular_width: float
    arc_tolerance: float = 0.01 # mm
    polygon: list = field(init=False)

    def __post_init__(self):
        from . import svg_import
        coords = svg_import.read_outline(self.filename, self.arc_tolerance)

        min_x, min_y, max_x, max_y = shapely.Polygon(coords).bounds
        if max_x < 0 or max_y < 0 or min_x > 0 or min_y > 0:
            # (0, 0) is not within the polygon's axis-aligned bounding box, recenter.
            (ox, oy), = shapely.Polygon(coords).centroid.coords
            warnings.warn(f'Polygon looks not centered, bounds are ({min_x:.2f}, {min_y:.2f}), ({max_x:.2f}, {max_y:.2f}). Aligning (0, 0) with polygon centroid at ({ox:.2f}, {oy:.2f})')
            coords = [(x-ox, y-oy) for x, y in coords]

        self.polygon  = coords
        super().__post_init__()

    @property
    def slug(self):
        return f'svg_{len(self.polygon)}n'

    @property
    def desc(self):
        return f'{len(self.polygon)} node imported SVG shape'


@dataclass
class PlanarInductor():
    shape: Shape
    turns: int
    twists: int
    trace_width: float = None
    clearance: float = None
    layers: int = 2
    via_diameter: float = 0.6
    via_drill: float = None
    via_offset: float = None
    stagger_inner_vias: bool = False
    stagger_outer_vias: bool = False
    keepout_zone: bool = True
    keepout_margin: float = 0.0
    copper_thickness: float = 0.035
    layer_pair: str = 'F.Cu,B.Cu'
    clockwise: bool = False
    approximate_arcs: bool = True
    arm_phase: float = 0.0

    def __post_init__(self):
        self.logger = logging.getLogger('kicoil')

        if self.turns < 1:
            raise ValueError(f'Error: PlanarInductor.turns must be 1 or more')

        if self.twists < 1:
            raise ValueError(f'Error: PlanarInductor.twists must be 1 or more')

        self.turns_per_layer = self.turns/self.layers
        self.sector_angle = 2*pi / self.twists

        if self.clockwise:
            self.sector_angle *= -1

        self.sweeping_angle = self.sector_angle * self.turns_per_layer
        self.spiral_pitch = (self.shape.outer_radius-self.shape.inner_radius) / self.turns_per_layer
        self.R = None # will be calculated during render

        if gcd(self.twists, self.turns) != 1:
            raise ValueError(f'For the geometry to work out, the twists parameter must be co-prime to turns, i.e. the two must have 1 as their greatest common divisor. You can print valid values for twists by running the kicoil CLI with --show-twists [turns number].\n\n'
                                       f'Right now, both are divisible by {gcd(self.twists, self.turns)}.\n'
                                       f'Valid twist counts for n={self.turns} turns are: {list(divisors(self.turns, max(self.turns, 25)))}\n'
                                       f'Valid turn counts for k={self.twists} twists are: {list(divisors(self.twists, max(self.twists, 25)))}')

        if (self.stagger_inner_vias or self.stagger_outer_vias) and self.twists%2 != 0:
            raise ValueError('For via staggering to work, twists must be even and turns must be odd.')

        # Neighboring turns come closer than the nominal pitch where they run at an angle to the outline, and at corners,
        # where their bends do not line up exactly. Instead of estimating this, measure the spiral center lines.
        pp, where = self.measure_turn_spacing()
        self.projected_spiral_pitch = pp
        self.logger.info(f'Nominal spiral pitch {self.spiral_pitch:.3f} mm, smallest distance between turns {pp:.3f} mm.')
        pitch_hint = ''
        if where is not None and round(pp, 3) < round(self.spiral_pitch, 3):
            pitch_hint = (f' The projected pitch is the smallest distance between the center lines of neighboring '
                          f'turns, {pp:.3f} mm (nominal pitch {self.spiral_pitch:.3f} mm), measured near '
                          f'({where[0]:.3f}, {where[1]:.3f}).')

        if self.trace_width is None and self.clearance is None:
            self.clearance = 0.15
            warnings.warn(f'Warning: Neither trace width nor clearance given. Defaulting to {self.clearance:.3f} mm clearance.')

        if self.trace_width is None:
            if round(self.clearance, 3) >= round(pp, 3):
                raise ValueError(f'Given clearance of {self.clearance:.3f} mm is larger than the projected spiral pitch '
                                 f'of {pp:.3f} mm. Reduce clearance or increase the size of the coil.' + pitch_hint)
            self.trace_width = pp - self.clearance
            self.logger.info(f'Calculated trace width for {self.clearance:.3f} mm clearance is {self.trace_width:.3f} mm.')

        elif round(self.trace_width, 3) >= round(pp, 3):
            raise ValueError(f'Given trace width of {self.trace_width:.3f} mm is larger than the projected spiral '
                             f'pitch of {pp:.3f} mm. Reduce trace width or increase the size of the coil.' + pitch_hint)

        elif self.clearance is None:
            self.clearance = pp - self.trace_width
            self.logger.info(f'Calculated clearance for {self.trace_width:.3f} mm trace width is {self.clearance:.3f} mm.')

        elif round(pp - self.trace_width, 3) < round(self.clearance, 3):
            raise ValueError(f'Actual clearance for {self.trace_width:.3f} mm trace width is {pp - self.trace_width:.3f} '
                             f'mm, which is lower than the given clearance of {self.clearance:.3f} mm. Reduce trace '
                             f'width to at most {floor((pp - self.clearance) * 1000) / 1000:.3f} mm, reduce the '
                             f'clearance or increase the size of the coil.' + pitch_hint)

        if round(self.via_diameter, 2) < round(self.trace_width, 2):
            self.logger.warning(f'Clipping via diameter from {self.via_diameter:.2f} mm to trace width of {self.trace_width:.2f} mm.')
            self.via_diameter = self.trace_width

        if self.via_drill is None:
            self.via_drill = max(self.via_diameter / 2, self.via_diameter - 1.2)
            self.logger.warning(f'No via drill given, defaulting to {self.via_drill:.2f} mm based on via diameter.')

        self._place_vias()

        t, _, b = self.layer_pair.partition(',')
        self.layer_pair = (t.strip(), b.strip())

        # For fill factor & inductance formulas, See https://coil32.net/pcb-coil.html for details
        d_avg = (2*self.shape.outer_radius + self.shape.inner_radius)/2
        phi = (2*self.shape.outer_radius - 2*self.shape.inner_radius) / (2*self.shape.outer_radius + 2*self.shape.inner_radius)
        c1, c2, c3, c4 = 1.00, 2.46, 0.00, 0.20 # FIXME for other shapes
        self.L = mu_0 * self.turns**2 * d_avg*1e3 * c1 / 2 * (log(c2/phi) + c3*phi + c4*phi**2)
        self.logger.info(f'Outer diameter: {2*self.shape.outer_radius:g} mm')
        self.logger.info(f'Average diameter: {d_avg:g} mm')
        self.logger.info(f'Inner diameter: {2*self.shape.inner_radius:g} mm')
        self.logger.info(f'Fill factor: {phi:g}')
        self.logger.info(f'Approximate inductance: {self.L:g} µH')

        # One arm runs from the outside to the inside of a layer
        start_angle, fold_angle, _end_angle = self.arm_angles(0)
        points, _angles = self.shape.compute_spiral(start_angle, fold_angle)
        self.track_length = sum(dist(p, q) for p, q in itertools.pairwise(points))*self.twists*self.layers
        self.logger.info(f'Approximate track length: {self.track_length:.2f} mm')

        A = self.copper_thickness/1e3 * self.trace_width/1e3 # trace cross-section area
        rho = 1.68e-8 # specific resistivity of copper
        self.R = self.track_length/1e3 * rho / A
        self.logger.info(f'Approximate resistance: {self.R:g} Ω')

    def _place_vias(self):
        """ Choose the via offset (if not given) and check that the vias do not overlap """
        # The inner via ring must not go past the center of a circle or the collapse depth of a polygon outline. A
        # staggered inner ring reaches three times the via offset inwards.
        max_offset = self.shape.max_via_offset / (3 if self.stagger_inner_vias else 1)
        need = self.via_diameter + self.clearance
        if self.via_offset is None:
            # Vias wider than the trace can come too close to a neighboring turn that runs at an angle: move them
            # inwards. Where the inner contour narrows, that can put several of them on top of each other: pull them
            # back until they clear.
            self.via_offset = min(max(0, (self.via_diameter - self.trace_width)/2), max_offset)
            for _ in range(VIA_OFFSET_STEPS):
                offset, gap = self.via_offset, self.via_track_clearance()[0]
                if gap >= self.clearance:
                    break
                self.via_offset = min(max_offset, offset + self.clearance - gap + 1e-4)
                if self.via_track_clearance()[0] <= gap:
                    self.via_offset = offset
                    break
            for offset in np.linspace(self.via_offset, 0, VIA_OFFSET_STEPS).tolist():
                self.via_offset = offset
                if self.via_ring_spacing()[0] >= need:
                    break
            self.logger.info(f'Autocalculated via offset {self.via_offset:.2f} mm')
        if isclose(self.via_offset, 0, abs_tol=1e-6):
            self.via_offset = 0

        self.inner_via_ring_radius = self.shape.inner_radius - self.via_offset
        self.outer_via_ring_radius = self.shape.outer_radius + self.via_offset
        self.logger.info(f'Inner via ring @r={self.inner_via_ring_radius:.2f} mm (from {self.shape.inner_radius:.2f} mm)')
        self.logger.info(f'Outer via ring @r={self.outer_via_ring_radius:.2f} mm (from {self.shape.outer_radius:.2f} mm)')

        closest, pair = self.via_ring_spacing()
        if closest < need:
            (x1, y1), (x2, y2) = pair
            message = (f'Overlapping vias: the vias at ({x1:.3f}, {y1:.3f}) and ({x2:.3f}, {y2:.3f}) are {closest:.3f} mm '
                       f'apart, but need {need:.3f} mm (via diameter plus clearance). Reduce the annular width, the via '
                       f'diameter, the via offset or the number of twists.')
            if closest < self.via_diameter:
                raise ValueError(message)
            warnings.warn(message)

    def arm_angles(self, i):
        """ Start, fold and end angle of the i'th arm pair. The arm on the first layer runs inwards from the start to the
        fold angle, the arm on the second layer runs outwards from the fold to the end angle. """
        start_angle = i*self.sector_angle + radians(self.arm_phase)
        fold_angle = start_angle + self.sweeping_angle
        end_angle = fold_angle + self.sweeping_angle
        return start_angle, fold_angle, end_angle

    def measure_turn_spacing(self):
        """ Smallest center line distance between neighboring turns on the same layer, at most the nominal pitch, and
        where it is (None if nothing is closer than the nominal pitch). """
        arms = [[], []]
        for i in range(self.twists):
            start_angle, fold_angle, end_angle = self.arm_angles(i)
            arms[0].append(self.shape.compute_spiral(start_angle, fold_angle))
            arms[1].append(self.shape.compute_spiral(end_angle, fold_angle))
        self._pieces = [center_line_pieces(layer_arms, PIECE_STEP) for layer_arms in arms[:self.layers]]
        results = [min_turn_spacing(pieces, self.spiral_pitch * (1 + 1e-9)) for pieces in self._pieces]
        best, where = min(results, key=lambda r: r[0])
        return (self.spiral_pitch, None) if where is None else (min(best, self.spiral_pitch), where)

    def via_positions(self):
        """ Positions of the inner and outer via rings for the current via offset. The outer ring has no via at the
        start of the first arm, where the pads are. """
        inner, outer = [], []
        for i in range(self.twists):
            start_angle, fold_angle, _end_angle = self.arm_angles(i)
            stagger = 2*self.via_offset if i%2 != 0 else 0
            r = self.shape.inner_radius - self.via_offset - (stagger if self.stagger_inner_vias else 0)
            inner.append(self.shape.project_point(r, fold_angle))
            r = self.shape.outer_radius + self.via_offset + (stagger if self.stagger_outer_vias else 0)
            outer.append(self.shape.project_point(r, start_angle))
        return inner, outer[1:]

    def via_ring_spacing(self):
        """ Smallest center distance between any two vias for the current via offset, and the two via positions """
        inner, outer = self.via_positions()
        return min(((dist(p, q), (p, q)) for p, q in itertools.combinations(inner + outer, 2)),
                   default=(inf, None), key=lambda e: e[0])

    def via_track_clearance(self):
        """ Smallest edge-to-edge distance between a via and a spiral arm, other than the half revolution of an arm that
        leads into the via, for the current via offset. Returns (gap, via position). """
        inner, outer = self.via_positions()
        vias = [] # (position, [(layer, arm, angle in revolutions where the arm connects to the via), ...])
        for i in range(self.twists):
            start_angle, fold_angle, _end_angle = self.arm_angles(i)
            vias.append((inner[i], [(0, i, fold_angle/(2*pi)), (1, i, fold_angle/(2*pi))]))
            if i > 0:
                j = (i - self.turns) % self.twists
                vias.append((outer[i-1], [(0, i, start_angle/(2*pi)), (1, j, self.arm_angles(j)[2]/(2*pi))]))
        best, where = inf, None
        for v, connections in vias:
            for layer, (geoms, arm, lo, hi) in enumerate(self._pieces):
                # Leave out the half revolution of each arm that leads into the via
                leads_in = np.zeros(len(geoms), dtype=bool)
                for conn_layer, conn_arm, t in connections:
                    if conn_layer == layer:
                        leads_in |= (arm == conn_arm) & (lo > t - 0.5) & (hi < t + 0.5)
                if not leads_in.all() and (d := shapely.distance(shapely.Point(v), geoms[~leads_in]).min()) < best:
                    best, where = float(d), v
        return best - self.via_diameter/2 - self.trace_width/2, where

    @property
    def default_footprint_name(self):
        return f'planar-coil-{self.shape.slug}-n{self.turns}-k{self.twists}'

    def render_footprint(self, name=None, circle_segments=DEFAULT_CIRCLE_SEGMENTS, geometry_debug_file=None):
        """ Render the coil as a KiCad footprint. Circles are drawn with circle_segments points per revolution. """
        if name is None:
            name = self.default_footprint_name

        from . import __version__
        footprint = Footprint(
                name=name,
                generator=kicad.Atom('kicoil'),
                generator_version=__version__,
                layer='F.Cu',
                descr=f"{self.turns} turn {self.shape.desc} twisted coil footprint, inductance approximately {self.L:.6f} µH. Generated by kicoil, version {__version__}.",
                clearance=self.clearance,
                zone_connect=0)

        # Fit the arcs as closely as circle_segments points per revolution approximate the outer circle
        arc_tolerance = self.shape.outer_radius * (1 - cos(pi / circle_segments))

        def add_track(points, layer):
            if self.approximate_arcs and isinstance(self.shape, CircleShape) and len(points) >= 3:
                footprint.arcs.extend(arc_approximate(points, self.trace_width, layer, arc_tolerance))
            else:
                footprint.lines.extend(kicad.make_line(*p1, *p2, self.trace_width, layer)
                                       for p1, p2 in zip(points, points[1:]) if p1 != p2)

        def add_line(p1, p2, layer):
            # Skip zero-length connections, e.g. when the via offset is zero
            if dist(p1, p2) > 1e-9:
                footprint.lines.append(kicad.make_line(*p1, *p2, self.trace_width, layer))

        arms_layers = [[], []]
        inner_vias, outer_vias = self.via_positions()
        debug = geometry_debug_file is not None
        for i in range(self.twists):
            start_angle, fold_angle, end_angle = self.arm_angles(i)

            # Handle the spiral arm
            points_layer0, _ = self.shape.compute_spiral(start_angle, fold_angle, fn=circle_segments, debug=debug)
            xn, yn = points_layer0[-1]
            add_track(points_layer0, self.layer_pair[0])

            if self.layers > 1:
                # Handle the returning arm on the bottom layer
                points_layer1, _ = self.shape.compute_spiral(end_angle, fold_angle, fn=circle_segments, debug=debug)
                points_layer1 = points_layer1[::-1]
                add_track(points_layer1, self.layer_pair[1])

            else:
                # Add a straight connecting segment connecting the inner point to the outside of the spiral.
                xq, yq = self.shape.project_point(self.shape.outer_radius, fold_angle)
                points_layer1 = [(xn, yn), (xq, yq)]
                add_line((xn, yn), (xq, yq), self.layer_pair[1])

            arms_layers[0].append(points_layer0)
            arms_layers[1].append(points_layer1)

        for i in range(self.twists):
            # Inner via ring
            points_layer0 = arms_layers[0][i]
            points_layer1 = arms_layers[1][i]
            xv, yv = inner_vias[i]
            add_line(points_layer0[-1], (xv, yv), self.layer_pair[0])
            add_line((xv, yv), points_layer1[0], self.layer_pair[1])
            footprint.pads.append(kicad.make_via(xv, yv,
                                                 self.via_diameter, self.via_drill, self.clearance,
                                                 self.layer_pair))

            # Outer via ring. At the start of the coil we place the pads instead.
            points_layer0 = arms_layers[0][i]
            points_layer1 = arms_layers[1][(i - self.turns) % self.twists]
            if i > 0:
                xv, yv = outer_vias[i-1]
                add_line(points_layer0[0], (xv, yv), self.layer_pair[0])
                add_line(points_layer1[-1], (xv, yv), self.layer_pair[1])
                footprint.pads.append(kicad.make_via(xv, yv,
                                                     self.via_diameter, self.via_drill, self.clearance,
                                                     self.layer_pair))

            else:
                # Place the pads at the ends of the spiral arms on the outer radius
                px, py = points_layer0[0]
                footprint.pads.append(kicad.make_pad(1, [self.layer_pair[0]], px, py, self.trace_width, self.clearance))
                qx, qy = points_layer1[-1]
                footprint.pads.append(kicad.make_pad(2, [self.layer_pair[1]], qx, qy, self.trace_width, self.clearance))

        if self.keepout_zone:
            pts = self.shape.offset_exterior(self.keepout_margin)
            footprint.zones.append(Zone(layers=['*.Cu'],
                hatch=Hatch(),
                filled_areas_thickness=False,
                keepout=ZoneKeepout(copperpour_allowed=False),
                polygon=ZonePolygon(pts=[XYCoord(x=x, y=y) for x, y in pts])))

        if geometry_debug_file:
            self.shape.sk.dump_to_pdf(geometry_debug_file)
        return footprint
