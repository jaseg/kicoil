""" Geometric checks of rendered coil footprints.

The rendered tracks, arcs and vias must form a single conductor from pad 1 to pad 2. Parts of the conductor that lie on
different turns must keep the clearance to each other, and so must the vias. The spiral tracks must stay within the
annulus between the outline and the central hole.

Two points of the conductor are on different turns when the conductor winds by at least half a revolution around the
center of the hole between them.
"""

import itertools
import math

import networkx as nx
import numpy as np
import pytest
import shapely
from gerbonara.cad.kicad.footprints import Atom

from kicoil.geometry import (PlanarInductor, CircleShape, RectangleShape, RegularPolygonShape, StarShape, SectorShape,
                             TrapezoidShape, SVGShape)

TOLERANCE = 1e-3 # mm, kicoil rounds trace width and clearance to µm
SEGMENT_LENGTH = 0.05 # mm, tracks are cut into pieces of at most this length for the clearance checks
ANNULUS_TOLERANCE = 0.05 # mm, chord error of the circle outlines from CircleShape.offset_exterior
ARC_POINTS = 32 # points per footprint arc
CLEARANCE = 0.15 # mm

L_SHAPE_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="40mm" height="30mm" viewBox="0 0 40 30">
  <path d="M 0 0 H 40 V 12 H 18 V 30 H 0 Z"/>
</svg>'''

ROUNDED_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="40mm" height="30mm" viewBox="0 0 40 30">
  <path d="M 5 0 H 35 A 5 5 0 0 1 40 5 V 25 C 40 30 30 30 20 22 C 10 30 0 30 0 25 V 5 A 5 5 0 0 1 5 0 Z"/>
</svg>'''


def svg_shape(tmp_path, svg, annular_width):
    path = tmp_path / 'outline.svg'
    path.write_text(svg)
    return SVGShape(str(path), annular_width)


def coil(shape, **kwargs):
    return PlanarInductor(shape, clearance=CLEARANCE, **kwargs)


CASES = {
    'circle':           lambda tmp: coil(CircleShape(30, 10), turns=6, twists=1),
    'circle_twisted':   lambda tmp: coil(CircleShape(30, 10), turns=7, twists=3),
    'circle_1layer':    lambda tmp: coil(CircleShape(30, 10), turns=4, twists=1, layers=1),
    'circle_staggered': lambda tmp: coil(CircleShape(40, 16), turns=5, twists=4, via_diameter=1.0,
                                         stagger_inner_vias=True, stagger_outer_vias=True),
    'rectangle':        lambda tmp: coil(RectangleShape(40, 20, 6), turns=7, twists=3),
    'rectangle_width':  lambda tmp: PlanarInductor(RectangleShape(40, 20, 6), turns=5, twists=2, trace_width=0.8),
    'rectangle_cw':     lambda tmp: coil(RectangleShape(40, 20, 6), turns=5, twists=2, clockwise=True),
    'rectangle_1layer': lambda tmp: coil(RectangleShape(40, 20, 6), turns=4, twists=1, layers=1),
    'hexagon':          lambda tmp: coil(RegularPolygonShape(30, 8, corners=6), turns=8, twists=3),
    'triangle':         lambda tmp: coil(RegularPolygonShape(40, 6, corners=3), turns=4, twists=1),
    'star':             lambda tmp: coil(StarShape(14, 40, 5, points=5), turns=5, twists=1),
    'sector':           lambda tmp: coil(SectorShape(10, 40, math.radians(90), 5), turns=5, twists=2),
    'trapezoid':        lambda tmp: coil(TrapezoidShape(40, 20, 8, 6), turns=7, twists=3),
    'svg_l_shape':      lambda tmp: coil(svg_shape(tmp, L_SHAPE_SVG, 5), turns=5, twists=1),
    'svg_rounded':      lambda tmp: coil(svg_shape(tmp, ROUNDED_SVG, 6), turns=7, twists=2),
}


def arc_points(arc):
    """ Points along a footprint arc, from its start through its mid point to its end """
    (x1, y1), (x2, y2), (x3, y3) = [(p.x, p.y) for p in (arc.start, arc.mid, arc.end)]
    # Center of the circle through the three points
    d = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    cx = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / d
    cy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / d
    a1, a2, a3 = [math.atan2(y - cy, x - cx) for x, y in ((x1, y1), (x2, y2), (x3, y3))]
    sweep = (a3 - a1) % (2*math.pi)
    if (a2 - a1) % (2*math.pi) > sweep: # the arc runs clockwise
        sweep -= 2*math.pi
    a = a1 + np.linspace(0, sweep, ARC_POINTS)
    r = math.dist((x1, y1), (cx, cy))
    points = list(zip(cx + r*np.cos(a), cy + r*np.sin(a)))
    return [(x1, y1), *points[1:-1], (x3, y3)]


def node(layer, point):
    """ Graph node of a track end point, on KiCad's 1 nm grid """
    return layer, round(point[0], 6), round(point[1], 6)


def conductor_graph(footprint):
    """ The footprint's copper as a graph. Nodes are (layer, x, y) end points, edges are tracks and arcs with their
    points, and vias connecting their two layers (without points). Tracks shorter than KiCad's 1 nm grid are left out. """
    graph = nx.Graph()
    tracks = [(line.layer, [(line.start.x, line.start.y), (line.end.x, line.end.y)]) for line in footprint.lines]
    tracks += [(arc.layer, arc_points(arc)) for arc in footprint.arcs]
    tracks = [(layer, points) for layer, points in tracks if node(layer, points[0]) != node(layer, points[-1])]
    for layer, points in tracks:
        graph.add_edge(node(layer, points[0]), node(layer, points[-1]), layer=layer, points=points)
    vias = [pad for pad in footprint.pads if pad.type == Atom.thru_hole]
    for via in vias:
        graph.add_edge(*[node(layer, (via.at.x, via.at.y)) for layer in via.layers], layer=None, points=[])
    assert graph.number_of_edges() == len(tracks) + len(vias), 'duplicate tracks or vias'

    pads = {pad.number: node(pad.layers[0], (pad.at.x, pad.at.y)) for pad in footprint.pads if pad.type == Atom.smd}
    return graph, pads['1'], pads['2']


def conductor_path(graph, start, end):
    """ Points along the conductor from start to end, cut into pieces of at most SEGMENT_LENGTH, and the layer of each
    piece """
    points, layers = [], []
    for a, b in itertools.pairwise(nx.shortest_path(graph, start, end)):
        edge = graph.edges[a, b]
        if edge['layer'] is None: # via
            continue
        track = edge['points'] if node(a[0], edge['points'][0]) == a else edge['points'][::-1]
        for p, q in itertools.pairwise(track):
            n = max(1, math.ceil(math.dist(p, q) / SEGMENT_LENGTH))
            points.extend(np.linspace(p, q, n + 1)[0 if not points else 1:])
            layers.extend([edge['layer']] * n)
    return np.array(points), np.array(layers)


def annulus(shape):
    outline = shapely.Polygon(shape.offset_exterior(0))
    hole = shapely.Polygon(shape.offset_exterior(shape.inner_radius - shape.outer_radius))
    return outline.difference(hole), hole


def check_conductor(footprint):
    """ The tracks, arcs and vias form a single path from pad 1 to pad 2 that does not cross itself on any layer """
    graph, pad1, pad2 = conductor_graph(footprint)
    assert nx.is_connected(graph), 'the conductor is broken'
    assert max(degree for _node, degree in graph.degree) <= 2, 'the conductor branches'
    assert graph.degree[pad1] == graph.degree[pad2] == 1, 'the conductor does not end at the pads'

    points, layers = conductor_path(graph, pad1, pad2)
    changes = np.flatnonzero(layers[1:] != layers[:-1]) + 1 # pieces where the path changes layer
    for a, b in itertools.pairwise([0, *changes, len(layers)]):
        run = shapely.LineString(points[a:b+1])
        assert run.is_simple, f'the conductor crosses or overlaps itself on {layers[a]} near {points[a]}'


def check_clearance(footprint, shape):
    """ Tracks of different turns on the same layer, vias, and vias and tracks of other turns keep the clearance """
    graph, pad1, pad2 = conductor_graph(footprint)
    points, layers = conductor_path(graph, pad1, pad2)
    width = max(track.stroke.width for track in footprint.lines + footprint.arcs)
    clearance = footprint.clearance

    # Winding angle around the center of the hole at each point, and in the middle of each piece
    cx, cy = annulus(shape)[1].representative_point().coords[0]
    winding = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    piece_winding = (winding[:-1] + winding[1:]) / 2
    pieces = shapely.linestrings(np.stack([points[:-1], points[1:]], axis=1))

    for layer in set(layers):
        on_layer = np.flatnonzero(layers == layer)
        i, j = shapely.STRtree(pieces[on_layer]).query(pieces[on_layer], predicate='dwithin',
                                                        distance=width + clearance - TOLERANCE)
        i, j = on_layer[i], on_layer[j]
        other_turn = np.abs(piece_winding[i] - piece_winding[j]) >= math.pi
        i, j = i[other_turn], j[other_turn]
        gaps = shapely.distance(pieces[i], pieces[j]) - width
        assert not len(gaps), f'turns on {layer} are {gaps.min():.3f} mm apart near {points[i[gaps.argmin()]]}'

    vias = [pad for pad in footprint.pads if pad.type == Atom.thru_hole]
    for a, b in itertools.combinations(vias, 2):
        gap = math.dist((a.at.x, a.at.y), (b.at.x, b.at.y)) - (a.size.x + b.size.x)/2
        assert gap >= clearance - TOLERANCE, f'vias at {a.at} and {b.at} are {gap:.3f} mm apart'
    for via in vias:
        center = (via.at.x, via.at.y)
        via_winding = winding[np.argmin(np.hypot(*(points - center).T))]
        other_turns = np.abs(piece_winding - via_winding) >= math.pi
        gap = shapely.distance(shapely.Point(center), pieces[other_turns]).min() - via.size.x/2 - width/2
        assert gap >= clearance - TOLERANCE, f'via at {center} is {gap:.3f} mm from another turn'


def check_annulus(footprint, shape):
    """ The tracks stay within the annulus. Leads into vias are left out: vias lie outside the annulus by design. """
    graph, _pad1, _pad2 = conductor_graph(footprint)
    via_nodes = {n for a, b, layer in graph.edges(data='layer') if layer is None for n in (a, b)}
    tracks = shapely.MultiLineString([points for a, b, points in graph.edges(data='points')
                                      if points and a not in via_nodes and b not in via_nodes])
    area, _hole = annulus(shape)
    outside = tracks.difference(area.buffer(ANNULUS_TOLERANCE))
    assert outside.is_empty, f'tracks leave the annulus at {outside.representative_point()}'


@pytest.mark.parametrize('case', CASES)
def test_geometry(case, tmp_path):
    inductor = CASES[case](tmp_path)
    footprint = inductor.render_footprint()
    check_conductor(footprint)
    check_clearance(footprint, inductor.shape)
    check_annulus(footprint, inductor.shape)


def wider_tracks(footprint):
    for track in footprint.lines + footprint.arcs:
        track.stroke.width *= 1.2

def larger_via(footprint):
    via = next(pad for pad in footprint.pads if pad.type == Atom.thru_hole)
    via.size.x = via.size.y = 3 * via.size.x

def missing_track(footprint):
    del footprint.lines[len(footprint.lines) // 2]

def spike_into_hole(footprint):
    # Move the point between the second and third track of the first arm to the center
    a, b = footprint.lines[1:3]
    a.end.x = a.end.y = b.start.x = b.start.y = 0

def crossed_tracks(footprint):
    # Swap two points of the first arm, so that p1-p2-p3-p4 becomes p1-p3-p2-p4
    a, b, c = footprint.lines[1:4]
    (a.end.x, a.end.y), (b.end.x, b.end.y) = (b.end.x, b.end.y), (b.start.x, b.start.y)
    (b.start.x, b.start.y), (c.start.x, c.start.y) = (a.end.x, a.end.y), (b.end.x, b.end.y)

@pytest.mark.parametrize('defect, check', [
    (wider_tracks, check_clearance),
    (larger_via, check_clearance),
    (missing_track, lambda footprint, shape: check_conductor(footprint)),
    (spike_into_hole, check_annulus),
    (spike_into_hole, check_clearance),
    (crossed_tracks, lambda footprint, shape: check_conductor(footprint))])
def test_checks_find_defects(defect, check):
    rectangle = coil(RectangleShape(40, 20, 6), turns=7, twists=3)
    footprint = rectangle.render_footprint()
    check(footprint, rectangle.shape)
    defect(footprint)
    with pytest.raises(AssertionError):
        check(footprint, rectangle.shape)
