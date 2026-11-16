import math
import copy
import itertools
import os
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from pathlib import Path
import importlib.resources
import lzma
import sys
import hashlib
import bisect

import numpy as np
import platformdirs
import shapely
import wasmtime


# Points closer than this (mm) are one point
DUPLICATE_POINT_TOLERANCE = 1e-6

# The straight skeleton only gives mitered offsets. The miter of a reflex corner that turns by phi moves 1/cos(phi/2)
# times as fast as the outline, so at a cusp it reaches far into the shape. Reflex corners that turn by more than
# ROUND_JOINT_MIN_TURN get a small round notch of ROUND_JOINT_RADIUS (mm) instead, whose offsets are arcs.
ROUND_JOINT_MIN_TURN = math.radians(120)
ROUND_JOINT_RADIUS = 0.01
ROUND_JOINT_SEGMENTS = 16 # per quarter circle

# Number of depths at which set_angle_reference measures the edge lengths
ANGLE_REFERENCE_SAMPLES = 32

TIME_EPS = 1e-9
ANGLE_EPS = 1e-9
ANGLE_GRID = 2**40 # vertex angles are multiples of 1/ANGLE_GRID, exact in sums with integers below 2**12

RAMP_MARGIN = 1e-3 # mm from the corners to the ramps of the flat corner layout; closer points give zero-length tracks
FLAT_CORNER_STEP = 1e-6 # revolutions, angle span in front of the corner of a flat stretch, see flat_corners


def interpolate(p1, p2, t, t_start=0, t_end=1):
    """ Interpolate along the line from point p1 to point p2 using t as the interpolation variable.
        t_start and t_end set the bounds of t, t_start at p1, and t_end at p2.
        When both interval ends coincide, clips and returns p1.
    """
    if math.isclose(t_start, t_end):
        return p1
    t_range = t_end - t_start
    t = (t - t_start) / t_range
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    return (x1 + t*dx, y1 + t*dy)


def edge_cycle(points):
    """ From a list of points return an iterator of all edges assuming they are a closed loop:
        [A B C] -> [AB BC CA]
    """
    return itertools.pairwise(itertools.chain(points, points[:1]))


def clean_polygon(points):
    """ The polygon as a counter-clockwise list of points without repeated points. Vertex 0 stays vertex 0. """
    ring = shapely.remove_repeated_points(shapely.LinearRing(points), DUPLICATE_POINT_TOLERANCE)
    if not ring.is_ccw:
        ring = ring.reverse()
    return list(ring.coords)[:-1]


def turn_angles(points):
    """ Turn angle in radians at each vertex of a counter-clockwise polygon, negative at reflex vertices. """
    p = np.asarray(points, dtype=float)
    incoming, outgoing = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
    cross = incoming[:, 0]*outgoing[:, 1] - incoming[:, 1]*outgoing[:, 0]
    return np.arctan2(cross, (incoming * outgoing).sum(axis=1))


def round_joints(points):
    """ Cut a round notch around each sharp reflex corner of a counter-clockwise polygon. Vertex 0 stays first. """
    corners = [p for p, turn in zip(points, turn_angles(points)) if turn < -ROUND_JOINT_MIN_TURN]
    if not corners:
        return points
    notches = shapely.MultiPoint(corners).buffer(ROUND_JOINT_RADIUS, quad_segs=ROUND_JOINT_SEGMENTS)
    rounded = shapely.Polygon(points).difference(notches)
    if not isinstance(rounded, shapely.Polygon): # (the notches cut it apart, or it is not a simple polygon)
        return points
    ring = rounded.exterior
    ring = list((ring if ring.is_ccw else ring.reverse()).coords)[:-1]
    start = min(range(len(ring)), key=lambda i: math.dist(ring[i], points[0]))
    return ring[start:] + ring[:start]


class WasmApp:
    def __init__(self, wasm_filename, cachedir="kicoil"):
        module_binary = importlib.resources.read_binary(__package__, wasm_filename)

        module_path_digest = hashlib.sha256(__file__.encode()).hexdigest()
        module_digest = hashlib.sha256(module_binary).hexdigest()
        cache_path = Path(os.getenv("KICOIL_CACHE_DIR", platformdirs.user_cache_dir(cachedir)))
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_filename = (cache_path / f'{wasm_filename}-{module_path_digest[:8]}-{module_digest[:16]}')

        self.engine = wasmtime.Engine()

        try:
            with cache_filename.open("rb") as cache_file:
                self.module = wasmtime.Module.deserialize(self.engine, lzma.decompress(cache_file.read()))
        except:
            print("Preparing to run {}. This might take a while...".format(wasm_filename), file=sys.stderr)
            self.module = wasmtime.Module(self.engine, module_binary)
            with cache_filename.open("wb") as cache_file:
                cache_file.write(lzma.compress(self.module.serialize(), preset=0))

    def run(self, stdin='', argv=[]):
        """ Run the module's main function and return (0, stdout). Raises ValueError with the module's stderr output if
        it exits with an error. """
        with NamedTemporaryFile('r') as stdout_f, NamedTemporaryFile('w') as stdin_f, \
                NamedTemporaryFile('r') as stderr_f:
            stdin_f.write(stdin)
            stdin_f.flush()

            wasi_cfg = wasmtime.WasiConfig()
            wasi_cfg.argv = argv
            wasi_cfg.stdin_file = stdin_f.name
            wasi_cfg.stdout_file = stdout_f.name
            wasi_cfg.stderr_file = stderr_f.name

            linker = wasmtime.Linker(self.engine)
            linker.define_wasi()
            store = wasmtime.Store(self.engine)
            store.set_wasi(wasi_cfg)
            self.app = linker.instantiate(store, self.module)
            linker.define_instance(store, "app", self.app)

            try:
                self.app.exports(store)["_start"](store)
            except wasmtime.ExitTrap as trap:
                if trap.code != 0:
                    raise ValueError(stderr_f.read().strip()) from None
            finally:
                sys.stderr.write(stderr_f.read()) # (also when the module aborts)
            return 0, stdout_f.read()


# Nodes compare and hash by identity: compute_skeleton makes one object per distinct (x, y, time), and the vertex tracks
# of dense outlines hold millions of references to them.
@dataclass(frozen=True, eq=False)
class SkeletonNode:
    x: float
    y: float
    time: float

    @property
    def pos(self):
        return self.x, self.y


skeleton_wasm = WasmApp('skeleton.wasm')

def compute_skeleton(exterior):
    """ Straight skeleton of a counter-clockwise simple polygon as (nodes, edges) """
    input_data = '\n'.join(f'{float(x)!r} {float(y)!r}' for x, y in exterior)
    rc, data = skeleton_wasm.run(input_data)

    # Parse output: each line is "x1 y1 x2 y2 t1 t2"
    node_map = {}  # Map (x, y, t) to SkeletonNode
    edges = []
    for line in data.strip().split('\n'):
        parts = line.split()
        if len(parts) != 6:
            continue
        x1, y1, x2, y2, t1, t2 = map(float, parts)
        for n in (x1, y1, t1), (x2, y2, t2):
            if n not in node_map:
                node_map[n] = SkeletonNode(*n)
        edges.append((node_map[x1, y1, t1], node_map[x2, y2, t2]))

    return list(node_map.values()), edges


def _track_point(track, depth, times):
    """ Position on a vertex track (a list of skeleton nodes with non-decreasing time) at the given depth, see
    Skeletonator.track_position. """
    if len(track) == 1:
        return track[0].pos
    if depth >= times[-1]:
        return track[-1].pos # beyond the end of the track the outline has collapsed
    j = 0 if depth <= times[0] else max(0, min(len(track) - 2, bisect.bisect_right(times, depth) - 1))
    n0, n1 = track[j], track[j+1]
    if n1.time - n0.time <= TIME_EPS:
        return n1.pos if depth >= n1.time else n0.pos
    return interpolate(n0.pos, n1.pos, depth, n0.time, n1.time)


class Skeletonator:
    """ Maps a polygon's straight skeleton to a coordinate system for spiral layout.

    Coordinates are an "angle" t (one revolution per unit, polygon vertex 0 at t=0) and a depth, the inward offset
    distance from the polygon outline (`self.radius - depth` is called the radius). Every vertex of the polygon moves
    along a chain of skeleton nodes ("track") as the outline is offset inwards; the offset outline at a given depth is
    the polygon formed by all vertices' positions on their tracks at that depth. This works up to the first split event,
    where the offset outline falls apart into several pieces: `self.max_depth`.

    The skeleton is computed for the outline with round notches at its sharp reflex corners (`self.poly`, see
    round_joints). Its vertices are the vertices of the spiral. `self.outline` is the polygon as given (normalized).

    The angle coordinate is fixed for the whole shape: vertex i always sits at angle `self.vertex_angles[i]`, no matter
    the depth, so that consecutive turns of a spiral and the arms of a multi-arm spiral line up at every vertex.
    """

    def __init__(self, poly, ref_depth=None):
        self.outline = clean_polygon(poly)
        self.poly = round_joints(self.outline)
        self.skeleton_nodes, self.skeleton_edges = compute_skeleton(self.poly)
        self.debug_arms = []
        xs, ys = zip(*self.poly)
        self.teps = 1e-9 * math.hypot(max(xs) - min(xs), max(ys) - min(ys)) # time tolerance of constant-time edges
        self.radius = max(n.time for n in self.skeleton_nodes)
        self._find_events()
        contour = {n.pos: n for n in self.skeleton_nodes if n.time == 0}
        self.tracks = [self._follow_track(contour[p]) for p in self.poly]
        self._track_times = [[n.time for n in track] for track in self.tracks]
        self.ref_depth = None
        self.set_angle_reference(ref_depth)

    def _find_events(self):
        """ Group the skeleton nodes joined by constant-time edges into events (CGAL emits several nodes joined by
        zero-length edges at degenerate events, and constant-time edges e.g. at the ridge of a rectangle). Vertex tracks
        go from event to event towards increasing time. An event with more than one way up splits the offset outline
        into several pieces. Sets self.next_node ({node: (node of its event from which the way up leaves, the node
        above)}) and self.max_depth (the time of the first split). """
        adjacency = {}
        for a, b in self.skeleton_edges:
            if a is not b:
                adjacency.setdefault(a, []).append(b)
                adjacency.setdefault(b, []).append(a)
        event = {}
        for n0 in adjacency:
            if n0 not in event:
                event[n0], todo = [n0], [n0]
                while todo:
                    n = todo.pop()
                    for m in adjacency[n]:
                        if m not in event and abs(m.time - n.time) <= self.teps:
                            event[m] = event[n0]
                            event[n0].append(m)
                            todo.append(m)
        self.max_depth = self.radius
        self.next_node = {}
        for group in {id(g): g for g in event.values()}.values():
            ups = {id(event[m]): (n, m) for n in group for m in adjacency[n] if m.time > n.time + self.teps}
            if len(ups) > 1:
                self.max_depth = min(self.max_depth, group[0].time)
            elif ups:
                self.next_node.update((n, *ups.values()) for n in group)

    def _follow_track(self, node):
        track = [node]
        while (step := self.next_node.get(track[-1])) is not None:
            track += step[1:] if step[0] is track[-1] else step
        return track

    def set_angle_reference(self, max_depth=None):
        """ Fix the angle coordinate of every vertex.

        Each edge gets an angle span proportional to its *shortest* length over the depth range [0, max_depth] that the
        spiral uses. The depth difference a spiral arm covers along an edge is proportional to the edge's angle span, so
        this bounds the slope of the spiral relative to the edge (and thus the loss of clearance between turns caused
        by that slope) at every depth, including edges that shrink (convex corners) or grow (reflex corners). Edges that
        collapse within the range get a span of zero: their vertices share one angle, and a spiral turn runs along them
        at constant depth.
        """
        max_depth = max(0, min(self.max_depth if max_depth is None else max_depth, self.max_depth))
        if max_depth == self.ref_depth:
            return
        self.ref_depth = max_depth
        self._set_edge_spans(self._min_edge_lengths(max_depth))

    def _min_edge_lengths(self, max_depth):
        """ Shortest length of every edge over the depths from 0 to max_depth. Edges that (nearly) collapse get exactly
        0, so that ties in _locate are exact. If everything collapses, the lengths on the outline. """
        depths = np.linspace(0, max_depth, ANGLE_REFERENCE_SAMPLES)
        positions = np.array([[self.track_position(i, d) for d in depths] for i in range(len(self.poly))])
        lengths = np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=2)
        min_len = lengths.min(axis=1)
        min_len[min_len <= 1e-9 * lengths[:, 0].sum()] = 0
        return min_len if min_len.any() else lengths[:, 0]

    def _set_edge_spans(self, lengths):
        """ Set vertex_angles and edge_spans so that each edge's angle span is proportional to its entry in lengths. """
        # Quantized to a binary grid, so that `revolution + angle` is exact and ties between vertices of a collapsing
        # group compare exactly. Vertices at the end sharing their angle with vertex 0 get an angle of exactly 1.
        angles = np.concatenate([[0], np.cumsum(lengths[:-1]) / np.sum(lengths)])
        angles = np.round(angles * ANGLE_GRID) / ANGLE_GRID
        self.vertex_angles = [1.0 if 1.0 - a < 1e-12 else a for a in angles.tolist()]
        self.edge_spans = [b - a for a, b in itertools.pairwise(self.vertex_angles + [1.0])]

    def track_position(self, i, depth):
        """ Position of polygon vertex i on the offset outline at the given depth. Depths beyond the end of the track are
        clamped to the track's end. Negative depths are extrapolated along the track's first piece, which is only a
        local extrapolation (used for the outer via ring), not a real outward offset. """
        return _track_point(self.tracks[i], depth, self._track_times[i])

    def snap_angle(self, t):
        """ Snap an angle to the nearest vertex angle if it is within ANGLE_EPS of it, so that comparisons of computed
        arm angles with vertex angles (several vertices can share one) are exact. """
        k = math.floor(t)
        frac = t - k
        i = bisect.bisect_left(self.vertex_angles, frac)
        for j in (i-1, i):
            if 0 <= j < len(self.vertex_angles) and abs(self.vertex_angles[j] - frac) < ANGLE_EPS:
                return k + self.vertex_angles[j]
        return k + 1.0 if 1.0 - frac < ANGLE_EPS else t

    def _locate(self, t):
        """ For angle t, return (k, i, f): t lies in revolution k on the edge from vertex i to vertex i+1 at fraction f.
        When several vertices share angle t, this returns the *first* of them (as the end of the edge leading to it,
        f=1). Spiral arms start and end there and include the other vertices of the group in their first turn. """
        k = math.floor(t)
        frac = t - k
        if frac <= 0:
            k, frac = k-1, 1.0
        i = bisect.bisect_left(self.vertex_angles, frac) - 1 # the edge leading to the first vertex with angle >= frac
        span = self.edge_spans[i]
        f = 1.0 if span <= 0 else (frac - self.vertex_angles[i]) / span
        return k, i, max(0.0, min(1.0, f))

    def _vertex_angle(self, k, i):
        if i >= len(self.poly):
            return k + 1 + self.vertex_angles[i - len(self.poly)]
        return k + self.vertex_angles[i]

    def _point_on_spiral(self, t, depth_fn):
        """ Point at angle t on the polyline through all vertices v at depth depth_fn(angle of v) """
        k, i, f = self._locate(t)
        p0 = self.track_position(i, depth_fn(self._vertex_angle(k, i)))
        p1 = self.track_position((i+1) % len(self.poly), depth_fn(self._vertex_angle(k, i+1)))
        return interpolate(p0, p1, f)

    def project_point(self, t, depth=0.0):
        """ Point at angle t on the offset outline at the given depth """
        return self._point_on_spiral(self.snap_angle(t), lambda a: depth)

    def iter_vertex_angles(self, t1, t2):
        """ Yield (angle, vertex index) for all vertices with t1 <= angle < t2, in order """
        for k in itertools.count(math.floor(t1) - 1):
            for i, a in enumerate(self.vertex_angles):
                if k + a >= t2:
                    return
                if k + a >= t1:
                    yield k + a, i

    def spiral(self, t1, t2, d1, d2, debug=False):
        """ Points and angles of a spiral arm from angle t1 at depth d1 to angle t2 at depth d2. The depth changes
        linearly with the angle. The points are in order from t1 to t2, also when t2 < t1. """
        if t2 < t1:
            points, angles = self.spiral(t2, t1, d2, d1, debug)
            return points[::-1], angles[::-1]
        t1, t2 = self.snap_angle(t1), self.snap_angle(t2)
        depth_fn = (lambda t: d1) if math.isclose(t1, t2) else (lambda t: d1 + (d2 - d1) * (t - t1) / (t2 - t1))

        # The end points are placed on the offset outline at constant depth, so that they do not depend on the
        # direction of the spiral: the ends of inward and outward arms meet exactly.
        points, angles, info = [self._point_on_spiral(t1, lambda a: d1)], [t1], [(1, d1, None)]
        for a, i in self.iter_vertex_angles(t1, t2):
            d = d1 if a == t1 else depth_fn(a)
            p = self.track_position(i, d)
            if math.dist(p, points[-1]) > 1e-12:
                points.append(p)
                angles.append(a)
                info.append((3, d, i))
        p = self._point_on_spiral(t2, lambda a: d2)
        if math.dist(p, points[-1]) > 1e-12 or len(points) == 1:
            points.append(p)
            angles.append(t2)
            info.append((2, d2, None))
        else:
            points[-1], angles[-1], info[-1] = p, t2, (2, d2, None)
        if debug:
            # Points as (position, kind: 1 start, 2 end, 3 vertex, False, radius, vertex index) for dump_to_pdf
            arm = [(p, kind, False, self.radius - d, i) for p, (kind, d, i) in zip(points, info)]
            self.debug_arms.append((arm, d1 > d2, t1, t2))
        return points, angles

    def flat_corners(self, depth):
        """ A copy of this skeletonator whose spirals keep their depth around the corners, down to the given depth, and
        change it only on a ramp in the middle of each edge. Returns None if no edge has room for a ramp. """
        poly, tracks, spans = [], [], []
        vertex, collapses = [], [] # per point: its vertex of self.poly (None on a ramp), whether its edge collapses
        min_len, turns = self._min_edge_lengths(depth), abs(turn_angles(self.poly))
        for i, ((x0, y0), (x1, y1)) in enumerate(edge_cycle(self.poly)):
            j = (i + 1) % len(self.poly)
            length = math.dist((x0, y0), (x1, y1))
            ex, ey = (x1 - x0) / length, (y1 - y0) / length

            def along_edge(v):
                """ Positions of vertex v along the edge at every depth where its track bends """
                depths = [0.0, depth, *(t for t in self._track_times[v] if 0 < t < depth)]
                return [(x - x0)*ex + (y - y0)*ey for x, y in (self.track_position(v, d) for d in depths)]

            def ramp_track(u, v, sign):
                """ Track of the point at u along the edge, on the edge's offset line. Outside the depths [0, depth],
                where only vias go, it stops at vertex v once v's track gets past it. """
                def at(d):
                    return SkeletonNode(x0 + u*ex - d*ey, y0 + u*ey + d*ex, d)

                def clamped(n):
                    return n if sign * ((n.x - x0)*ex + (n.y - y0)*ey - u) > 0 else at(n.time)
                outward = SkeletonNode(*self.track_position(v, -depth), -depth)
                deeper = [n for n in self.tracks[v] if n.time > depth]
                return [clamped(outward), at(0.0), at(depth)] + [clamped(n) for n in deeper]

            poly.append(self.poly[i])
            tracks.append(self.tracks[i])
            vertex.append(i)
            # The ramp lies between the two corners at every depth
            lo, hi = max(along_edge(i)) + RAMP_MARGIN, min(along_edge(j)) - RAMP_MARGIN
            if hi - lo > RAMP_MARGIN:
                poly += [(x0 + lo*ex, y0 + lo*ey), (x0 + hi*ex, y0 + hi*ey)]
                tracks += [ramp_track(lo, i, 1), ramp_track(hi, j, -1)]
                spans += [0.0, hi - lo, 0.0]
                vertex += [None, None]
                collapses += [False, False, False]
            else:
                spans.append(0.0)
                collapses.append(min_len[i] <= RAMP_MARGIN)
        if not any(spans):
            return None
        # An arm that starts or ends on a flat stretch, and its via, belong at its sharpest corner, not at its start. A
        # tiny span in front of that corner makes it the first point at the stretch's angle. The edge with that span
        # must not collapse (its ends would come out as a hairpin), else it goes on the edge before.
        n, ramps, step = len(spans), [s > 0 for s in spans], FLAT_CORNER_STEP * sum(spans)
        for k in range(n):
            if ramps[k]: # the flat stretch runs from the ramp's end, point k + 1, to the next ramp's start
                end = next(m for m in itertools.count(k + 1) if ramps[m % n])
                corner = max(range(k + 2, end), key=lambda m: turns[vertex[m % n]])
                spans[max(e for e in range(k + 1, corner) if not collapses[e % n]) % n] = step

        flat = copy.copy(self)
        flat.poly, flat.tracks = poly, tracks
        flat._track_times = [[node.time for node in track] for track in tracks]
        # Each ramp gets an angle span (and so a depth change) proportional to its length
        flat._set_edge_spans(spans)
        return flat

    def dump_to_pdf(self, filename):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(10, 20))
        ax = axs.flatten()[0]

        # polygon outline
        poly_x = [p[0] for p in self.outline] + [self.outline[0][0]]
        poly_y = [p[1] for p in self.outline] + [self.outline[0][1]]
        ax.plot(poly_x, poly_y, '-', color='black', linewidth=.5, label='Polygon')

        # skeleton edges
        for node1, node2 in self.skeleton_edges:
            ax.plot([node1.x, node2.x], [node1.y, node2.y], '-', color='gray', linewidth=.5, alpha=0.7)

        # skeleton nodes
        for n in self.skeleton_nodes:
            if n in self.next_node:
                ax.plot(n.x, n.y, 'o', color='black', markersize=3, alpha=0.5)
            else:
                ax.plot(n.x, n.y, 'o', markerfacecolor='none', markeredgecolor='magenta', markersize=4)

        count = {True: 0, False: 0}
        for arm, direction, t1, t2 in self.debug_arms:

            for case_filter in [1, 2]:
                xs = [x for (x, y), case, start, r, i in arm if case == case_filter]
                ys = [y for (x, y), case, start, r, i in arm if case == case_filter]
                ax.plot(xs, ys, 'o', linestyle='none', markersize=4, markeredgecolor='red' if case_filter == 1 else 'blue')

            xs = [x for (x, y), case, start, r, i in arm if start]
            ys = [y for (x, y), case, start, r, i in arm if start]
            ax.plot(xs, ys, 'o', linestyle='none', markersize=2, markeredgecolor='black')

            xs = [x for (x, y), case, start, r, i in arm]
            ys = [y for (x, y), case, start, r, i in arm]
            ax.plot(xs, ys, linewidth=.2, color='red' if direction else 'blue')

            align = 'left' if direction else 'right'
            ax.text(xs[-1], ys[-1], f'{count[direction]}', size=3, horizontalalignment=align)
            ax.text(xs[0], ys[0], f'{count[direction]}', size=3, horizontalalignment=align, color='gray')

            ax2 = axs.flatten()[1]
            xs = [i for p, case, start, r, i in arm if i is not None]
            ys = [r for p, case, start, r, i in arm if i is not None]
            ax2.plot(xs, ys, linewidth=.2, color='red' if direction else 'blue')
            if xs:
                ax2.text(xs[-1], ys[-1], f'{count[direction]}')
                ax2.text(xs[0], ys[0], f'{count[direction]}')

            count[direction] += 1

        xs, ys = zip(*(self.track_position(0, (i/99) * self.max_depth) for i in range(100)))
        ax.plot(xs, ys, '--', linewidth=.5, color='black')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Polygon Skeleton (radius: {self.radius:.3f}, max depth: {self.max_depth:.3f})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.invert_yaxis()

        fig.savefig(filename, bbox_inches='tight', dpi=600)
        plt.close(fig)
