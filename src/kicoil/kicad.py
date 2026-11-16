
from gerbonara.cad.kicad.pcb import Board, TrackSegment, TrackArc, Via
import copy

from gerbonara.cad.kicad.footprints import Atom, AtPos, XYCoord, Pad, Line, Arc, Stroke, Drill, Footprint
from gerbonara.cad.kicad.base_types import DrawnProperty
from kipy.board_types import BoardArc, BoardSegment, Pad as KipyPad, PadStackShape, PadType, Zone, ZoneType, \
        ZoneConnectionStyle
from kipy.geometry import Angle, Vector2, PolygonWithHoles, PolyLineNode
from kipy.util import from_mm
from kipy.util.board_layer import layer_from_canonical_name, iter_copper_layers

def make_pad(num, layer, x, y, diameter, clearance):
    return Pad(
            number=str(num),
            type=Atom.smd,
            shape=Atom.circle,
            at=AtPos(x=x, y=y),
            size=XYCoord(x=diameter, y=diameter),
            layers=layer,
            clearance=clearance,
            zone_connect=0)

def make_line(x1, y1, x2, y2, trace_width, layer):
    return Line(
                start=XYCoord(x=x1, y=y1),
                end=XYCoord(x=x2, y=y2),
                layer=layer, 
                stroke=Stroke(width=trace_width))

def make_arc(x1, y1, x2, y2, xm, ym, trace_width, layer):
    return Arc(
                start=XYCoord(x=x1, y=y1),
                mid=XYCoord(x=xm, y=ym),
                end=XYCoord(x=x2, y=y2),
                layer=layer, 
                stroke=Stroke(width=trace_width))


def make_via(x, y, diameter, drill, clearance, layers):
    return Pad(number="NC",
                     type=Atom.thru_hole,
                     shape=Atom.circle,
                     at=AtPos(x=x, y=y),
                     size=XYCoord(x=diameter, y=diameter),
                     drill=Drill(diameter=drill),
                     layers=layers,
                     clearance=clearance, 
                     zone_connect=0)


def footprint_to_board(footprint):
    """ Convert a coil footprint into a board: tracks, arcs and vias become free board items, the keepout zone becomes a
    board rule area. The terminal pads are SMD pads, which can only exist inside a footprint, so they are placed on the
    board in a footprint of their own (named after the coil footprint with a "-terminals" suffix). """
    board = Board.empty_board(
            zones=list(footprint.zones),
            # Built with keywords: gerbonara's TrackSegment.from_footprint_line passes layer and locked positionally in
            # the wrong order, which wrote every straight segment with layer "False".
            track_segments=[TrackSegment(start=line.start, end=line.end, width=line.width or line.stroke.width,
                                         layer=line.layer) for line in footprint.lines],
            track_arcs=[TrackArc(start=arc.start, mid=arc.mid, end=arc.end, width=arc.stroke.width, layer=arc.layer)
                        for arc in footprint.arcs],
            vias=[Via.from_pad(pad) for pad in footprint.pads if pad.type == Atom.thru_hole])

    terminal_pads = [copy.deepcopy(pad) for pad in footprint.pads if pad.type != Atom.thru_hole]
    if terminal_pads:
        terminals = Footprint(
                name=f'{footprint.name or "planar-coil"}-terminals',
                layer='F.Cu',
                descr=f'Terminal pads of {footprint.name or "a planar coil"}',
                properties=[DrawnProperty(key='Reference', value='L1', at=AtPos(x=0, y=0), layer='F.SilkS', hide=True),
                            DrawnProperty(key='Value', value=footprint.name or 'planar-coil', at=AtPos(x=0, y=0),
                                          layer='F.Fab', hide=True)],
                pads=terminal_pads)
        board.footprints.append(terminals)

    # gerbonara (1.6.0) cannot serialize the empty list of legacy "module" footprints that Board() creates by default
    # (MappingError in Board.serialize()). None is skipped by the serializer, and is what a parsed board without legacy
    # footprints holds as well.
    board.legacy_footprints = None
    return board


def update_footprint_instance(instance, footprint):
    """ Replace the items of a kipy FootprintInstance with those of a coil footprint, keeping its position and
    orientation. """
    # kipy stores footprint items with board coordinates. Build them at the origin and let the position and
    # orientation setters move them into place.
    position, orientation = instance.position, instance.orientation
    instance.position = Vector2()
    instance.orientation = Angle()
    instance.definition.items = kipy_items(footprint)
    instance.orientation = orientation
    instance.position = position

    instance.definition.id.library = ''
    instance.definition.id.name = footprint.name
    instance.attributes.exclude_from_bill_of_materials = True
    instance.overrides.copper_clearance = from_mm(footprint.clearance)
    instance.overrides.zone_connection = ZoneConnectionStyle.ZCS_NONE


def kipy_items(footprint):
    """ Convert the lines, arcs, pads and keepout zones of a footprint into kipy footprint items. """
    def point(xy):
        return Vector2.from_xy_mm(xy.x, xy.y)

    items = []
    for line in footprint.lines:
        seg = BoardSegment()
        seg.start, seg.end = point(line.start), point(line.end)
        seg.attributes.stroke.width = from_mm(line.stroke.width)
        seg.layer = layer_from_canonical_name(line.layer)
        items.append(seg)

    for ref in footprint.arcs:
        arc = BoardArc()
        arc.start, arc.mid, arc.end = point(ref.start), point(ref.mid), point(ref.end)
        arc.attributes.stroke.width = from_mm(ref.stroke.width)
        arc.layer = layer_from_canonical_name(ref.layer)
        items.append(arc)

    for ref in footprint.pads:
        pad = KipyPad()
        pad.number = ref.number
        pad.position = point(ref.at)
        pad.padstack.layers = [layer_from_canonical_name(name) for name in ref.layers]
        layer, = pad.padstack.copper_layers
        layer.shape = PadStackShape.PSS_CIRCLE
        layer.size = point(ref.size)
        if ref.type == Atom.thru_hole:
            pad.pad_type = PadType.PT_PTH
            pad.padstack.drill.diameter = Vector2.from_xy_mm(ref.drill.diameter, ref.drill.diameter)
        else:
            pad.pad_type = PadType.PT_SMD
        items.append(pad)

    for ref in footprint.zones:
        zone = Zone()
        zone.type = ZoneType.ZT_RULE_AREA
        zone.layers = list(iter_copper_layers()) # "*.Cu"
        zone.proto.rule_area_settings.keepout_copper = not ref.keepout.copperpour_allowed
        outline = PolygonWithHoles()
        for xy in ref.polygon.pts:
            outline.outline.append(PolyLineNode.from_xy(from_mm(xy.x), from_mm(xy.y)))
        zone.outline = outline
        items.append(zone)

    return items
