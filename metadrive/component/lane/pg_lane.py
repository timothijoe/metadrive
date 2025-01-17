import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import DrivableAreaProperty


class PGLane(AbstractLane):
    POLYGON_SAMPLE_RATE = 1
    radius = 0.0

    def __init__(self):
        super(PGLane, self).__init__()
        # one should implement how to get polygon in property def polygon(self)
        self._polygon = None

    def construct_sidewalk(self, block, lateral):
        radius = self.radius
        segment_num = int(self.length / DrivableAreaProperty.SIDEWALK_LENGTH)
        for segment in range(segment_num):
            lane_start = self.position(segment * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            lane_end = self.position((segment + 1) * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            if segment == segment_num - 1:
                lane_end = self.position(self.length, lateral)
            if radius == 0:
                factor = 1
            else:
                if self.is_clockwise():
                    factor = (1 - block.SIDEWALK_LINE_DIST / radius)
                else:
                    factor = (1 + block.SIDEWALK_WIDTH / radius) * (1 + block.SIDEWALK_LINE_DIST / radius)
            node_path_list = self.construct_sidewalk_segment(
                block,
                lane_start,
                lane_end,
                length_multiply=factor,
                extra_thrust=DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
            )
            self._node_path_list.extend(node_path_list)

    @property
    def polygon(self):
        raise NotImplementedError("Overwrite this function to allow getting polygon for this lane")

    def get_polygon(self):
        return self.polygon

    def has_polygon(self):
        return True if self.polygon is not None else False
