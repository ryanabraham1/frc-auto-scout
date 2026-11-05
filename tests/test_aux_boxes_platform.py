"""Close-up robot boxes must not depend on which OCR the platform has."""
from types import SimpleNamespace

import numpy as np

from autoscout import ocr, pipeline
from autoscout.vision.detect import Det


class _Video:
    width, height, fps = 200, 200, 4.0

    def iter_frames(self, t0, t1, step):
        yield 1.0, np.zeros((200, 200, 3), np.uint8)


def test_boxes_are_collected_when_apple_vision_is_available(monkeypatch):
    monkeypatch.setattr(ocr, "has_vision", lambda: True)
    monkeypatch.setattr(pipeline, "_aux_detector", lambda: lambda pane, conf: [Det(10, 100, 90, 190, .9, "blue")])
    monkeypatch.setattr(pipeline, "reader", lambda: SimpleNamespace(read_many=lambda crops, teams: [{1351: .9}]))
    cal = SimpleNamespace(aux_views={"blue_reef": (0, 0, 200, 200)}, frame_size=(200, 200))
    timeline = SimpleNamespace(clock=None, samples=[SimpleNamespace(t=0.0), SimpleNamespace(t=5.0)])
    boxes: list = []

    reads = pipeline.collect_aux_reads(_Video(), cal, timeline, {"red": [254], "blue": [1351]}, 0.25, boxes)

    assert [b[1:6] + b[7:8] for b in boxes] == [["blue_reef", 10, 100, 90, 190, 1351]]
    assert reads == [(1.0, "blue_reef", 1351, .9)]
