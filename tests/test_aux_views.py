import cv2
import numpy as np

from autoscout.vision.aux_views import AuxViewNormalizer, panels


def test_cropped_reef_panels_keep_their_coordinates_and_exclude_the_barge():
    rng = np.random.default_rng(12)
    reference = rng.integers(0, 256, (300, 1200, 3), dtype=np.uint8)
    views = {"blue_reef": (0, 60, 600, 240), "red_reef": (600, 60, 600, 240)}
    frame = rng.integers(0, 256, reference.shape, dtype=np.uint8)
    frame[:60] = reference[:60]
    frame[60:, :350] = reference[60:, 100:450]
    frame[60:, 850:] = reference[60:, 700:1050]
    frame[60:, 350:354] = (255, 0, 255)
    frame[60:, 846:850] = (255, 0, 255)
    assert len(panels(frame, (60, 240))) == 3
    normalize = AuxViewNormalizer(reference, views)
    fixed = normalize(frame)
    assert len(normalize.transforms) == 2
    assert np.array_equal(fixed[:60], reference[:60])
    for x in (160, 760):
        assert np.abs(fixed[100:240, x:x + 150].astype(float) - reference[100:240, x:x + 150]).mean() < 8
    assert not fixed[60:, :90].any()  # cropped-out pixels are unknown, never barge imagery


def test_regular_layout_and_single_views_are_unchanged():
    frame = np.full((200, 400, 3), 100, dtype=np.uint8)
    normalize = AuxViewNormalizer(frame, {"blue": (0, 100, 200, 100), "red": (200, 100, 200, 100)})
    assert normalize(frame) is frame
    assert AuxViewNormalizer(frame, {"one": (0, 0, 400, 200)})(frame) is frame
