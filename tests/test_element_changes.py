import numpy as np

from autoscout.vision.element_changes import ElementWatcher


def test_later_occlusion_does_not_redate_an_existing_coral():
    watcher = ElementWatcher({"reef": (0, 0, 80, 80)}, (80, 80))
    for t in np.arange(0, 52, .25):
        image = np.zeros((80, 80, 3), np.uint8)
        # Original placement with a brief occlusion, then a long later visit
        # hides it completely. Its return has stronger contrast than its debut.
        present = (10 <= t < 30 and not 13 <= t < 14) or t >= 38
        if present:
            image[25:40, 25:35] = 255
        watcher.step(float(t), image)
    changes = watcher.changes()
    assert len(changes) == 1
    assert 8 <= changes[0]["t"] <= 12
