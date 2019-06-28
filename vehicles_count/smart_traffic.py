import os
import logging
import logging.handlers
import random
import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

import utils
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)
IMAGE_DIR = "./out"
VIDEO_SOURCE = "test_kp_metro.mp4"
SHAPE = (720, 1280) 
EXIT_PTS = np.array([
    [[720, 329],  [775, 720], [1280, 720], [1280, 330]],
    [[464, 47], [107, 260], [568, 268], [567, 47]]
])


def train_bg_subtractor(inst, cap, num=500):
    print ('Yo, am learning the background...\nSo that my team no_name_yet ._. can later write a code to subtract background from cctv feed and count vehicles for SIH ;-;')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR),
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    cap = skvideo.io.vreader(VIDEO_SOURCE)

    train_bg_subtractor(bg_subtractor, cap, num=500)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Some shit happened, bye...")
            break

        _frame_number += 1

        if _frame_number % 2 != 0:
            continue

        frame_number += 1


        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()


if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating a folder `%s` to output raw frames", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
