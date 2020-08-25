import os
import csv
import logging
import subprocess

from smoke.utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'bicycle',
    1: 'bus',
    2: 'car',
    3: 'construction_vehicle',
    4: 'motorcycle',
    5: 'pedestrian',
    6: 'trailer',
    7: 'truck'
}


def nusc_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing nuscenes detection evaluation: ")
        do_nusc_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_nusc_detection_evaluation(dataset,
                                 predictions,
                                 output_folder,
                                 logger
                                ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    for image_id, prediction in predictions.items():
        sample_id = image_id.split()[-1]
        generate_nusc_3d_detection(prediction)


def generate_nusc_3d_detection(prediction):
    # TODO: finish generate detection
    pass