import os
import argparse
from omegaconf import OmegaConf

from muscall.tasks.retrieval import Retrieval
from muscall.tasks.classification import Zeroshot
from muscall.utils.utils import get_root_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MusCALL model")

    parser.add_argument(
        "model_id",
        type=str,
        help="experiment id under which trained model was saved",
    )
    parser.add_argument(
        "task",
        type=str,
        help="name of the evaluation task (retrieval or zeroshot)",
    )
    parser.add_argument(
        "--test_set_size",
        type=int,
        help="size of the random testing set",
        default=1000,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of dataset for zeroshot classification",
    )
    parser.add_argument(
        "--device_num",
        type=str,
        default="0",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()
    model_id = params.model_id

    muscall_config = OmegaConf.load(
        os.path.join(get_root_dir(), "save/experiments/{}/config.yaml".format(model_id))
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    if params.task == "retrieval":
        evaluation = Retrieval(muscall_config, params.test_set_size)
    elif params.task == "zeroshot":
        evaluation = Zeroshot(muscall_config, params.dataset_name)
    else:
        raise ValueError("{} task not supported".format(params.task))

    evaluation.evaluate()
