import os
import random
from io import open
import json
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.functional as F


def load_conf(path_to_yaml):
    """Wrapper for configuration file loading through OmegaConf."""
    conf = OmegaConf.load(path_to_yaml)
    if "env" in conf.keys() and conf.env.base_dir is None:
        OmegaConf.update(conf, "env.base_dir", get_root_dir())
    return conf


def merge_conf(base_conf_path, dataset_conf_path, model_conf_path):
    """Wrapper for to merge multiple config files through OmegaConf."""
    base_conf = load_conf(base_conf_path)
    dataset_conf = load_conf(dataset_conf_path)
    model_conf = load_conf(model_conf_path)

    conf = OmegaConf.merge(base_conf, dataset_conf, model_conf)
    return conf


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def update_conf_with_cli_params(params, config):
    params_dict = vars(params)
    for param in params_dict:
        if params_dict[param] is not None:
            # TODO automatically find mapping for arbitrary depth
            if param in config.keys():
                new_param = params_dict[param]
                if isinstance(new_param, str) and new_param.lower() in [
                    "true",
                    "false",
                ]:
                    new_param = new_param.lower() == "true"
                OmegaConf.update(config, "{}".format(param), new_param)
            else:
                for top_level_key in config.keys():
                    if isinstance(config[top_level_key], dict):
                        if param in list(config[top_level_key].keys()):
                            new_param = params_dict[param]
                            if isinstance(new_param, str) and new_param.lower() in [
                                "true",
                                "false",
                            ]:
                                new_param = new_param.lower() == "true"
                            OmegaConf.update(
                                config, "{}.{}".format(top_level_key, param), new_param
                            )


def save_json(output_path, content):
    with open(output_path, "w") as outfile:
        json.dump(content, outfile)


def get_root_dir():
    # TODO: below should be run only once, then saved
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "../.."))
    return root


def normalize(x, p, dim):
    norm = F.norm(x, p=p, dim=dim)
    return x / norm


def scale(x, axis=0):
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    scale = std[std == 0.0] = 1.0
    x -= mean
    x /= scale
    return np.array(x, dtype=np.float32)
