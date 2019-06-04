#!/usr/bin/env python3

"""
Uility functions in logging and IO.
"""

import os
import shutil
import logging

DEFAULT_VERBOSITY = 4

_ws_dir = None

_logger = None
_LOGGING_FORMAT = "[%(asctime)s %(levelname)5s %(filename)s %(funcName)s:%(lineno)s] %(message)s"
# REVIEW josephz: How do I enable file-logging as well?
logging.basicConfig(format=_LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# REVIEW josephz: The logger is pretty broken right now -- in particular, the verbosity doesn't transfer recursively.
def get_logger(name, level=logging.DEBUG, verbosity=DEFAULT_VERBOSITY):
    level = max(level, logging.CRITICAL - 10 * verbosity)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _get_util_logger():
    global _logger
    if _logger is None:
        _logger = get_logger("Utility")
    return _logger


def _get_path_from_env(envVar):
    path = os.getenv(envVar, None)
    assert path is not None, \
        "Environment variable '{}' not found: " \
        "please check project installation and ~/.bashrc".format(envVar)
    return path


# REVIEW josephz: This should be configurable without having to touch code.
def getWsDir(envName="WS_PATH"):
    global _ws_dir
    if _ws_dir is None:
        _ws_dir = _get_path_from_env(envName)
    return _ws_dir


def get_rel_data_path(*relPath):
    return os.path.join(getWsDir(), "data", *relPath)


def get_rel_raw_path(*relPath):
    return get_rel_data_path("raw", *relPath)


def get_rel_weights_path(*relPath):
    return get_rel_data_path("weights", *relPath)


def get_rel_datasets_path(*relPath):
    return get_rel_data_path("datasets", *relPath)


def get_rel_pickles_path(*relPath):
    return get_rel_data_path("pickles", *relPath)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def mv(src, dst, mkdirMode=True, force=False):
    """ Moves src to dst as if `mv` was used. Both src and dst are relative to root,
    which is set to the 'data path' by default. With the mkdir option, we enforce
    the dst to be a path and we support "move to dir" behavior. Otherwise we support
    "move to dir and rename file" behavior.
    """
    assert os.path.exists(src), "'{}' not found".format(src)

    # In mkdir mode, we enforce the dst to be a path to allow "move to dir" behavior.
    # Otherwise we are supporting "move to dir and rename" behavior.
    if not dst.endswith('/') and mkdirMode:
        dst += '/'
    dstHeadPath, _ = os.path.split(dst)
    mkdir(dstHeadPath)

    if os.path.isdir(dst):
        _get_util_logger().info("Moving '{}' into directory '{}'".format(src, dst))
    else:
        _get_util_logger().info("Renaming '{}' to '{}'".format(src, dst))

    if force:
        shutil.copy(src, dst)
    else:
        shutil.move(src, dst)


def _ensure_exists(path: str):
    assert isinstance(path, str)
    assert os.path.exists(path), "Could not find dir: {}".format(path)


def ensure_dir(path: str):
    _ensure_exists(path)
    assert os.path.isdir(path), "Is not a directory: {}".format(path)


def ensure_file(path: str):
    _ensure_exists(path)
    assert os.path.isfile(path), "Is not a file: {}".format(path)


def ensure_path_free(path: str):
    assert isinstance(path, str)
    assert not os.path.exists(path), "Path already exists: {}".format(path)


def rm(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in str_to_bool.dict:
        raise ValueError("Need bool; got {}".format(type(s)))
    return str_to_bool.dict[s.lower()]
str_to_bool.dict = {
    "yes": True,
    "y": True,
    "true": True,
    "+": True,
    "no": False,
    "n": False,
    "false": False,
    "-": False
}
