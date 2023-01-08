import numpy as np
from spiceypy import spiceypy as spice
from typing import Callable, Iterator, Iterable, Optional, Tuple, Union, Sequence


def load_spice_kernel(kernel_dir: str, list_of_kernel: Union[str, Iterable[str]]):
    """
    This function loads the spice kernels which have been passed as input

    :parameter kernel_dir: base folder path of kernels
    :parameter list_of_kernel: list of kernels from base folder
    """
    meta_kernel = []
    for kernel_name in list_of_kernel:
        meta_kernel.append(kernel_dir + kernel_name)
    spice.furnsh(meta_kernel)
    return


def close_spice():
    """
    This function close the all the spice kernels
    """
    spice.kclear()
    return


def utc2et(utc: str):
    """
    This function return the Ephemeris Time (ET) of the given Coordinated Universal Time (UTC) epoch
    NOTE: UTC corresponds to the Greenwich Mean Time (GMT)
    """
    return spice.str2et(utc)


def get_ephem_state(target_id: str, epoch: Union[np.ndarray, float], observer_id: str, ref_frame='J2000',
                    correction='NONE'):
    """
    This function return the ephemeris position and velocity of the target celestial object (spacecraft, planet,
    star, satellite, asteroid, ...) as seen in the given reference frame centered at the observer location
    """
    ephem_state, one_way_light_time = spice.spkezr(target_id, epoch, ref_frame, correction, observer_id)
    return ephem_state, one_way_light_time


def get_ephem_position(target_id: str, epoch: Union[np.ndarray, float], observer_id: str, ref_frame='J2000',
                       correction='NONE'):
    """
    This function return the ephemeris position of the target celestial object (spacecraft, planet,
    star, satellite, asteroid, ...) as seen in the given reference frame centered at the observer location
    """
    ephem_pos, one_way_light_time = spice.spkpos(target_id, epoch, ref_frame, correction, observer_id)
    return ephem_pos, one_way_light_time
