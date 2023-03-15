# Python libraries
import os
import spiceypy as spice


# set default base folder of spice kernel data
default_base_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('..', '..', '..', 'spice_kernels')
)
# set default list of spice kernel data
default_kernel_list = 'meta_kernel.tm'


def load_spice_kernel(base_dir=default_base_folder, list_of_kernel=[default_kernel_list]):
    """
    This function loads the spice kernels which have been passed as input

    :parameter base_dir: base folder path of kernels
    :parameter list_of_kernel: list of kernels from base folder
    """
    meta_kernel = []
    for kernel_name in list_of_kernel:
        meta_kernel.append(os.path.join(base_dir, kernel_name))
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


def get_ephem_state(target_id, epoch, observer_id, ref_frame='J2000', correction='NONE'):
    """
    This function return the ephemeris position and velocity of the target celestial object (spacecraft, planet,
    star, satellite, asteroid, ...) as seen in the given reference frame centered at the observer location

    VARIABLE            I/O  DESCRIPTION
    ------------------  ---  --------------------------------------------------
    target_id            I   Target body name.
    epoch                I   Observer epoch.
    ref_frame            I   Reference frame of output state vector.
    correction           I   Aberration correction flag.
    observer_id          I   Observing body name.
    ephem_state          O   State of target.
    one_way_light_time   O   One way light time between observer and target.
    """
    if type(target_id) == str:
        ephem_state, one_way_light_time = spice.spkezr(target_id, epoch, ref_frame, correction, observer_id)
    else:
        ephem_state, one_way_light_time = spice.spkez(target_id, epoch, ref_frame, correction, observer_id)
    return ephem_state, one_way_light_time


def get_ephem_position(target_id, epoch, observer_id, ref_frame='J2000', correction='NONE'):
    """
    This function return the ephemeris position of the target celestial object (spacecraft, planet,
    star, satellite, asteroid, ...) as seen in the given reference frame centered at the observer location
    """
    ephem_pos, one_way_light_time = spice.spkpos(target_id, epoch, ref_frame, correction, observer_id)
    return ephem_pos, one_way_light_time


if __name__ == '__main__':
    load_spice_kernel()

    date = ["1988 June 13, 13:29:48", "1988 June 14, 13:29:48"]
    date_et = utc2et(date)
    # print(date_et)

    rr_sun, lt = get_ephem_position("SUN", date_et, "EARTH", ref_frame='IAU_EARTH', correction='NONE')
    #print(rr_sun)
    print(rr_sun[0])
    print(rr_sun[1])
    close_spice()
