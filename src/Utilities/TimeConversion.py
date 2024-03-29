import math
from math import floor


def jd2GMST(JD):
    """
    This function returns the Greenwich Mean Sidereal Time from a given Julian Date (JD)
    INPUTS            Unit        Description
    :param JD:      Julian Date [days]

    :return: GMST - Greenwich Mean Sidereal Time [hours]

    Reference: http://aa.usno.navy.mil/faq/docs/GAST.php

    Sidereal time is a system of timekeeping based on the rotation of the
    Earth with respect to the fixed stars in the sky. More specifically, it
    is the measure of the hour angle of the vernal equinox. If the hour angle
    is measured with respect to the true equinox, apparent sidereal time is
    being measured. If the hour angle is measured with respect to the mean
    equinox, mean sidereal time is being measured. When the measurements are
    made with respect to the meridian at Greenwich, the times are referred to
    as Greenwich mean sidereal time (GMST) and Greenwich apparent sidereal
    time (GAST).

    Given below is a simple algorithm for computing apparent sidereal time to
    an accuracy of about 0.1 second, equivalent to about 1.5 arcseconds on
    the sky. The input time required by the algorithm is represented as a
    Julian date (Julian dates can be used to determine Universal Time.)

    Let JD be the Julian date of the time of interest.  Let JD0 be the Julian
    date of the previous midnight (0h) UT (the value of JD0 will end in .5
    exactly), and let H be the hours of UT elapsed since that time. Thus we
    have JD = JD0 + H/24.

    For both of these Julian dates, compute the number of days and fraction
    (+ or -) from 2000 January 1, 12h UT, Julian date 2451545.0:
    D = JD - 2451545.0
    D0 = JD0 - 2451545.0

    Then the Greenwich mean sidereal time in hours is
    GMST = 6.697374558 + 0.06570982441908 D0 + 1.00273790935 H + 0.000026 T^2

    where T = D/36525 is the number of centuries since the year 2000; thus
    the last term can be omitted in most applications. It will be necessary
    to reduce GMST to the range 0h to 24h. Setting H = 0 in the above formula
    yields the Greenwich mean sidereal time at 0h UT, which is tabulated in
    The Astronomical Almanac.

    The following alternative formula can be used with a loss of precision of
    0.1 second per century:

    GMST = 18.697374558 + 24.06570982441908 D

    where, as above, GMST must be reduced to the range 0h to 24h. The
    equations for GMST given above are adapted from those given in Appendix A
    of USNO Circular No. 163 (1981).

    The Greenwich apparent sidereal time is obtained by adding a correction
    to the Greenwich mean sidereal time computed above. The correction term
    is called the nutation in right ascension or the equation of the
    equinoxes. Thus,

    GAST = GMST + eqeq.

    The equation of the equinoxes is given as eqeq = ?? cos ? where ??, the
    nutation in longitude, is given in hours approximately by

    ?? ? -0.000319 sin ? - 0.000024 sin 2L

    with ?, the Longitude of the ascending node of the Moon, given as

    ? = 125.04 - 0.052954 D,

    and L, the Mean Longitude of the Sun, given as

    L = 280.47 + 0.98565 D.

    ? is the obliquity and is given as

    ? = 23.4393 - 0.0000004 D.

    The above expressions for ?, L, and ? are all expressed in degrees.

    The mean or apparent sidereal time locally is found by obtaining the
    local longitude in degrees, converting it to hours by dividing by 15, and
    then adding it to or subtracting it from the Greenwich time depending on
    whether the local position is east (add) or west (subtract) of Greenwich.

    If you need apparent sidereal time to better than 0.1 second accuracy on
    a regular basis, consider using the Multiyear Interactive Computer
    Almanac, MICA. MICA provides very accurate almanac data in tabular form
    for a range of years.

    NOTES ON ACCURACY
    The maximum error resulting from the use of the above formulas for
    sidereal time over the period 2000-2100 is 0.432 seconds; the RMS error
    is 0.01512 seconds. To obtain sub-second accuracy in sidereal time, it is
    important to use the form of Universal Time called UT1 as the basis for
    the input Julian date.

    The maximum value of the equation of the equinoxes is about 1.1 seconds,
    so if an error of ~1 second is unimportant, the last series of formulas
    can be skipped entirely. In this case set eqeq = 0 and GAST = GMST, and
    use either UT1 or UTC as the Universal Time basis for the input Julian
    date.
    """
    # Find JD0 with closest midnight
    if (JD - floor(JD)) > 0.5:
        JD0 = floor(JD) + 0.5
    else:
        JD0 = floor(JD) - 0.5

    H = (JD - JD0) * 24     # Time in hours past previous midnight
    D = JD - 2451545.0      # Number of days since J2000
    D0 = JD0 - 2451545.0    # Number of days since J2000
    T = D / 36525           # Number of centuries since J2000

    GMST_unwrapped = 6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H + 0.000026 * T ** 2
    GMST = GMST_unwrapped % 24
    return GMST


def LocalTime(UTC, longitude):
    """
    Compute Local Time (LT) given the corresponding UTC and the longitude of the local position.

    :param UTC:         Coordinated Universal Time
    :param longitude:   Longitude of the local position
    :return LT:         Local Time LT
    """
    LT = (UTC + longitude / 15) % 24
    return LT


def fracday2hms(fracDay):
    m, h = math.modf(fracDay * 24)
    s, m = math.modf(m * 60)
    s = s * 60
    return [h, m, s]


def hms2fracday(hms):
    h, m, s = hms
    fracDay = (h + (m + s / 60) / 60) / 24
    return fracDay


if __name__ == '__main__':
    # sidereal year
    day = 365.256363004 - 365
    # day = 365.242190402 - 365
    hms = fracday2hms(day)
    print(day)
    print(hms)
    print(hms2fracday(hms))


