import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, GCRS, ITRS, CartesianRepresentation

# v Bad practice
import warnings
warnings.filterwarnings('ignore')   
# ^ Not really advisable to do, in general

peyton_hall = EarthLocation(lat=40.346422*u.deg, lon=-74.651648*u.deg, height=43*u.m) # Peyton Hall, ground level
utc_offset = -4 * u.hour # Eastern Daylight Time


# ENTER TIMES OF OBSERVATIONS (one for each alt/az measurement)
times = Time(['2023-10-4 17:05:30', '2023-10-4 17:10:30']) - utc_offset
alt_angles = u.deg * [1, 17]                               # ENTER ALTITUDE (ELEVATION) ANGLE (degrees above your horizon)
az_angles = u.deg * [313,26]                               # ENTER AZIMUTH ANGLE (degrees clockwise from your geographic north)
obs_dist = u.km * [2235,1128]                               # ENTER RANGE (km distance b/w you and the satellite at time of obsv.)

sat_altaz = AltAz(alt = alt_angles, az = az_angles, distance=obs_dist, obstime = times, location = peyton_hall)
sat_GCRS = sat_altaz.transform_to(GCRS(obstime=times))


print("#   RA (deg)       Dec (deg)      Range (km)")
for i in range(len(alt_angles)):
    ra_deg = sat_GCRS.ra[i].deg
    dec_deg = sat_GCRS.dec[i].deg
    rng_km = sat_GCRS.distance[i].to(u.km).value
    print(f"{i+1}   {ra_deg:10.6f}   {dec_deg:10.6f}   {rng_km:10.3f}") 
