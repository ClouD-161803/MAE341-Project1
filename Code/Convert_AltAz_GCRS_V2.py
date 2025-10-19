import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, GCRS, ITRS, CartesianRepresentation

# v Bad practice
import warnings
warnings.filterwarnings('ignore')   
# ^ Not really advisable to do, in general

peyton_hall = EarthLocation(lat=40.346422*u.deg, lon=-74.651648*u.deg, height=43*u.m) # Peyton Hall, ground level
utc_offset = -4 * u.hour # Eastern Daylight Time


time = Time('2020-9-16 00:00:00') - utc_offset              # ENTER TIME OF OBSERVATION
alt_angles = u.deg * [29, 65]                               # ENTER ALTITUDE (ELEVATION) ANGLE (degrees above your horizon)
az_angles = u.deg * [325,167]                               # ENTER AZIMUTH ANGLE (degrees clockwise from your geographic north)
obs_dist = u.km * [3137,2085]                               # ENTER RANGE (km distance b/w you and the satellite at time of obsv.)

sat_altaz = AltAz(alt = alt_angles, az = az_angles, distance=obs_dist, obstime = time, location = peyton_hall)
sat_ITRS_topocentric = ITRS(sat_altaz.spherical, obstime=time)
ITRS_geo_coords = sat_ITRS_topocentric.cartesian + peyton_hall.get_itrs(time).cartesian
sat_ITRS_geo = ITRS(ITRS_geo_coords, representation_type='cartesian')
sat_GCRS = sat_ITRS_geo.transform_to(GCRS)



print("    RA (deg)      Dec (deg),   Radius (km)") 
for x in range(len(alt_angles)):
    temp = str(sat_GCRS.spherical[x])[1:-16]
    print(f"{x+1}   {temp:20}") 

