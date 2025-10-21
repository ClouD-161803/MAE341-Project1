import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, GCRS

# Coordinates (Peyton Hall)
lat = 40.3463
lon = -74.6580
peyton_hall = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=43*u.m)  # ground level
utc_offset = -4 * u.hour  # EDT

# Observation 1 data
obs1_dates = ['2025-10-19']*9
obs1_times = ['21:18:49','21:19:00','21:20:00','21:21:00','21:22:00','21:23:00','21:24:00','21:25:00','21:25:38']
obs1_az = [287.6, 286.3, 273.1, 262.7, 241.0, 213.6, 190.3, 176.0, 169.3]
obs1_el = [10.2, 11.0, 17.9, 21.8, 26.3, 25.9, 20.2, 13.7, 10.1]
obs1_rng = [1628, 1576, 1226, 1083, 954, 541, 1137, 1419, 1657]

# Observation 2 data
obs2_dates = ['2025-10-20']*9
obs2_times = ['19:31:03','19:32:00','19:33:00','19:34:00','19:35:00','19:36:00','19:37:00','19:38:00','19:38:16']
obs2_az = [319.7, 325.7, 337.5, 6.1, 56.7, 88.6, 101.9, 108.3, 109.5]
obs2_el = [10.5, 17.5, 28.0, 42.2, 43.7, 29.9, 18.6, 11.3, 9.8]
obs2_rng = [1636, 1264, 930, 694, 677, 886, 1216, 1579, 1675]

# Combine data
all_dates = obs1_dates + obs2_dates
all_times = obs1_times + obs2_times
all_az = obs1_az + obs2_az
all_el = obs1_el + obs2_el
all_rng = obs1_rng + obs2_rng

# Time objects
times = Time([f"{d} {t}" for d,t in zip(all_dates, all_times)]) - utc_offset
alt_angles = u.deg * all_el
az_angles = u.deg * all_az
obs_dist = u.km * all_rng

# Transform to GCRS
sat_altaz = AltAz(alt=alt_angles, az=az_angles, distance=obs_dist, obstime=times, location=peyton_hall)
sat_GCRS = sat_altaz.transform_to(GCRS(obstime=times))

# Print results in table format
print("#    Date         Time       Az (deg)   El (deg)   Range (km)    RA (deg)       Dec (deg)")
for i in range(len(times)):
    print(f"{i+1:2} {all_dates[i]} {all_times[i]:>8}  {all_az[i]:8.1f}   {all_el[i]:7.1f}   {all_rng[i]:8.0f}   "
          f"{sat_GCRS.ra[i].deg:10.6f}   {sat_GCRS.dec[i].deg:10.6f}")