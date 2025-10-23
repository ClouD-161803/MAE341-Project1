# save as find_elevation_crossings.py
import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
from datetime import datetime, timedelta, timezone
from scipy.optimize import brentq
import pytz

# ---- USER SETTINGS ----
# Put the TLE lines here (update to latest if you fetched fresh)
tle_line0 = "STARLINK-3988"
tle_line1 = "1 52625U 22052AD 25290.97929201 .00001463 00000+0 11001-3 0 9997"
tle_line2 = "2 52625 53.2153 189.7745 0001582 88.0844 272.0330 15.08844230189448"

# Observer location (Princeton, NJ) — change if needed
observer_lat = 40.3463        # degrees north
observer_lon = -74.6580       # degrees (West is negative)
observer_elevation_m = 0      # metres

# Target elevation (degrees). Use -80 for your case.
target_elevation_deg = -80.0

# Time window to search (UTC)
# Choose window around approximate nadir time found in screenshots:
approx_utc = datetime(2025,10,21,20,54,31, tzinfo=timezone.utc)
window_minutes = 20   # +/- minutes around approx_utc to search
t0 = approx_utc - timedelta(minutes=window_minutes)
t1 = approx_utc + timedelta(minutes=window_minutes)
dt_seconds = 1.0  # sampling resolution in seconds for initial bracket search

# ---- END USER SETTINGS ----

# Load timescale
ts = load.timescale()

# Build satellite object
sat = EarthSatellite(tle_line1, tle_line2, tle_line0, ts)

# Build observer location (topos)
observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)

# helper: compute topocentric elevation (degrees) at a given UTC datetime
def elevation_at_datetime(dt_utc):
    t = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond/1e6)
    difference = sat - observer
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees, az.degrees, distance.km, sat.at(t).subpoint().latitude.degrees, sat.at(t).subpoint().longitude.degrees

# Sample across the window to find sign changes around target elevation
seconds = int((t1 - t0).total_seconds())
times = [t0 + timedelta(seconds=i) for i in range(0, seconds+1, int(max(1, dt_seconds)))]
elevs = [elevation_at_datetime(tt)[0] for tt in times]

# find intervals where elevation crosses target
cross_intervals = []
for i in range(len(times)-1):
    e1 = elevs[i] - target_elevation_deg
    e2 = elevs[i+1] - target_elevation_deg
    if e1 == 0.0:
        cross_intervals.append((times[i], times[i]))
    elif e1 * e2 < 0:
        cross_intervals.append((times[i], times[i+1]))

# refine each crossing using root finding (brentq) on a continuous function of seconds
def elev_seconds(s):
    dt = t0 + timedelta(seconds=s)
    alt, az, dist, lat, lon = elevation_at_datetime(dt)
    return alt - target_elevation_deg

results = []
for (ta, tb) in cross_intervals:
    # seconds from t0
    sa = (ta - t0).total_seconds()
    sb = (tb - t0).total_seconds()
    try:
        root_s = brentq(elev_seconds, sa, sb, xtol=1e-3)
        dt_root = t0 + timedelta(seconds=float(root_s))
        alt, az, dist, lat, lon = elevation_at_datetime(dt_root)
        results.append({
            "utc": dt_root.replace(tzinfo=timezone.utc),
            "elevation_deg": alt,
            "azimuth_deg": az,
            "distance_km": dist,
            "subsat_lat_deg": lat,
            "subsat_lon_deg": lon
        })
    except Exception as e:
        # fallback: midpoint
        dt_mid = ta + (tb - ta)/2
        alt, az, dist, lat, lon = elevation_at_datetime(dt_mid)
        results.append({
            "utc": dt_mid.replace(tzinfo=timezone.utc),
            "elevation_deg": alt,
            "azimuth_deg": az,
            "distance_km": dist,
            "subsat_lat_deg": lat,
            "subsat_lon_deg": lon,
            "note": f"rootfinder failed: {e}"
        })

# Print results (UTC and local EDT)
eastern = pytz.timezone('US/Eastern')
print("Found {} crossings of elevation {}° between {} and {}".format(len(results), target_elevation_deg, t0.isoformat(), t1.isoformat()))
for r in results:
    utc = r["utc"]
    local = utc.astimezone(eastern)
    print("UTC: {}, Local (EDT): {}".format(utc.strftime("%Y-%m-%d %H:%M:%S"), local.strftime("%Y-%m-%d %H:%M:%S")) )
    print("  Elev (deg): {:.3f}, Az (deg): {:.3f}, Dist (km): {:.3f}".format(r["elevation_deg"], r["azimuth_deg"], r["distance_km"]))
    print("  Sub-satellite lat (°N): {:.5f}, lon (°E): {:.5f}".format(r["subsat_lat_deg"], r["subsat_lon_deg"]))
    if "note" in r:
        print("  NOTE:", r["note"])
    print()
