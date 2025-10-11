import numpy as np

# WGS84 ellipsoid constants
a = 6378137.0           # Semi-major axis in meters
f = 1 / 298.257223563   # Flattening
e2 = f * (2 - f)        # Square of eccentricity

def lla_to_ecef(lat, lon, alt):
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = a / np.sqrt(1 - e2 * (np.sin(lat)**2))
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e2) * N + alt) * np.sin(lat)
    return np.array([x, y, z])

# Realistic GPS position (degrees, meters)
latitude = 37.4275
longitude = -122.1697
altitude = 30.0

pos_init_ecef = lla_to_ecef(latitude, longitude, altitude)

# Save to CSV
np.savetxt('pos_init.csv', pos_init_ecef.reshape(1, -1), delimiter=',')

print("pos_init.csv generated with ECEF coordinates:", pos_init_ecef)

