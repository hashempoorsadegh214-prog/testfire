import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# محدوده جغرافیایی استان فارس
MIN_LAT, MAX_LAT = 27.0, 31.6
MIN_LON, MAX_LON = 50.4, 55.6

# شبکه نقاط نمونه‌برداری
lats = np.linspace(MIN_LAT, MAX_LAT, 14)
lons = np.linspace(MIN_LON, MAX_LON, 14)

points = []
fwi_values = []

print("در حال دریافت داده‌های هواشناسی استان فارس...")

for lat in lats:
    for lon in lons:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum&timezone=Asia%2FTehran&forecast_days=2"
            res = requests.get(url, timeout=10).json()
            
            temp = res['daily']['temperature_2m_max'][1]
            rh = res['daily']['relative_humidity_2m_mean'][1]
            wind = res['daily']['wind_speed_10m_max'][1]
            rain = res['daily']['precipitation_sum'][1]
            
            # فرمول FWI
            fwi = (temp * 0.4) + (wind * 0.3) - (rh * 0.15) - (rain * 2)
            if fwi < 0: 
                fwi = 0
                
            points.append((lon, lat))
            fwi_values.append(fwi)
        except Exception:
            continue

points = np.array(points)
fwi_values = np.array(fwi_values)

# درونیابی فضایی پیوسته (RBF)
grid_lon, grid_lat = np.meshgrid(
    np.linspace(MIN_LON, MAX_LON, 500),
    np.linspace(MIN_LAT, MAX_LAT, 500)
)

rbf = Rbf(points[:, 0], points[:, 1], fwi_values, function='multiquadric', smooth=0.1)
grid_fwi = rbf(grid_lon, grid_lat)
grid_fwi = np.clip(grid_fwi, 0, 45)

# ساخت تصویر با شفافیت بالا
plt.figure(figsize=(10, 10), dpi=300)
ax = plt.axes([0, 0, 1, 1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# رنگ‌بندی شفاف (YlOrRd)
cmap = plt.cm.YlOrRd

plt.imshow(
    grid_fwi, 
    extent=[MIN_LON, MAX_LON, MIN_LAT, MAX_LAT], 
    origin='lower', 
    cmap=cmap, 
    vmin=0, 
    vmax=35,
    alpha=0.65
)

plt.savefig('fwi_map.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()

print("فایل fwi_map.png با موفقیت تولید شد.")
