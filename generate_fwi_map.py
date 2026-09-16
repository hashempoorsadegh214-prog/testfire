import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# ۱. تعریف دامنه جغرافیایی استان فارس
MIN_LAT, MAX_LAT = 27.0, 31.6
MIN_LON, MAX_LON = 50.4, 55.6

# شبکه ایستگاهی/نقاط نمونه‌برداری جهت دریافت داده
lats = np.linspace(MIN_LAT, MAX_LAT, 12)
lons = np.linspace(MIN_LON, MAX_LON, 12)

points = []
fwi_values = []

print("در حال دریافت داده‌های پیش‌بینی هواشناسی استان فارس...")

# ۲. دریافت داده‌های Open-Meteo برای شبکه نقاط
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
        except Exception as e:
            continue

points = np.array(points)
fwi_values = np.array(fwi_values)

# ۳. درونیابی فضایی پیوسته (RBF / Spatial Interpolation)
grid_lon, grid_lat = np.meshgrid(
    np.linspace(MIN_LON, MAX_LON, 400),
    np.linspace(MIN_LAT, MAX_LAT, 400)
)

rbf = Rbf(points[:, 0], points[:, 1], fwi_values, function='multiquadric', smooth=0.1)
grid_fwi = rbf(grid_lon, grid_lat)
grid_fwi = np.clip(grid_fwi, 0, 50) # محدودسازی بازه FWI

# ۴. رندر تصویر رستری شفاف (PNG)
plt.figure(figsize=(10, 10), dpi=300)
ax = plt.axes([0, 0, 1, 1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# پالت رنگی استاندارد FWI (سبز -> زرد -> نارنجی -> قرمز -> عنابی)
cmap = plt.cm.get_cmap('YlOrRd')

plt.imshow(
    grid_fwi, 
    extent=[MIN_LON, MAX_LON, MIN_LAT, MAX_LAT], 
    origin='lower', 
    cmap=cmap, 
    vmin=0, 
    vmax=40,
    alpha=0.65
)

# ذخیره تصویر با پس‌زمینه شفاف
plt.savefig('fwi_map.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()

print("تصویر رستری پهنه‌بندی fwi_map.png با موفقیت تولید شد.")
