# Daegu-Gyeongbuk-Wildfire-Forecast
A coordinate-based wildfire risk forecasting system for the Gyeongbuk–Daegu region, featuring a newly developed FFDRI_new index, GEE-derived environmental time-series, and seasonal GRU/LSTM models for 1–7 day danger prediction.

## What’s inside this project

### **FFDRI_new (regional fire danger index)**
- Incorporates **DWI, FMI, TMI + NDVI + sunlight_era5**
- Designed to reflect **Gyeongbuk–Daegu’s local vegetation, terrain, and seasonal patterns**
- Achieves higher explanatory power than the **national FFDRI**

---

### **High-resolution time-series dataset (2019–2024)**
- 15 representative wildfire-vulnerable coordinates
- Extracted via **Google Earth Engine** (ERA5-Land, MODIS NDVI, DEM, slope, etc.)

---

### **Seasonal forecasting models**
- **GRU/LSTM** architectures
- **14-day lookback → 7-day multi-step prediction**
- Separate **Spring (Feb–Apr)** and **Fall–Winter (Oct–Dec)** models
- Achieves **Correlation ≈ 0.93**, stable **RMSE ≈ 2.0**

---

### **Case-study validation**
- Verified on real wildfire events  
  e.g., **2025.4.28 Daegu Hamjisan fire**

---

### **Spatial visualization system**
- Generates **regional wildfire danger maps** (Low → Very High)
- Ready for **real-world decision support integration**

