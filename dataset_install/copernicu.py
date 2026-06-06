import cdsapi

c = cdsapi.Client()

years = [str(y) for y in range(1985, 2026)]

for year in years:
    print(f"🔄 baixando {year}")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "surface_pressure",
                "evaporation",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": year,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "12:00"],
            "area": [
                -5.0, -37.5,   # norte, oeste
                -6.5, -36.0    # sul, leste (região do açude)
            ],
            "format": "netcdf"
        },
        f"data/copernicus/era5_{year}.nc"
    )

print("✔ download concluído")