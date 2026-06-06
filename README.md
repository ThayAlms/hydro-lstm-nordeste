> Volume forecasting and anomaly detection for the Armando Ribeiro Gonçalves Reservoir (Açu — RN, Brazil) using Bidirectional LSTM and Isolation Forest.

---

## About

The **Armando Ribeiro Gonçalves Reservoir**, located on the Piranhas-Açu River in the semiarid region of Rio Grande do Norte, is one of the largest reservoirs in northeastern Brazil, with a capacity of 2.4 billion m³. Its management is critical for human supply, agriculture, and energy generation across the region.

This project applies machine learning to historical reservoir data (1987–2023) with two main goals:

1. **Forecast future volume** using a Bidirectional LSTM, fed by volume history, satellite precipitation, and the ONI climate index (El Niño/La Niña).
2. **Detect anomalous months** in the historical series using Isolation Forest, flagging extreme droughts and flood events that deviate from expected seasonal behavior.

---

## Results

| Metric | Value |
|--------|-------|
| MAE (Test set) | **2.68%** |
| RMSE (Test set) | **3.20%** |
| Correlation (r) | **0.878** |
| Training epochs | 286 (early stopping) |
| Anomalies detected | 31 months out of 442 (7%) |

---

## Project structure

```
piranhas-ml/
│
├── data/
│   ├── ana/
│   │   └── arg.csv                         # Daily volume data (ANA)
│   ├── ERA5 - Copernicus/
│   │   └── copernicus.csv                  # Precipitation & temperature (ERA5)
│   ├── ONI (El Niño)/
│   │   └── oni.csv                         # ONI index — NOAA
│   └── dados_regiao_armando_ribeiro.csv     # Unified dataset (generated)
│
├── graficos/
│   ├── 01_loss_convergencia.png
│   ├── 02_predicao_vs_real.png
│   ├── 03_historico_oni.png
│   ├── 04_dispersao_residuos.png
│   ├── 05_projecao_cenarios.png
│   ├── 06_anomalias_serie_temporal.png
│   ├── 07_mapa_calor_anomalias.png
│   ├── 08_dispersao_features.png
│   ├── 09_top_eventos_extremos.png
│   └── anomalias.txt
│
├── dataset_assemble.py      # Merges ANA + Copernicus + ONI
├── train_model.py           # Trains the BiLSTM and generates forecasts
├── anomaly_detection.py     # Isolation Forest over the historical series
├── analise.txt              # Full technical report
└── README.md
```

---

## Pipeline

```
ANA (arg.csv) ──┐
                ├──► dataset_assemble.py ──► dados_regiao_armando_ribeiro.csv
Copernicus ─────┤                                        │
                │                                        ▼
ONI ────────────┘                 ┌──────────── train_model.py
                                  │                  (BiLSTM)
                                  │             MAE: 2.68% | r: 0.878
                                  │
                                  └──────────── anomaly_detection.py
                                                  (Isolation Forest)
                                                  31 anomalies detected
```

---

## Data sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Daily volume | [ANA — National Water Agency](https://www.snirh.gov.br/hidrotelemetria/) | Stage, useful volume, and % capacity |
| Precipitation & temperature | [ERA5 — Copernicus Climate Change Service](https://cds.climate.copernicus.eu/) | Monthly atmospheric reanalysis |
| ONI index | [NOAA — Climate Prediction Center](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php) | Equatorial Pacific sea surface temperature anomaly |

---

## Models

### Bidirectional LSTM (`train_model.py`)

A recurrent neural network that learns temporal patterns in both directions of the series. It takes a 12-month sliding window with 3 features (volume, precipitation, ONI) and predicts the volume of the following month.

```
Input (12 × 3)
    └─► BiLSTM(64) + Dropout(0.25)
            └─► LSTM(32) + Dropout(0.20)
                    └─► Dense(16, relu)
                            └─► Dense(1) → Volume (%)
```

Regularization: L2 (1e-4) on all recurrent layers.  
Callbacks: EarlyStopping (patience=60) + ReduceLROnPlateau (factor=0.5, patience=25).

**Temporal split:**

| Set | Period | Samples |
|-----|--------|---------|
| Train | 1983 – 2018 | 372 |
| Validation | 2019 – 2021 | 36 |
| Test | 2022 – 2024 | 24 |

### Isolation Forest (`anomaly_detection.py`)

Detects months that behave in isolation within the feature space. Beyond the 3 raw features, engineered features are added to amplify the anomaly signal:

- **ΔVolume** — month-over-month volume change
- **Seasonal deviation** — difference between the month's volume and its historical monthly mean
- **Precipitation × ONI** — interaction between local and macro-climate drivers

Configuration: 200 trees, contamination 7%.

---

## Key findings

**Extreme droughts and El Niño**  
The 2015/2016 El Niño produced the most critical period on record: 6 of the 10 most anomalous months belong to that episode, with ONI above +2.0 and reservoir volume reaching a historic low of 20.5% in January 2016. The mechanism is the northward displacement of the Intertropical Convergence Zone (ITCZ), which deprives the semiarid region of rainfall from March to June.

**El Niño ↔ La Niña asymmetry**  
Strong La Niña does not necessarily produce proportional flooding. The 2011 flood event (ONI −1.0) confirmed the relationship, but the high-water events of 1987 and 1992 occurred under moderate El Niño conditions — suggesting that local precipitation and mesoscale systems carry more weight than the ONI index alone in flood events.

**Seasonality of anomalies**  
45% of detected anomalies are concentrated in March and April, the peak of the rainy season. This is when variability is highest: the presence or absence of rainfall in this window determines the reservoir's behavior throughout the second half of the year.

**2024 forecasts**  
With the reservoir at ~55% at the end of 2023, the difference between extreme scenarios (strong La Niña vs. strong El Niño) is only 2.92 percentage points for June 2024. Recent volume is the dominant short-term predictor; the ONI impact amplifies over longer horizons and when the reservoir is below 30%.

---

## Usage

### 1. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### 2. Build the dataset

```bash
python dataset_assemble.py
```

### 3. Train the model and generate forecasts

```bash
python train_model.py
```

### 4. Run anomaly detection

```bash
python anomaly_detection.py
```

Charts are saved automatically to `graficos/` and the technical report to `analise.txt`.

---

## Requirements

```
Python       >= 3.9
TensorFlow   >= 2.10
scikit-learn >= 1.2
pandas       >= 1.5
numpy        >= 1.23
matplotlib   >= 3.6
```

---

## Academic context

Project developed as part of a **Neural Networks and Deep Learning** course, focused on hydrological time series applications in the Brazilian semiarid context and climate variability associated with the El Niño–Southern Oscillation (ENSO) phenomenon.

---

## License

MIT
