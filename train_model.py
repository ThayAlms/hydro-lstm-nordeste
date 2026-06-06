"""
╔══════════════════════════════════════════════════════════════════╗
║   LSTM — Reservatório Armando Ribeiro Gonçalves (Açu)           ║
║   Previsão de Volume (%) com janelas temporais deslizantes       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── PALETA DE CORES ────────────────────────────────────────────────────────
C = {
    "azul":       "#2563EB",
    "azul_claro": "#93C5FD",
    "laranja":    "#F97316",
    "verde":      "#16A34A",
    "vermelho":   "#DC2626",
    "roxo":       "#7C3AED",
    "cinza":      "#6B7280",
    "fundo":      "#F8FAFC",
    "grade":      "#E2E8F0",
}

matplotlib.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.facecolor": C["fundo"],
    "figure.facecolor": "white",
    "axes.grid":      True,
    "grid.color":     C["grade"],
    "grid.linestyle": "--",
    "grid.alpha":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ─── 0. PASTAS ───────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
pasta_saida    = os.path.join(script_dir, "graficos")
pasta_relatorio = script_dir

os.makedirs(pasta_saida, exist_ok=True)


# ─── 1. CARREGAR DADOS ───────────────────────────────────────────────────────
possiveis = [
    os.path.join(script_dir, "data", "dados_regiao_armando_ribeiro.csv"),
    os.path.join(script_dir, "dados_regiao_armando_ribeiro.csv"),
]
data_path = next((p for p in possiveis if os.path.exists(p)), None)

if data_path is None:
    raise FileNotFoundError("Arquivo 'dados_regiao_armando_ribeiro.csv' não encontrado.")

print(f"✔  Dataset: {data_path}")
df_raw = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
df_raw["Data"] = pd.to_datetime(df_raw["Data"])
df_raw.set_index("Data", inplace=True)

# Corrige escala (valores > 100 indicam que estão em décimos de %)
if df_raw["Volume (%)"].max() > 200:
    df_raw["Volume (%)"] /= 100.0

FEATURES = ["Volume (%)", "precipitacao", "oni_anomala"]
df_m = df_raw[FEATURES].resample("ME").mean().interpolate("linear")

print(f"✔  Período: {df_m.index[0].date()} → {df_m.index[-1].date()} ({len(df_m)} meses)")


# ─── 2. NORMALIZAÇÃO ─────────────────────────────────────────────────────────
scaler_all    = MinMaxScaler()
scaler_volume = MinMaxScaler()

scaled      = scaler_all.fit_transform(df_m)
scaler_volume.fit(df_m[["Volume (%)"]])


# ─── 3. JANELAS TEMPORAIS ────────────────────────────────────────────────────
LOOKBACK = 12   # meses de histórico
FORECAST  = 1   # passo à frente

def make_windows(data, lookback, forecast):
    X, y = [], []
    for i in range(len(data) - lookback - forecast + 1):
        X.append(data[i : i + lookback, :])
        y.append(data[i + lookback + forecast - 1, 0])   # target = Volume (%)
    return np.array(X), np.array(y)

X, y = make_windows(scaled, LOOKBACK, FORECAST)
dates_y = df_m.index[LOOKBACK:]


# ─── 4. SPLIT CRONOLÓGICO ────────────────────────────────────────────────────
TREINO_FIM = "2018-12-31"
VAL_FIM    = "2021-12-31"
TESTE_FIM  = "2024-12-31"

tr = (dates_y >= "1983-01-01") & (dates_y <= TREINO_FIM)
vl = (dates_y >  TREINO_FIM)  & (dates_y <= VAL_FIM)
te = (dates_y >  VAL_FIM)     & (dates_y <= TESTE_FIM)

X_tr, y_tr = X[tr], y[tr]
X_vl, y_vl = X[vl], y[vl]
X_te, y_te = X[te], y[te]

print(f"   Treino: {tr.sum()} | Validação: {vl.sum()} | Teste: {te.sum()}")


# ─── 5. MODELO LSTM BIDIRECIONAL ─────────────────────────────────────────────
def build_model(n_steps, n_features):
    m = Sequential([
        Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(1e-4)),
                      input_shape=(n_steps, n_features)),
        Dropout(0.25),
        LSTM(32, kernel_regularizer=l2(1e-4)),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

model = build_model(LOOKBACK, X_tr.shape[2])
model.summary()

EPOCHS     = 1000
BATCH_SIZE = 32

callbacks = [
    EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=25, min_lr=1e-6, verbose=1),
]

print(f"\nTreinando (máx {EPOCHS} épocas, early stopping ativo)...")
history = model.fit(
    X_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_vl, y_vl),
    callbacks=callbacks,
    verbose=1,
)
epocas_reais = len(history.history["loss"])
print(f"✔  Convergiu em {epocas_reais} épocas")


# ─── 6. MÉTRICAS NO TESTE ────────────────────────────────────────────────────
pred_sc = model.predict(X_te, verbose=0)
pred    = scaler_volume.inverse_transform(pred_sc).flatten()
real    = scaler_volume.inverse_transform(y_te.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(real, pred)
rmse = np.sqrt(mean_squared_error(real, pred))
corr = np.corrcoef(real, pred)[0, 1]

print(f"\n📊 Teste → MAE: {mae:.2f}%  |  RMSE: {rmse:.2f}%  |  Corr: {corr:.3f}")


# ─── 7. PREVISÃO FUTURA — MÚLTIPLOS CENÁRIOS ONI ─────────────────────────────
def forecast_future(model, last_window, scaler_all, scaler_volume, df_m,
                    oni_value, months=6):
    """Previsão autoregressiva para N meses, dado cenário ONI."""
    oni_min = df_m["oni_anomala"].min()
    oni_max = df_m["oni_anomala"].max()
    oni_sc  = (oni_value - oni_min) / (oni_max - oni_min)

    precip_sc = last_window[-LOOKBACK:, 1].mean()   # média recente de precipitação
    window = last_window.copy()
    preds = []

    for _ in range(months):
        p = model.predict(window.reshape(1, LOOKBACK, window.shape[1]), verbose=0)[0, 0]
        preds.append(p)
        new_step = np.array([[p, precip_sc, oni_sc]])
        window = np.vstack([window[1:], new_step])

    vals = scaler_volume.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return np.clip(vals, 0, 100)

MONTHS_FUTURE = 6
base_window   = scaled[-LOOKBACK:, :]

cenarios = {
    "La Niña forte (−1.5)":  forecast_future(model, base_window, scaler_all, scaler_volume, df_m, -1.5, MONTHS_FUTURE),
    "Neutro (0.0)":           forecast_future(model, base_window, scaler_all, scaler_volume, df_m,  0.0, MONTHS_FUTURE),
    "El Niño moderado (+1.2)":forecast_future(model, base_window, scaler_all, scaler_volume, df_m,  1.2, MONTHS_FUTURE),
    "El Niño forte (+1.8)":   forecast_future(model, base_window, scaler_all, scaler_volume, df_m,  1.8, MONTHS_FUTURE),
}

future_dates = pd.date_range(
    start=df_m.index[-1] + pd.offsets.MonthEnd(1),
    periods=MONTHS_FUTURE, freq="ME"
)


# ─── 8. GRÁFICOS ─────────────────────────────────────────────────────────────

def save(fig, nome):
    p = os.path.join(pasta_saida, nome)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {p}")


# — G1: Curva de Loss ——————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(10, 4))
loss_tr = history.history["loss"]
loss_vl = history.history["val_loss"]
ep = np.arange(1, len(loss_tr) + 1)

ax.plot(ep, loss_tr, color=C["azul"],    lw=1.8, label="Loss Treino (MSE)")
ax.plot(ep, loss_vl, color=C["laranja"], lw=1.8, label="Loss Validação (MSE)", alpha=0.9)
ax.fill_between(ep, loss_tr, loss_vl, alpha=0.07, color=C["laranja"])
ax.axvline(ep[-1], color=C["verde"], lw=1.2, linestyle=":", label=f"Parada (época {ep[-1]})")

best_vl = np.argmin(loss_vl)
ax.scatter(best_vl + 1, loss_vl[best_vl], color=C["verde"], s=80, zorder=5,
           label=f"Melhor val_loss = {loss_vl[best_vl]:.5f}")

ax.set_title("Curva de Convergência — LSTM Bidirecional", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Épocas"); ax.set_ylabel("MSE")
ax.legend(fontsize=9); ax.xaxis.set_major_locator(MaxNLocator(10))
fig.tight_layout()
save(fig, "01_loss_convergencia.png")


# — G2: Predito vs Real no Teste ——————————————————————————————————————————
dates_te = dates_y[te]
residuos = real - pred

fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]})

ax1 = axes[0]
ax1.plot(dates_te, real, color=C["azul"],    lw=2,   label="Volume Real")
ax1.plot(dates_te, pred, color=C["laranja"], lw=2,   label="Previsão LSTM", linestyle="--")
ax1.fill_between(dates_te,
                 np.clip(pred - rmse, 0, 100),
                 np.clip(pred + rmse, 0, 100),
                 color=C["laranja"], alpha=0.15, label=f"±1 RMSE ({rmse:.1f}%)")
ax1.set_title("Predição no Conjunto de Teste (2022–2024)", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Volume (%)"); ax1.legend(fontsize=9)
ax1.text(0.02, 0.05, f"MAE = {mae:.2f}%   RMSE = {rmse:.2f}%   r = {corr:.3f}",
         transform=ax1.transAxes, fontsize=9, color=C["cinza"],
         bbox=dict(facecolor="white", edgecolor=C["grade"], boxstyle="round,pad=0.4"))

ax2 = axes[1]
ax2.bar(dates_te, residuos, color=[C["verde"] if r >= 0 else C["vermelho"] for r in residuos],
        width=20, alpha=0.75)
ax2.axhline(0, color=C["cinza"], lw=1)
ax2.set_ylabel("Resíduo (%)"); ax2.set_xlabel("Data")

fig.tight_layout()
save(fig, "02_predicao_vs_real.png")


# — G3: Série histórica completa + Contexto ONI ———————————————————————————
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

ax1 = axes[0]
ax1.fill_between(df_m.index, df_m["Volume (%)"], alpha=0.2, color=C["azul"])
ax1.plot(df_m.index, df_m["Volume (%)"], color=C["azul"], lw=1.5)

# Marca os períodos de treino / val / teste
ax1.axvspan(pd.Timestamp("1983-01-01"), pd.Timestamp(TREINO_FIM),
            alpha=0.06, color=C["verde"], label="Treino (1983–2018)")
ax1.axvspan(pd.Timestamp(TREINO_FIM), pd.Timestamp(VAL_FIM),
            alpha=0.10, color=C["laranja"], label="Validação (2019–2021)")
ax1.axvspan(pd.Timestamp(VAL_FIM), pd.Timestamp(TESTE_FIM),
            alpha=0.10, color=C["vermelho"], label="Teste (2022–2024)")

ax1.set_ylabel("Volume (%)"); ax1.set_title("Volume Histórico — Açu (1987–2024)", fontsize=13, fontweight="bold", pad=12)
ax1.legend(fontsize=9, loc="lower left")

ax2 = axes[1]
oni = df_m["oni_anomala"]
ax2.fill_between(oni.index, oni, 0,
                 where=(oni >= 0), color=C["laranja"], alpha=0.55, label="El Niño")
ax2.fill_between(oni.index, oni, 0,
                 where=(oni < 0),  color=C["azul"],    alpha=0.55, label="La Niña")
ax2.axhline(0, color=C["cinza"], lw=0.8)
ax2.axhline( 0.5, color=C["laranja"], lw=0.8, linestyle=":")
ax2.axhline(-0.5, color=C["azul"],    lw=0.8, linestyle=":")
ax2.set_ylabel("ONI (anomalia °C)"); ax2.set_xlabel("Data")
ax2.legend(fontsize=9, loc="lower left")

fig.tight_layout()
save(fig, "03_historico_oni.png")


# — G4: Dispersão e Correlação ———————————————————————————————————————————
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.scatter(real, pred, alpha=0.55, s=28, color=C["azul"], edgecolors="none")
lim = [min(real.min(), pred.min()) - 2, max(real.max(), pred.max()) + 2]
ax1.plot(lim, lim, color=C["cinza"], lw=1.2, linestyle="--", label="Ideal")
# Linha de regressão
coef = np.polyfit(real, pred, 1)
linha = np.poly1d(coef)
xs = np.linspace(lim[0], lim[1], 100)
ax1.plot(xs, linha(xs), color=C["laranja"], lw=1.5, label=f"Reg. linear (r={corr:.3f})")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Volume Real (%)"); ax1.set_ylabel("Volume Previsto (%)")
ax1.set_title("Dispersão — Real vs Previsto", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)

ax2 = axes[1]
hist_r = np.histogram(residuos, bins=20)
ax2.hist(residuos, bins=20, color=C["azul"], alpha=0.7, edgecolor="white")
mu, sigma = residuos.mean(), residuos.std()
ax2.axvline(mu, color=C["laranja"], lw=1.5, linestyle="--", label=f"Média = {mu:.2f}%")
ax2.axvline(mu + sigma, color=C["cinza"], lw=1, linestyle=":", label=f"±σ = {sigma:.2f}%")
ax2.axvline(mu - sigma, color=C["cinza"], lw=1, linestyle=":")
ax2.set_xlabel("Resíduo (%)"); ax2.set_ylabel("Frequência")
ax2.set_title("Distribuição dos Resíduos", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)

fig.tight_layout()
save(fig, "04_dispersao_residuos.png")


# — G5: Projeção Futura — Múltiplos Cenários ONI ——————————————————————————
JANELA_HIST = 36   # meses de histórico para contexto no gráfico
hist_dates  = df_m.index[-JANELA_HIST:]
hist_vol    = df_m["Volume (%)"].values[-JANELA_HIST:]

cores_cenarios = {
    "La Niña forte (−1.5)":   C["azul"],
    "Neutro (0.0)":            C["cinza"],
    "El Niño moderado (+1.2)": C["laranja"],
    "El Niño forte (+1.8)":    C["vermelho"],
}

fig, ax = plt.subplots(figsize=(13, 5))

# Histórico recente
ax.fill_between(hist_dates, hist_vol, alpha=0.12, color=C["azul"])
ax.plot(hist_dates, hist_vol, color=C["azul"], lw=2, label="Volume Real (histórico)")
ax.axvline(df_m.index[-1], color=C["cinza"], lw=1.2, linestyle="--", alpha=0.7)

# Cenários
for nome, vals in cenarios.items():
    cor = cores_cenarios[nome]
    ax.plot(future_dates, vals, color=cor, lw=2.2, linestyle="--",
            marker="o", markersize=5, label=nome)
    ax.fill_between(future_dates,
                    np.clip(vals - rmse, 0, 100),
                    np.clip(vals + rmse, 0, 100),
                    color=cor, alpha=0.08)

ax.set_title(f"Projeção de Volume — Açu (próximos {MONTHS_FUTURE} meses, 4 cenários ONI)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Volume (%)"); ax.set_xlabel("Data")
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="upper left")

# Anotação da última previsão de cada cenário
for nome, vals in cenarios.items():
    cor = cores_cenarios[nome]
    ax.annotate(f"{vals[-1]:.1f}%",
                xy=(future_dates[-1], vals[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8.5, color=cor, va="center")

fig.tight_layout()
save(fig, "05_projecao_cenarios.png")


# ─── 9. RELATÓRIO TXT ────────────────────────────────────────────────────────
MESES_PT = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
            7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}

linhas = []
linhas += [
    "=" * 69,
    "    RELATÓRIO TÉCNICO — REDE NEURAL LSTM · AÇUDE ARMANDO RIBEIRO    ",
    "=" * 69,
    "",
    "1. METODOLOGIA",
    "-" * 69,
    f"  Reservatório : Engenheiro Armando Ribeiro Gonçalves (Açu) — RN",
    f"  Arquitetura  : LSTM Bidirecional (BiLSTM)",
    f"  Lookback     : {LOOKBACK} meses | Horizonte: {FORECAST} mês à frente",
    f"  Features     : Volume (%), Precipitação Satelital, Anomalia ONI",
    f"  Épocas reais : {epocas_reais} (early stopping | máx 1000)",
    f"  Batch size   : {BATCH_SIZE}  |  Otimizador: Adam + ReduceLR",
    f"  Regularização: L2 (1e-4) + Dropout (0.25 / 0.20)",
    "",
    "2. DIVISÃO DOS DADOS",
    "-" * 69,
    f"  Treino     : 1983 → 2018 ({tr.sum()} amostras)",
    f"  Validação  : 2019 → 2021 ({vl.sum()} amostras)",
    f"  Teste      : 2022 → 2024 ({te.sum()} amostras — dados inéditos)",
    "",
    "3. DESEMPENHO NO CONJUNTO DE TESTE",
    "-" * 69,
    f"  MAE  : {mae:.2f}%",
    f"  RMSE : {rmse:.2f}%",
    f"  Corr : {corr:.4f}",
    "",
    "4. PROJEÇÕES FUTURAS — CENÁRIOS ONI",
    "-" * 69,
]

for nome, vals in cenarios.items():
    ultimo = vals[-1]
    mes_pt = MESES_PT[future_dates[-1].month]
    linhas.append(f"  [{nome:30s}] → {ultimo:.2f}% em {mes_pt}/{future_dates[-1].year}")

linhas += [
    "",
    "5. GRÁFICOS GERADOS",
    "-" * 69,
    "  01_loss_convergencia.png   — Curva de loss (treino vs validação)",
    "  02_predicao_vs_real.png    — Predito vs Real + resíduos (teste)",
    "  03_historico_oni.png       — Série histórica + índice ONI",
    "  04_dispersao_residuos.png  — Dispersão Real×Previsto + distribuição",
    "  05_projecao_cenarios.png   — Projeção futura (4 cenários ONI)",
    "",
    "6. ANÁLISE DO ESPECIALISTA",
    "-" * 69,
    "  [Convergência do loss — comente aqui]",
    "  [Overfitting / Underfitting — comente aqui]",
    "  [Impacto do El Niño no Açu — comente aqui]",
    "",
    "=" * 69,
]

relatorio_path = os.path.join(pasta_relatorio, "analise.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas))

print(f"\n📝 Relatório: {relatorio_path}")
print(f"\n✅ Pipeline concluído. Gráficos em: {pasta_saida}")