"""
╔══════════════════════════════════════════════════════════════════╗
║   DETECÇÃO DE ANOMALIAS — Isolation Forest                      ║
║   Reservatório Engenheiro Armando Ribeiro Gonçalves (Açu)       ║
╚══════════════════════════════════════════════════════════════════╝

Uso standalone:
    python anomaly_detection.py

Também pode ser importado pelo train_model.py:
    from anomaly_detection import detectar_anomalias
    df_monthly['anomalia'] = detectar_anomalias(df_monthly)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "#F8FAFC",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "#E2E8F0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C = {
    "normal":    "#2563EB",
    "anomalia":  "#DC2626",
    "oni_pos":   "#F97316",
    "oni_neg":   "#2563EB",
    "seco":      "#D97706",
    "umido":     "#0891B2",
    "neutro":    "#6B7280",
    "fundo_an":  "#FEF2F2",
    "grade":     "#E2E8F0",
}


# ─── 1. CARREGAR DADOS ───────────────────────────────────────────────────────
def carregar_dados(script_dir: str) -> pd.DataFrame:
    possiveis = [
        os.path.join(script_dir, "data", "dados_regiao_armando_ribeiro.csv"),
        os.path.join(script_dir, "dados_regiao_armando_ribeiro.csv"),
    ]
    data_path = next((p for p in possiveis if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError("Arquivo 'dados_regiao_armando_ribeiro.csv' não encontrado.")

    print(f"✔  Dataset: {data_path}")
    df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
    df["Data"] = pd.to_datetime(df["Data"])
    df.set_index("Data", inplace=True)

    if df["Volume (%)"].max() > 200:
        df["Volume (%)"] /= 100.0

    monthly = df[["Volume (%)", "precipitacao", "oni_anomala"]].resample("ME").mean()
    monthly = monthly.interpolate("linear")
    print(f"✔  {len(monthly)} meses carregados ({monthly.index[0].date()} → {monthly.index[-1].date()})")
    return monthly


# ─── 2. ENGENHARIA DE FEATURES ───────────────────────────────────────────────
def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas que amplificam o sinal de anomalia:
      - variação mensal do volume (ΔVol)
      - média móvel de 3 meses do volume
      - desvio em relação à sazonalidade (volume - média do mês histórico)
      - precipitação × anomalia ONI (interação climatica)
    """
    f = df.copy()

    # Variação absoluta mês a mês
    f["delta_volume"]     = f["Volume (%)"].diff()

    # Desvio sazonal: quanto o volume do mês difere da média histórica daquele mês
    media_mensal          = f.groupby(f.index.month)["Volume (%)"].transform("mean")
    f["desvio_sazonal"]   = f["Volume (%)"] - media_mensal

    # Média móvel 3 meses
    f["vol_mm3"]          = f["Volume (%)"].rolling(3).mean()

    # Interação clima: precipitação × ONI (amplifica eventos extremos combinados)
    f["precip_x_oni"]     = f["precipitacao"] * f["oni_anomala"]

    # Remove NaNs gerados pelas operações
    f = f.dropna()
    return f


# ─── 3. ISOLATION FOREST ────────────────────────────────────────────────────
def detectar_anomalias(
    df: pd.DataFrame,
    contamination: float = 0.07,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Treina Isolation Forest e retorna o DataFrame original enriquecido com:
      - anomalia     : 1 = anômalo, 0 = normal
      - score_if     : score de anomalia (mais negativo = mais anômalo)
      - tipo_anomalia: 'Seca extrema' | 'Excesso hídrico' | 'Normal'

    contamination=0.07 → considera ~7% dos meses como anômalos (~31 de 444).
    """
    feats = construir_features(df)

    COLUNAS_IF = [
        "Volume (%)",
        "precipitacao",
        "oni_anomala",
        "delta_volume",
        "desvio_sazonal",
        "precip_x_oni",
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(feats[COLUNAS_IF])

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=len(COLUNAS_IF),
        random_state=random_state,
    )
    iso.fit(X)

    feats["anomalia"]  = (iso.predict(X) == -1).astype(int)
    feats["score_if"]  = iso.score_samples(X)   # mais negativo = mais anômalo

    # Classificar o tipo de anomalia
    def _tipo(row):
        if row["anomalia"] == 0:
            return "Normal"
        return "Seca extrema" if row["desvio_sazonal"] < 0 else "Excesso hídrico"

    feats["tipo_anomalia"] = feats.apply(_tipo, axis=1)

    n_an = feats["anomalia"].sum()
    n_se = (feats["tipo_anomalia"] == "Seca extrema").sum()
    n_eh = (feats["tipo_anomalia"] == "Excesso hídrico").sum()
    print(f"✔  Anomalias detectadas: {n_an} meses  ({n_se} secas extremas | {n_eh} excessos hídricos)")

    return feats


# ─── 4. GRÁFICOS ─────────────────────────────────────────────────────────────
def plotar(feats: pd.DataFrame, pasta_saida: str):

    normais   = feats[feats["anomalia"] == 0]
    secas     = feats[feats["tipo_anomalia"] == "Seca extrema"]
    excessos  = feats[feats["tipo_anomalia"] == "Excesso hídrico"]

    def save(fig, nome):
        p = os.path.join(pasta_saida, nome)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   💾 {p}")


    # ── G1: Série de Volume com anomalias marcadas ────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

    ax1 = axes[0]
    ax1.fill_between(feats.index, feats["Volume (%)"], alpha=0.12, color=C["normal"])
    ax1.plot(feats.index, feats["Volume (%)"], color=C["normal"], lw=1.5, label="Volume (%)")

    for idx, row in secas.iterrows():
        ax1.axvspan(idx - pd.offsets.MonthBegin(1), idx,
                    alpha=0.25, color=C["seco"], zorder=2)
    for idx, row in excessos.iterrows():
        ax1.axvspan(idx - pd.offsets.MonthBegin(1), idx,
                    alpha=0.25, color=C["umido"], zorder=2)

    ax1.scatter(secas.index,   secas["Volume (%)"],   color=C["seco"],   s=45, zorder=5,
                label=f"Seca extrema ({len(secas)} meses)")
    ax1.scatter(excessos.index, excessos["Volume (%)"], color=C["umido"], s=45, zorder=5,
                marker="^", label=f"Excesso hídrico ({len(excessos)} meses)")

    ax1.set_ylabel("Volume (%)"); ax1.set_ylim(0, 110)
    ax1.set_title("Detecção de Anomalias — Reservatório Açu (Isolation Forest)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=9, loc="lower left")

    # Painel de Anomaly Score
    ax2 = axes[1]
    cores_score = [C["seco"] if t == "Seca extrema" else
                   (C["umido"] if t == "Excesso hídrico" else C["neutro"])
                   for t in feats["tipo_anomalia"]]
    ax2.bar(feats.index, feats["score_if"], color=cores_score, width=25, alpha=0.75)
    ax2.axhline(feats.loc[feats["anomalia"] == 0, "score_if"].min(),
                color=C["anomalia"], lw=1.2, linestyle="--",
                label="Limiar de anomalia")
    ax2.set_ylabel("Score IF\n(mais negativo = mais anômalo)")
    ax2.legend(fontsize=9)

    # Painel ONI
    ax3 = axes[2]
    oni = feats["oni_anomala"]
    ax3.fill_between(oni.index, oni, 0, where=(oni >= 0), color=C["oni_pos"], alpha=0.55, label="El Niño")
    ax3.fill_between(oni.index, oni, 0, where=(oni < 0),  color=C["oni_neg"], alpha=0.55, label="La Niña")
    ax3.axhline(0, color=C["neutro"], lw=0.8)
    ax3.axhline( 0.5, color=C["oni_pos"], lw=0.8, linestyle=":")
    ax3.axhline(-0.5, color=C["oni_neg"], lw=0.8, linestyle=":")
    ax3.set_ylabel("ONI (anomalia °C)"); ax3.set_xlabel("Data")
    ax3.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    save(fig, "06_anomalias_serie_temporal.png")


    # ── G2: Mapa de calor mensal de anomalias ────────────────────────────
    feats_m = feats.copy()
    feats_m["ano"] = feats_m.index.year
    feats_m["mes"] = feats_m.index.month

    pivot_vol = feats_m.pivot_table(index="ano", columns="mes", values="Volume (%)", aggfunc="mean")
    pivot_an  = feats_m.pivot_table(index="ano", columns="mes", values="anomalia",  aggfunc="max")

    meses_abr = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                              gridspec_kw={"width_ratios": [3, 1]})

    ax1 = axes[0]
    im = ax1.imshow(pivot_vol.values, aspect="auto", cmap="RdYlBu",
                    vmin=0, vmax=100, interpolation="nearest")

    # Marca as anomalias com X
    anos_list = list(pivot_vol.index)
    for i, ano in enumerate(anos_list):
        for j, mes in enumerate(range(1, 13)):
            try:
                if pivot_an.loc[ano, mes] == 1:
                    tipo = feats_m.loc[
                        (feats_m["ano"] == ano) & (feats_m["mes"] == mes),
                        "tipo_anomalia"
                    ].values
                    cor = C["seco"] if len(tipo) > 0 and tipo[0] == "Seca extrema" else C["umido"]
                    ax1.text(j, i, "✕", ha="center", va="center",
                             fontsize=7, color=cor, fontweight="bold")
            except KeyError:
                pass

    ax1.set_xticks(range(12)); ax1.set_xticklabels(meses_abr, fontsize=9)
    ax1.set_yticks(range(len(anos_list)))
    ax1.set_yticklabels(anos_list, fontsize=7)
    ax1.set_title("Mapa de Volume (%) por Mês/Ano\n(✕ = anomalia detectada)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mês"); ax1.set_ylabel("Ano")
    plt.colorbar(im, ax=ax1, label="Volume (%)", shrink=0.8)

    # Frequência de anomalias por mês
    ax2 = axes[1]
    freq_mes = feats_m.groupby("mes")["anomalia"].sum()
    bars = ax2.barh(range(12), freq_mes.values,
                    color=[C["seco"] if v > freq_mes.mean() else C["neutro"] for v in freq_mes.values],
                    alpha=0.8)
    ax2.set_yticks(range(12)); ax2.set_yticklabels(meses_abr, fontsize=9)
    ax2.set_xlabel("Qtd. de anomalias"); ax2.invert_yaxis()
    ax2.set_title("Anomalias\npor Mês", fontsize=11, fontweight="bold")
    ax2.axvline(freq_mes.mean(), color=C["anomalia"], lw=1.2,
                linestyle="--", label=f"Média ({freq_mes.mean():.1f})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "07_mapa_calor_anomalias.png")


    # ── G3: Dispersão 3D projetada — Volume × Precipitação × ONI ─────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pares = [
        ("Volume (%)",    "precipitacao", "Volume (%) vs Precipitação"),
        ("Volume (%)",    "oni_anomala",  "Volume (%) vs ONI"),
        ("precipitacao",  "oni_anomala",  "Precipitação vs ONI"),
    ]

    for ax, (cx, cy, titulo) in zip(axes, pares):
        ax.scatter(normais[cx],  normais[cy],
                   color=C["normal"], alpha=0.35, s=18, label="Normal", zorder=3)
        ax.scatter(secas[cx],    secas[cy],
                   color=C["seco"],   alpha=0.85, s=55, label="Seca extrema",
                   edgecolors="white", lw=0.5, zorder=5)
        ax.scatter(excessos[cx], excessos[cy],
                   color=C["umido"],  alpha=0.85, s=55, marker="^",
                   label="Excesso hídrico", edgecolors="white", lw=0.5, zorder=5)
        ax.set_xlabel(cx); ax.set_ylabel(cy)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Espaço de Features — Anomalias Isolation Forest",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "08_dispersao_features.png")


    # ── G4: Linha do tempo de eventos extremos anotada ───────────────────
    anomalos = feats[feats["anomalia"] == 1].copy()
    anomalos = anomalos.sort_values("score_if")    # mais anômalos primeiro
    top_n = min(15, len(anomalos))
    top   = anomalos.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(feats.index, feats["Volume (%)"], alpha=0.10, color=C["normal"])
    ax.plot(feats.index, feats["Volume (%)"], color=C["normal"], lw=1.5)

    for idx, row in top.iterrows():
        cor  = C["seco"] if row["tipo_anomalia"] == "Seca extrema" else C["umido"]
        ax.scatter(idx, row["Volume (%)"], color=cor, s=70, zorder=6, edgecolors="white", lw=0.8)
        label = idx.strftime("%m/%Y")
        offset = 4 if row["Volume (%)"] < 50 else -12
        ax.annotate(label, xy=(idx, row["Volume (%)"]),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=6.5, color=cor, fontweight="bold")

    legenda = [
        Patch(color=C["seco"],  label="Seca extrema"),
        Patch(color=C["umido"], label="Excesso hídrico"),
    ]
    ax.legend(handles=legenda, fontsize=9, loc="lower left")
    ax.set_ylabel("Volume (%)"); ax.set_xlabel("Data")
    ax.set_title(f"Top {top_n} Eventos Extremos Detectados (Isolation Forest)",
                 fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    save(fig, "09_top_eventos_extremos.png")


# ─── 5. RELATÓRIO TXT ────────────────────────────────────────────────────────
def gerar_relatorio(feats: pd.DataFrame, pasta_saida: str):
    secas    = feats[feats["tipo_anomalia"] == "Seca extrema"].sort_values("score_if")
    excessos = feats[feats["tipo_anomalia"] == "Excesso hídrico"].sort_values("score_if")

    # Frequência por mês do ano
    freq_mes = feats.groupby(feats.index.month)["anomalia"].sum()
    mes_critico = freq_mes.idxmax()
    MESES = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
             7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}

    # Correlação anomalia × ONI
    corr_oni = feats["anomalia"].corr(feats["oni_anomala"].abs())

    linhas = [
        "=" * 69,
        "   RELATÓRIO DE ANOMALIAS — ISOLATION FOREST · AÇUDE ARMANDO RIBEIRO",
        "=" * 69,
        "",
        "1. CONFIGURAÇÃO DO MODELO",
        "-" * 69,
        "  Algoritmo      : Isolation Forest (sklearn)",
        "  N° de árvores  : 200",
        "  Contaminação   : 7% (fração esperada de anomalias)",
        "  Features usadas: Volume (%), Precipitação, ONI, ΔVolume,",
        "                   Desvio Sazonal, Precipitação × ONI",
        "",
        "2. RESUMO DOS RESULTADOS",
        "-" * 69,
        f"  Total de meses analisados : {len(feats)}",
        f"  Anomalias detectadas      : {feats['anomalia'].sum()}",
        f"  → Secas extremas          : {len(secas)}",
        f"  → Excessos hídricos       : {len(excessos)}",
        f"  Mês mais crítico histórico: {MESES[mes_critico]} ({int(freq_mes[mes_critico])} ocorrências)",
        f"  Correlação anomalia × |ONI|: {corr_oni:.3f}",
        "",
        "3. TOP 10 SECAS EXTREMAS",
        "-" * 69,
    ]

    for _, row in secas.head(10).iterrows():
        linhas.append(
            f"  {row.name.strftime('%m/%Y')}  |  Vol: {row['Volume (%)']:.1f}%  "
            f"|  Precip: {row['precipitacao']:.2f} mm  "
            f"|  ONI: {row['oni_anomala']:+.2f}  "
            f"|  Score: {row['score_if']:.4f}"
        )

    linhas += [
        "",
        "4. TOP 10 EXCESSOS HÍDRICOS",
        "-" * 69,
    ]

    for _, row in excessos.head(10).iterrows():
        linhas.append(
            f"  {row.name.strftime('%m/%Y')}  |  Vol: {row['Volume (%)']:.1f}%  "
            f"|  Precip: {row['precipitacao']:.2f} mm  "
            f"|  ONI: {row['oni_anomala']:+.2f}  "
            f"|  Score: {row['score_if']:.4f}"
        )

    linhas += [
        "",
        "5. FREQUÊNCIA DE ANOMALIAS POR MÊS DO ANO",
        "-" * 69,
    ]
    for m, v in freq_mes.items():
        barra = "█" * int(v)
        linhas.append(f"  {MESES[m]:<12}: {barra} ({int(v)})")

    linhas += [
        "",
        "6. INTEGRAÇÃO COM LSTM",
        "-" * 69,
        "  O score_if e o flag anomalia podem ser usados como features",
        "  adicionais no train_model.py para melhorar a LSTM em períodos",
        "  extremos. Importe com:",
        "    from anomaly_detection import carregar_dados, detectar_anomalias",
        "    feats = detectar_anomalias(df_monthly)",
        "    df_monthly['anomalia'] = feats['anomalia']",
        "    df_monthly['score_if'] = feats['score_if']",
        "",
        "7. ANÁLISE DO ESPECIALISTA",
        "-" * 69,
        "  [Comente os eventos de seca identificados — 2012, 2016?]",
        "  [Há relação clara entre secas extremas e El Niño/La Niña?]",
        "  [Os excessos hídricos coincidem com La Niñas fortes?]",
        "",
        "=" * 69,
    ]

    path = os.path.join(pasta_saida, "anomalias.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"   📝 {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    pasta_saida = os.path.join(script_dir, "graficos")
    os.makedirs(pasta_saida, exist_ok=True)

    df_monthly = carregar_dados(script_dir)
    feats      = detectar_anomalias(df_monthly, contamination=0.07)
    plotar(feats, pasta_saida)
    gerar_relatorio(feats, pasta_saida)

    print("\n✅ Detecção de anomalias concluída.")
    print(f"   Gráficos: {pasta_saida}")