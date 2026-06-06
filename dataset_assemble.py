import os
import io
import warnings
import pandas as pd

# Silenciar avisos de depreciação futuros do Pandas para deixar o terminal limpo
warnings.filterwarnings("ignore", category=FutureWarning)

# Descobre automaticamente a pasta onde o script está salvo
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos absolutos baseados na sua estrutura de pastas
base_path = os.path.join(script_dir, "data")
arg_path = os.path.join(base_path, "ana", "arg.csv")
copernicus_path = os.path.join(base_path, "ERA5 - Copernicus", "copernicus.csv")
oni_path = os.path.join(base_path, "ONI (El Niño)", "oni.csv")

# Validação para garantir que o script encontra os arquivos físicos
for nome, caminho in [("ANA (arg.csv)", arg_path), ("Copernicus", copernicus_path), ("ONI", oni_path)]:
    if not os.path.exists(caminho):
        print(f"❌ ERRO: O arquivo do {nome} não foi encontrado no caminho:")
        print(f"   -> {caminho}")
        print("Verifique se as pastas estão no local correto.\n")
        exit()

print("✔ Arquivos localizados com sucesso! Iniciando o tratamento dos dados...")

# 1. TRATAMENTO DO DATASET ANA (arg.csv)
with open(arg_path, "r", encoding="utf-8-sig", errors="ignore") as f:
    html_content = f.read()

df_ana_list = pd.read_html(io.StringIO(html_content))
df_ana = df_ana_list[0]

# Limpar espaços em branco dos nomes das colunas
df_ana.columns = df_ana.columns.str.strip()

# Limpar espaços em branco de todas as colunas que possuem dados textuais
colunas_texto = df_ana.select_dtypes(include=["object", "string"]).columns
for col in colunas_texto:
    df_ana[col] = df_ana[col].astype(str).str.strip()

# Converter os números do padrão brasileiro (vírgula) para o padrão do Python (ponto)
colunas_numericas = ["Capacidade (hm³)", "Cota (m)", "Volume Útil (hm³)", "Volume (%)"]
for col in colunas_numericas:
    df_ana[col] = df_ana[col].astype(str).str.replace(",", ".").astype(float)

# Padronizar a Data para o formato universal do Pandas
df_ana["Data"] = pd.to_datetime(df_ana["Data da Medição"], format="%d/%m/%Y")
df_ana = df_ana.drop(columns=["Data da Medição"])


# 2. TRATAMENTO DO DATASET COPERNICUS (copernicus.csv)
# Carrega o arquivo e força a primeira coluna (que estava sem nome) a se chamar "Data"
df_copernicus = pd.read_csv(copernicus_path)

if df_copernicus.columns[0].startswith("Unnamed") or df_copernicus.columns[0] == "":
    df_copernicus = df_copernicus.rename(columns={df_copernicus.columns[0]: "Data"})

# Converte para data ignorando erros ou valores corrompidos (transforma em NaT se houver erro e depois removemos)
df_copernicus["Data"] = pd.to_datetime(df_copernicus["Data"], format="%Y-%m-%d", errors="coerce")

# Remove linhas caso alguma data tenha vindo completamente inválida/vazia
df_copernicus = df_copernicus.dropna(subset=["Data"])

# Remove a coluna 'ano' duplicada se ela existir
if "ano" in df_copernicus.columns:
    df_copernicus = df_copernicus.drop(columns=["ano"])


# 3. TRATAMENTO DO DATASET ONI (oni.csv)
df_oni = pd.read_csv(oni_path)

# Mapeia o trimestre móvel para o mês central correspondente
mapa_mes_central = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
}
df_oni["mes"] = df_oni["SEAS"].map(mapa_mes_central)
df_oni = df_oni.rename(columns={"YR": "ano", "TOTAL": "oni_total", "ANOM": "oni_anomala"}).drop(columns=["SEAS"])


# 4. UNIFICAÇÃO (MERGE)
# Mesclar ANA e Copernicus por Data
df_merged = pd.merge(df_ana, df_copernicus, on="Data", how="inner")

# Criar chaves temporárias de ano/mês para cruzar com o ONI (que é mensal/trimestral)
df_merged["ano"] = df_merged["Data"].dt.year
df_merged["mes"] = df_merged["Data"].dt.month

df_final = pd.merge(df_merged, df_oni, on=["ano", "mes"], how="left")

# Limpar as chaves repetidas e ordenar as colunas
df_final = df_final.drop(columns=["ano", "mes"])

# Identificar todas as colunas disponíveis para reordenar sem perder nada
colunas_finais = ["Data", "Estado", "Nome", "Capacidade (hm³)", "Cota (m)", 
                  "Volume Útil (hm³)", "Volume (%)", "precipitacao", 
                  "temperatura", "oni_total", "oni_anomala"]
# Garante que só vai ordenar colunas que realmente existem no DataFrame final
colunas_ordenadas = [c for c in colunas_finais if c in df_final.columns]
df_final = df_final[colunas_ordenadas]

# Salvar o CSV final limpo e padronizado na pasta 'data'
output_path = os.path.join(base_path, "dados_regiao_armando_ribeiro.csv")
df_final.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

print(f"✨ Sucesso absoluto! O arquivo unificado foi gerado em:\n   -> {output_path}")