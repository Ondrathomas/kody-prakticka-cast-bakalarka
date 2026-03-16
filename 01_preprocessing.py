import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import holidays

# Strojové učení a předzpracování
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Hluboké učení
import tensorflow as tf
from tensorflow.keras.utils import to_categorical



# 2. TRANSFORMACE ČASOVÉ ŘADY A ANALÝZA KVALITY
# Převod v_ts z milisekund na standardní datetime (Korekce časové reprezentace)
df_raw['datetime'] = pd.to_datetime(df_raw['v_ts'], unit='ms')

# Extrakce roku pro ověření kompletnosti dat v jednotlivých letech
df_raw['rok'] = df_raw['datetime'].dt.year

# 3. DIAGNOSTIKA DATASETU PŘED PREPROCESSINGEM
print("-" * 30)
print(f"Celkový počet záznamů v paměti: {len(df_raw):,}")

# Identifikace duplicit v surovém exportu (podklad pro budoucí preprocessing)
duplicity = df_raw.duplicated().sum()
print(f"Počet nalezených duplicitních řádků: {duplicity:,}")
print(f"Procentuální podíl duplicit: {(duplicity / len(df_raw)) * 100:.4f} %")

# Přehled naměřených intervalů v letech (podklad pro Tabulku 2.4 v práci)
print("\nRozdělení počtu záznamů podle let:")
print(df_raw['rok'].value_counts().sort_index())

# Kontrolní výpis transformovaných dat
print("\nUkázka datasetu po úpravě času:")
print(df_raw[['v_ts', 'datetime', 'id_numeric', 'profile', 'anonymized_value']].head())

