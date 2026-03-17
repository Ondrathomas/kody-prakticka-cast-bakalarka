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



# -- 2. TRANSFORMACE ČASOVÉ ŘADY A ANALÝZA KVALITY
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


# 3. ANALÝZA KONTINUITY A IDENTIFIKACE VÝPADKŮ 
# Definice funkce pro výpočet chybějících 15min intervalů u jednotlivých ID
def analyze_continuity(input_df):
    print("\n" + "-"*30)
    print("Spouštím podrobnou analýzu kontinuity dat...")
    
    # Filtrujeme profil +A15 (hlavní spotřeba) pro konzistenci analýzy
    df_ana = input_df[input_df['profile'] == '+A15'].copy()
    
    results = []
    unique_ids = df_ana['id_numeric'].unique()
    
    for uid in unique_ids:
        # Výběr konkrétního odběrného místa a seřazení podle času
        sub = df_ana[df_ana['id_numeric'] == uid].sort_values('datetime')
        
        # Výpočet časových mezer mezi řádky v hodinách
        diffs = sub['datetime'].diff().dt.total_seconds() / 3600
        
        # Mezera větší než 0.25h (15 min) je považována za výpadek
        gaps = diffs[diffs > 0.25]
        
        num_records = len(sub)
        num_gaps = len(gaps)
        max_gap = gaps.max() if not gaps.empty else 0
        
        # Výpočet teoretické úplnosti (kolik záznamů by tam mělo být od prvního do posledního měření)
        timespan_h = (sub['datetime'].max() - sub['datetime'].min()).total_seconds() / 3600
        expected = (timespan_h * 4) + 1
        
        # Výpočet procentuální ztráty dat
        missing_pct = max(0, (1 - (num_records / expected)) * 100) if expected > 0 else 0
            
        results.append({
            'ID': uid,
            'Počet záznamů': num_records,
            'Výpadek [%]': round(missing_pct, 2),
            'Max. mezera [h]': round(max_gap, 2),
            'Počet mezer': num_gaps
        })

    return pd.DataFrame(results)

# Provedení analýzy na mém datasetu
continuity_res = analyze_continuity(df_raw)

# Seřazení výsledků podle závažnosti výpadků pro identifikaci extrémů
continuity_res = continuity_res.sort_values('Výpadek [%]', ascending=False)

# Výpočet klíčových metrik pro textovou část práce
prumerny_vypadek = continuity_res['Výpadek [%]'].mean()
pocet_budov = len(continuity_res)

# Výpis pro kontrolu ve VS Code a
print("\n--- STATISTIKA KONTINUITY (Top 10 ID s největšími výpadky) ---")
print(continuity_res.head(10))


print(f"Celkový počet analyzovaných objektů: {pocet_budov}")
print(f"Celková průměrná chybovost v datasetu: {prumerny_vypadek:.2f} %")

# Identifikace ID s výpadkem nad 5 % pro zdůvodnění v textu
outliers = continuity_res[continuity_res['Výpadek [%]'] > 5]
print(f"Počet objektů s chybovostí nad 5 %: {len(outliers)}")


# 5.--- ANALÝZA INTEGRITY A KONTINUITY DAT ---

# 1. Definice profilů pro filtraci
cumulative_profiles = ['+E*(m)', '-E*(m)']
main_profile = '+A15'           # Činný odběr
reactive_ind_profile = '+Ri15'  # Induktivní jalovina
reactive_cap_profile = '+Rc15'  # Kapacitní jalovina

# 2. Separace dat
df_cumulative = df_raw[df_raw['profile'].isin(cumulative_profiles)].copy()
df_work = df_raw[~df_raw['profile'].isin(cumulative_profiles)].copy()

# 3. Identifikace výroben (FVE) - hledáme přetoky v profilu -A15
fve_ids = df_work[(df_work['profile'] == '-A15') & (df_work['anonymized_value'] > 0)]['id_numeric'].unique()

# 4. Výpočet statistik pro výpis
count_total = len(df_raw)
count_e = len(df_cumulative)
count_work = len(df_work)
count_a15 = len(df_work[df_work['profile'] == main_profile])
count_ri15 = len(df_work[df_work['profile'] == reactive_ind_profile])
count_rc15 = len(df_work[df_work['profile'] == reactive_cap_profile])

# 5. FORMÁTOVANÝ VÝPIS
print(f"{'Parametr analýzy (Profil)':<45} | {'Hodnota / Počet':>15}")
print("-" * 65)
print(f"{'Celkový počet řádků (surová data)':<45} | {count_total:>15,}")
print(f"{'Vyfiltrované kumulativní stavy (+E*, -E*)':<45} | {count_e:>15,}")
print(f"{'Počet řádků pro výkonovou analýzu (celkem)':<45} | {count_work:>15,}")
print(f"{'  z toho profil +A15 (Činný odběr)':<45} | {count_a15:>15,}")
print(f"{'  z toho profil +Ri15 (Induktivní jalovina)':<45} | {count_ri15:>15,}")
print(f"{'  z toho profil +Rc15 (Kapacitní jalovina)':<45} | {count_rc15:>15,}")
print("-" * 65)
print(f"{'Počet objektů typu „Pouze odběr“':<45} | {len(df_work['id_numeric'].unique()) - len(fve_ids):>15}")
print(f"{'Počet objektů typu „Výrobna (FVE)“':<45} | {len(fve_ids):>15}")
print(f"{'Identifikátory výroben (detekováno dle -A15)':<45} | {sorted(list(map(int, fve_ids)))}")

