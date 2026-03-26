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



# -- 2. TRANSFORMACE ČASOVÉ ŘADY A ANALÝZA KVALITY --
# Převod v_ts z milisekund na standardní datetime (Korekce časové reprezentace)
df_raw['datetime'] = pd.to_datetime(df_raw['v_ts'], unit='ms')

# Extrakce roku pro ověření kompletnosti dat v jednotlivých letech
df_raw['rok'] = df_raw['datetime'].dt.year

#  3. DIAGNOSTIKA DATASETU PŘED PREPROCESSINGEM 
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

# 4. ANALÝZA KONTINUITY A IDENTIFIKACE VÝPADKŮ 
# Definice funkce pro výpočet chybějících 15min intervalů u jednotlivých ID
def analyze_continuity(input_df):
    print("\n" + "-"*40)
    print("SPOUŠTÍM PODROBNOU ANALÝZU KONTINUITY DAT...")
    print("-"*40)
    
    # Filtrujeme profil +A15 (hlavní spotřeba) pro konzistenci analýzy
    # Pokud v datech profil +A15 nemáš, tento řádek zakomentuj
    df_ana = input_df[input_df['profile'] == '+A15'].copy()
    
    # Pokud by byl df_ana prázdný (např. jiný název profilu), použijeme celý df
    if df_ana.empty:
        df_ana = input_df.copy()
    
    results = []
    unique_ids = df_ana['id_numeric'].unique()
    
    for uid in unique_ids:
        # Výběr konkrétního odběrného místa a seřazení podle času
        sub = df_ana[df_ana['id_numeric'] == uid].sort_values('datetime')
        
        # Výpočet časových mezer mezi po sobě jdoucími řádky (v hodinách)
        # diff() spočítá rozdíl mezi řádkem n a n-1
        diffs = sub['datetime'].diff().dt.total_seconds() / 3600
        
        # Mezera větší než 0.25h (15 min) je považována za výpadek v měření
        gaps = diffs[diffs > 0.25]
        
        num_records = len(sub)
        num_gaps = len(gaps)
        max_gap = gaps.max() if not gaps.empty else 0
        
        # Výpočet teoretické úplnosti (kolik záznamů by tam mělo být podle času start-cíl)
        duration = sub['datetime'].max() - sub['datetime'].min()
        timespan_h = duration.total_seconds() / 3600
        
        # Počet očekávaných 15min intervalů: (hodiny * 4) + 1 (počáteční bod)
        expected = (timespan_h * 4) + 1
        
        # Výpočet procentuální ztráty dat (pokud expected > 0)
        missing_pct = (1 - (num_records / expected)) * 100 if expected > 0 else 0
        # Ošetření záporných hodnot (stává se při překryvech/duplicitách)
        missing_pct = max(0, missing_pct)
            
        results.append({
            'ID': uid,
            'Záznamů': num_records,
            'Výpadek [%]': round(missing_pct, 2),
            'Max. mezera [h]': round(max_gap, 2),
            'Počet mezer': num_gaps
        })

    return pd.DataFrame(results)

# --- SPUŠTĚNÍ ANALÝZY ---
# Voláme funkci na datasetu df_raw, který už prošel transformací času
continuity_res = analyze_continuity(df_raw)

# Seřazení výsledků podle závažnosti výpadků
continuity_res = continuity_res.sort_values('Výpadek [%]', ascending=False)

# Výpočet statistik pro textovou část BP
prumerny_vypadek = continuity_res['Výpadek [%]'].mean()
pocet_budov = len(continuity_res)

print("\n--- STATISTIKA KONTINUITY (Top 10 ID s největšími výpadky) ---")
print(continuity_res.head(10).to_string(index=False))

print("-" * 40)
print(f"Celkový počet analyzovaných objektů: {pocet_budov}")
print(f"Celková průměrná chybovost v datasetu: {prumerny_vypadek:.2f} %")

# Identifikace ID s výpadkem nad 5 % pro zdůvodnění v textu
outliers = continuity_res[continuity_res['Výpadek [%]'] > 5]
print(f"Počet objektů s chybovostí nad 5 %: {len(outliers)}")
print("-" * 40)



import pandas as pd
import numpy as np

# ---  AUDIT SUROVÉHO DATASETU (S korektní technickou terminologií) ---
print("Zahajuji audit surového datasetu...")

# Předpokládáme, že df_audit obsahuje těch surových 23,051,008 řádků
count_raw = len(df_audit)

# Definice profilů pro audit (Technické označení dle energetických standardů)
cumulative_profiles = ['+E*(m)', '-E*(m)']
main_p = '+A15'  # Činný odběr
ind_p  = '+Ri15' # Jalový odběr induktivní (motory, trafa)
cap_p  = '+Rc15' # Jalový odběr kapacitní (LED, IT zdroje)
fve_p  = '-A15'  # Činná dodávka (FVE)

# Dynamické určení sloupce s hodnotou
target_col = 'value'
if 'value_kWh' in df_audit.columns: target_col = 'value_kWh'
elif 'anonymized_value' in df_audit.columns: target_col = 'anonymized_value'

# --- IMPLEMENTACE NUMPY LOGIKY (Ochrana proti duplicitním indexům) ---
profile_array = df_audit['profile'].values
value_array = df_audit[target_col].values
id_array = df_audit['id_numeric'].values

# Identifikace FVE (Objekty s prokazatelnou dodávkou do sítě)
mask_fve = (profile_array == fve_p) & (value_array > 0)
fve_ids = np.unique(id_array[mask_fve])
fve_ids = sorted([int(i) for i in fve_ids if pd.notna(i)])

# Výpočty počtů profilů (Sumace přes NumPy pole)
count_cumul = np.sum(np.isin(profile_array, cumulative_profiles))
c_a15 = np.sum(profile_array == main_p)
c_ri15 = np.sum(profile_array == ind_p)
c_rc15 = np.sum(profile_array == cap_p)
c_fve = np.sum(profile_array == fve_p)

count_work_total = count_raw - count_cumul 
count_relevant = c_a15 + c_ri15 + c_rc15 + c_fve 
count_other = count_work_total - count_relevant

# 5.  FORMÁTOVANÝ VÝSTUP (TABULKA 3.7 - S TECHNICKÝM POPISEM)
print("\nTabulka 3.7.: Detailní statistika výkonových profilů a klasifikace objektů")
print("="*85)
print(f"{'Parametr analýzy (Interní kód)':<45} | {'Technický popis (Praxe)':<23} | {'Počet řádků':>12}")
print("="*85)
print(f"{'Celkový počet řádků (surová data)':<45} | {'Všechny záznamy':<23} | {count_raw:>12,}")
print(f"{'Vyfiltrované kumulativní stavy (+E*, -E*)':<45} | {'Celkové registry':<23} | {count_cumul:>12,}")
print(f"{'Počet řádků pro výkonovou analýzu':<45} | {'Čtvrthodinová data':<23} | {count_work_total:>12,}")
print("-" * 85)
print(f"{'Záznamy určené k analýze (celkem)':<45} | {'Relevantní profily':<23} | {count_relevant:>12,}")
print(f"{'   profil +A15':<45} | {'Činný odběr':<23} | {c_a15:>12,}")
print(f"{'   profil +Ri15':<45} | {'Jalový ind. odběr':<23} | {c_ri15:>12,}")
print(f"{'   profil +Rc15':<45} | {'Jalový kap. odběr':<23} | {c_rc15:>12,}")
print(f"{'   profil -A15':<45} | {'Činná dodávka (FVE)':<23} | {c_fve:>12,}")
print("-" * 85)
print(f"{'Ostatní nezařazené záznamy (šum)':<45} | {'Nevyužitá data':<23} | {count_other:>12,}")
print("-" * 85)
total_objects = len(np.unique(id_array))
print(f"{'Počet objektů typu „Pouze odběr“':<45} | {'Běžní spotřebitelé':<23} | {total_objects - len(fve_ids):>12}")
print(f"{'Počet objektů typu „Výrobna (FVE)“':<45} | {'Prosumeři / Zdroje':<23} | {len(fve_ids):>12}")
print(f"{'Identifikátory výroben (EAN)':<45} | {str(fve_ids):<38}")
print("="*85 + "\n")

# --- 3. EXPORT OČIŠTĚNÉHO DATASETU ---
# Odstraníme balast (count_other) a ponecháme jen to, co budeme modelovat
print("Vytvářím finální df_work pro neuronovou síť...")
df_work = df_audit[df_audit['profile'].isin([main_p, ind_p, cap_p, fve_p])].copy()
df_work = df_work.reset_index(drop=True)



