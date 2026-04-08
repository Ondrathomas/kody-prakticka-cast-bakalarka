
import pandas as pd

# Funkce pro načtení zůstává stejná
def load_and_prepare_data(filepath):
    dtypes = {
        'ID záznamu': 'int64',
        'v_ts': 'int64',
        'id_numeric': 'int16',
        'ean': 'str',
        'profile': 'category',
        's': 'int16',
        'value': 'float32'
    }
    return pd.read_csv(filepath, dtype=dtypes, low_memory=False)

# --- NAČTENÍ ---
path = r"C:\Users\ondra\OneDrive\Plocha\cez-data\Prvotni_Dataset_Surovy.csv"
df_raw = load_and_prepare_data(path)

# --- ÚPRAVA PRO IDENTICKÝ VÝSTUP ---
# 1. Přejmenování přesně podle tvého LaTeXu
df_display = df_raw.rename(columns={
    'ID záznamu': 'ID záznamu',
    'v_ts': 'v_ts [ms]',
    'id_numeric': 'id_n',
    'ean': 'EAN budovy',
    'profile': 'prof.',
    's': 's',
    'value': 'value [kWh]'
})

# 2. Nastavení formátu pro velká čísla (přidá mezery jako v LaTeXu)
pd.options.display.float_format = '{:,.2f}'.format
# Pro celá čísla (timestampy) musíme použít trik s formatters, aby měly mezery
formatters = {
    'v_ts [ms]': lambda x: f"{x:,}".replace(",", " "),
    'ID záznamu': lambda x: f"{x:,}".replace(",", " ")
}

# 3. Výběr sloupců
cols = ['ID záznamu', 'v_ts [ms]', 'id_n', 'EAN budovy', 'prof.', 's', 'value [kWh]']

# --- VÝSTUP ---
print("\nVÝSLEDNÁ DATA (Head + Tail):")
print("-" * 110)

head = df_display[cols].head(5)
tail = df_display[cols].tail(5)

# Tisk bez indexu (index=False), aby tam nebylo to levé číslování 0,1,2...
print(head.to_string(index=False, formatters=formatters))
print("      " + ".".join(["  " for _ in range(10)])) # Hezčí tečky
print(tail.to_string(index=False, header=False, formatters=formatters))

print("-" * 110)
print(f"Poznámka: Celkový rozměr matice činí {df_raw.shape[0]:,} řádků × {df_raw.shape[1]} sloupců.")





