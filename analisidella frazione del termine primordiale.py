# analisidella frazione del termine primordiale.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ PASSO 1: CARICA IL DATAFRAME SALVATO NELL'ANALISI IBRIDA
df = pd.read_csv("gpf_redmapper_improved_results.csv")  # <-- Assicurati che il nome sia corretto

# ✅ PASSO 2: Calcola la densità barionica, se non è già presente
if 'rho_bar' not in df.columns:
    V_200 = (4/3) * np.pi * df['R200_kpc']**3
    df['rho_bar'] = df['Mbar'] / V_200

# ✅ PASSO 3: Definisci i parametri del modello ibrido barionico
A_opt = 300.0   # Come trovato nell'analisi
B_opt = 9.93    # Come trovato
gamma_opt = 0.546  # Come trovato

# ✅ PASSO 4: Ricostruisci i termini del modello
df['rho_dm_emergent'] = A_opt * np.sqrt(df['rho_bar'])
df['rho_dm_primordial'] = B_opt * (df['rho_bar'] ** gamma_opt)

# Evita divisioni per zero
total_dm = df['rho_dm_emergent'] + df['rho_dm_primordial']
df['frazione_primordiale'] = df['rho_dm_primordial'] / (total_dm + 1e-30)

# ✅ PASSO 5: Grafico della frazione primordiale vs ricchezza
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['LAMBDA'], df['frazione_primordiale'], 
                     c=df['Z_LAMBDA'], cmap='plasma', s=50, alpha=0.7)
plt.colorbar(scatter, label='Redshift')
plt.xlabel('Ricchezza (λ)')
plt.ylabel('Frazione DM "primordiale"')
plt.title('Transizione graduale: componente emergente → componente barionica aggiuntiva')
plt.xscale('log')
plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.6, label='50%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("transizione_gpf_lambda.png", dpi=150, bbox_inches='tight')
plt.show()

# ✅ Opzionale: mostra statistiche
print(f"Frazione primordiale media: {df['frazione_primordiale'].mean():.3f}")
print(f"Frazione primordiale in ammassi ricchi (λ > 80): {df[df['LAMBDA'] > 80]['frazione_primordiale'].mean():.3f}")
print(f"Frazione primordiale in ammassi poveri (λ < 50): {df[df['LAMBDA'] < 50]['frazione_primordiale'].mean():.3f}")
