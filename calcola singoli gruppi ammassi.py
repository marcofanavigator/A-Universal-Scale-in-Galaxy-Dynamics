import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CARICA I RISULTATI PRECEDENTI ---
df_results = pd.read_csv("gpf_redmapper_sva1_results.csv")

# --- DIVIDI IN GRUPPI PER RICCHEZZA (lambda) ---
df_results['lambda_group'] = pd.cut(df_results['LAMBDA'], 
                                    bins=[0, 50, 100, 1000], 
                                    labels=['Low (λ < 50)', 'Medium (50 ≤ λ < 100)', 'High (λ ≥ 100)'])

# --- STATISTICHE PER GRUPPO ---
summary = df_results.groupby('lambda_group').agg(
    n_ammassi=('ID', 'count'),
    lambda_media=('LAMBDA', 'mean'),
    sigma_v_oss_media=('sigma_v_obs', 'mean'),
    sigma_v_gpf_media=('sigma_v_gpf', 'mean'),
    ratio_medio=('ratio', 'mean'),
    ratio_mediano=('ratio', 'median')
).round(2)

print("\n" + "="*80)
print("RIEPILOGO PER GRUPPI DI RICCHEZZA")
print("="*80)
print(summary)

# --- GRAFICO A BARRE DEL RAPPORTO MEDIO E MEDIANO ---
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(summary))
width = 0.35

ax.bar(x - width/2, summary['ratio_medio'], width, label='Rapporto Medio', color='skyblue', edgecolor='black')
ax.bar(x + width/2, summary['ratio_mediano'], width, label='Rapporto Mediano', color='lightcoral', edgecolor='black')

ax.set_xlabel('Gruppo di Ricchezza')
ax.set_ylabel('Rapporto σ_v^GPF / σ_v^oss')
ax.set_title('Confronto GPF vs Osservato per Gruppi di Ricchezza')
ax.set_xticks(x)
ax.set_xticklabels(summary.index)
ax.legend()
ax.grid(True, alpha=0.3)

# Aggiungi etichette sopra le barre
for i, (mean_val, median_val) in enumerate(zip(summary['ratio_medio'], summary['ratio_mediano'])):
    ax.text(i - width/2, mean_val + 0.02, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.text(i + width/2, median_val + 0.02, f'{median_val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("gpf_groups_analysis.png", dpi=150)
plt.show()

# --- SALVA IL RIEPILOGO ---
summary.to_csv("gpf_groups_summary.csv")
print(f"\n📊 Riepilogo salvato in 'gpf_groups_summary.csv'")
