import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- COSTANTI FISICHE ---
G = 4.3009e-6  # kpc km² s⁻² M☉⁻¹
c = 299792.458  # km/s
A_GPF = 300.0  # M_sun^0.5 / kpc^1.5

# --- CARICA I DATI ---
print("🚀 Caricamento dati...")
df_clusters = pd.read_csv("J_ApJS_224_1_cat_dr8.csv")
df_members = pd.read_csv("J_ApJS_224_1_mmb_dr8.csv")

# --- MOSTRA I NOMI DELLE COLONNE ---
print("Colonnes in catsva1:", df_clusters.columns.tolist())
print("Colonnes in catsvamm1:", df_members.columns.tolist())

# --- RINOMINA COLONNE CHIAVE ---
df_clusters = df_clusters.rename(columns={
    'zlambda': 'Z_LAMBDA',
    'lambda': 'LAMBDA'
})

df_members = df_members.rename(columns={
    'zspec': 'Z'
})

# --- FILTRA AMMASSI DI ALTA QUALITÀ ---
print("🔍 Applicazione filtri sugli ammassi...")
df_clusters = df_clusters[
    (df_clusters['LAMBDA'] > 30) &
    (df_clusters['Z_LAMBDA'] >= 0.1) &
    (df_clusters['Z_LAMBDA'] <= 0.4)
]

print(f"✅ {len(df_clusters)} ammassi selezionati")

# --- UNISCI CON MEMBRI ---
print("🔗 Unione con membri...")
df_members_filtered = df_members[df_members['PMem'] > 0.5]  # solo membri probabili

# Calcola dispersione di velocità osservata
grouped = df_members_filtered.groupby('ID')['Z'].agg(['std', 'count']).reset_index()
grouped = grouped.rename(columns={'std': 'sigma_z', 'count': 'n_members'})
grouped['sigma_v_obs'] = grouped['sigma_z'] * c  # km/s

# Unisci con dati ammasso
df_analysis = pd.merge(df_clusters, grouped, on='ID', how='inner')

# Filtra per almeno 10 membri
df_analysis = df_analysis[df_analysis['n_members'] >= 10]
print(f"✅ {len(df_analysis)} ammassi con almeno 10 membri")

# --- STIMA R200 DA LAMBDA ---
# R200 (Mpc) ≈ 0.008 * lambda^0.6
df_analysis['R200_Mpc'] = 0.008 * (df_analysis['LAMBDA'] ** 0.6)
df_analysis['R200_kpc'] = df_analysis['R200_Mpc'] * 1000  # Mpc → kpc

# --- STIMA MASSA BARIONICA ---
df_analysis['Mbar'] = 1e12 * df_analysis['LAMBDA']  # M_sun

# Densità barionica media
df_analysis['rho_bar'] = df_analysis['Mbar'] / (4/3 * np.pi * df_analysis['R200_kpc']**3)

# --- APPLICA GPF ---
df_analysis['rho_dm'] = A_GPF * np.sqrt(df_analysis['rho_bar'])
df_analysis['Mdm'] = (4/3 * np.pi * df_analysis['R200_kpc']**3) * df_analysis['rho_dm']

# Velocità circolare e dispersione prevista
df_analysis['v_circ'] = np.sqrt(G * (df_analysis['Mbar'] + df_analysis['Mdm']) / df_analysis['R200_kpc'])
df_analysis['sigma_v_gpf'] = df_analysis['v_circ'] / np.sqrt(3)

# --- CONFRONTO FINALE ---
df_analysis['ratio'] = df_analysis['sigma_v_gpf'] / df_analysis['sigma_v_obs']

# --- SALVA RISULTATI ---
columns_to_save = [
    'ID', 'Z_LAMBDA', 'LAMBDA', 'Mbar', 'Mdm', 'R200_kpc',
    'sigma_v_obs', 'sigma_v_gpf', 'ratio', 'n_members'
]
df_results = df_analysis[columns_to_save].copy()
df_results.to_csv("gpf_redmapper_sva1_results.csv", index=False)

# --- RIEPILOGO STATISTICO ---
print("\n" + "="*60)
print("RIEPILOGO FINALE")
print("="*60)
print(f"Ammassi analizzati: {len(df_results)}")
print(f"Media σ_v osservato: {df_results['sigma_v_obs'].mean():.1f} km/s")
print(f"Media σ_v GPF: {df_results['sigma_v_gpf'].mean():.1f} km/s")
print(f"Rapporto medio GPF/oss: {df_results['ratio'].mean():.2f}")
print(f"Mediana rapporto: {df_results['ratio'].median():.2f}")

# --- GRAFICO CHIAVE ---
plt.figure(figsize=(10, 8))
plt.scatter(df_results['sigma_v_obs'], df_results['sigma_v_gpf'],
            c=df_results['LAMBDA'], cmap='viridis', s=70, alpha=0.8)
plt.plot([100, 1500], [100, 1500], 'k--', label='Diagonale (perfetto accordo)', lw=2)
plt.xlabel('σ_v osservato (km/s)')
plt.ylabel('σ_v GPF (km/s)')
plt.title('Confronto dispersione di velocità negli ammassi DES SVA1\n(GPF vs Osservato)')
plt.colorbar(label='Ricchezza (λ)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("gpf_vs_clusters_sva1.png", dpi=150)
plt.show()

print("\n✅ Analisi completata!")
print("📊 Risultati salvati in 'gpf_redmapper_sva1_results.csv' e grafico generato.")
