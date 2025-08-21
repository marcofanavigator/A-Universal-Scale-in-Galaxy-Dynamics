# 🔬 ANALISI GPF COMPLETA - VERSIONE FINALE CORRETTA
# Tutti i modelli sono dimensionalmente consistenti
# Correzione di: TypeError: 'function' object is not subscriptable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
import os

print("🚀 ANALISI GPF COMPLETA: MODELLI FISICAMENTE VALIDI")
print("=" * 80)

# --- COSTANTI FISICHE ---
G = 4.3009e-6          # kpc km² s⁻² M☉⁻¹
c = 299792.458         # km/s
h = 0.7                # Parametro di Hubble
A_GPF_ORIGINAL = 300.0 # M_sun^0.5 / kpc^1.5

# --- CARICA I DATI ---
print("📁 Caricamento dati...")
try:
    df_clusters = pd.read_csv("J_ApJS_224_1_cat_dr8.csv")
    df_members = pd.read_csv("J_ApJS_224_1_mmb_dr8.csv")
except FileNotFoundError as e:
    print(f"❌ File non trovato: {e}")
    print("Assicurati che i file siano nella stessa cartella dello script.")
    exit()

# Rinomina colonne
df_clusters = df_clusters.rename(columns={'zlambda': 'Z_LAMBDA', 'lambda': 'LAMBDA'})
df_members = df_members.rename(columns={'zspec': 'Z'})

# Filtra ammassi validi
df_clusters = df_clusters[
    (df_clusters['LAMBDA'] > 30) &
    (df_clusters['Z_LAMBDA'] >= 0.1) &
    (df_clusters['Z_LAMBDA'] <= 0.4)
]
print(f"✅ {len(df_clusters)} ammassi selezionati")

# Calcola dispersione di velocità con 3-sigma clipping
def robust_velocity_dispersion(group):
    z_vals = group['Z'].dropna().values
    if len(z_vals) < 10:
        return pd.Series({'sigma_z': np.nan, 'n_members': len(z_vals)})
    z_mean = np.mean(z_vals)
    for _ in range(3):  # Rimozione iterativa degli outlier
        z_std = np.std(z_vals)
        if z_std == 0:
            break
        z_vals = z_vals[np.abs(z_vals - z_mean) < 3 * z_std]
        if len(z_vals) < 5:
            break
        z_mean = np.mean(z_vals)
    sigma_z = np.std(z_vals) if len(z_vals) >= 5 else np.nan
    return pd.Series({'sigma_z': sigma_z, 'n_members': len(z_vals)})

df_members_filtered = df_members[df_members['PMem'] > 0.5]
grouped = df_members_filtered.groupby('ID').apply(robust_velocity_dispersion).reset_index()
grouped['sigma_v_obs'] = grouped['sigma_z'] * c  # km/s

# Unisci dati
df = pd.merge(df_clusters, grouped, on='ID', how='inner')
df = df[(df['n_members'] >= 10) & (df['sigma_v_obs'] > 0) & (df['sigma_v_obs'] < 2000)]
print(f"✅ {len(df)} ammassi con dispersione valida")

# --- RELAZIONI SCALING ---
print("⚙️ Calcolo masse, raggi e densità...")
E_z = np.sqrt(0.3 * (1 + df['Z_LAMBDA'])**3 + 0.7)
df['R200_Mpc'] = 1.48 * (df['LAMBDA'] / 40)**0.2 / E_z / h
df['R200_kpc'] = df['R200_Mpc'] * 1000
V_200 = (4/3) * np.pi * df['R200_kpc']**3

# Massa barionica
M_star_pivot = 2.35e13 * h**(-1)
df['M_star'] = M_star_pivot * (df['LAMBDA'] / 30)**1.12 * (1 + df['Z_LAMBDA'])**(-0.3)
f_gas = 0.156 / 0.048
df['M_gas'] = df['M_star'] * f_gas
df['Mbar'] = df['M_star'] + df['M_gas']

# Densità barionica media
df['rho_bar'] = df['Mbar'] / V_200  # M_sun / kpc^3

# --- MODELLO GPF PURO ---
df['rho_dm_gpf'] = A_GPF_ORIGINAL * np.sqrt(df['rho_bar'])
df['Mdm_gpf'] = V_200 * df['rho_dm_gpf']
df['M_tot_gpf'] = df['Mbar'] + df['Mdm_gpf']
df['sigma_v_gpf'] = np.sqrt(G * df['M_tot_gpf'] / (5 * df['R200_kpc']))
df['ratio_gpf'] = df['sigma_v_gpf'] / df['sigma_v_obs']

# --- MODELLO IBRIDO (GPF + ΛCDM) ---
def estimate_lambda_cdm_component(lambda_richness, z):
    M_halo_200 = 1.0e14 * (lambda_richness / 40)**1.08 * (1 + z)**(-0.3)
    f_cosmic = 0.048 / 0.309
    M_bar_cosmic = M_halo_200 * f_cosmic
    return M_halo_200 - M_bar_cosmic

df['Mdm_lambda_cdm'] = estimate_lambda_cdm_component(df['LAMBDA'], df['Z_LAMBDA'])
df['Mdm_hybrid'] = df['Mdm_gpf'] + df['Mdm_lambda_cdm']
df['M_tot_hybrid'] = df['Mbar'] + df['Mdm_hybrid']
df['sigma_v_hybrid'] = np.sqrt(G * df['M_tot_hybrid'] / (5 * df['R200_kpc']))
df['ratio_hybrid'] = df['sigma_v_hybrid'] / df['sigma_v_obs']

# --- OTTIMIZZAZIONE PARAMETRI GPF (CORRETTA) ---
print("\n1️⃣ OTTIMIZZAZIONE PARAMETRI GPF")
print("-" * 40)

def gpf_model_function(params, lambda_vals, z_vals, sigma_obs):
    A_gpf, alpha_scale = params
    R200_kpc = 1.48 * (lambda_vals / 40)**0.2 / np.sqrt(0.3 * (1 + z_vals)**3 + 0.7) / h * 1000
    M_star = 2.35e13 * h**(-1) * (lambda_vals / 30)**1.12 * (1 + z_vals)**(-0.3)
    M_bar = M_star * (1 + 0.156 / 0.048)
    V_200 = (4/3) * np.pi * R200_kpc**3
    rho_bar = M_bar / V_200
    A_effective = A_gpf * (lambda_vals / 40)**alpha_scale
    rho_dm_gpf = A_effective * np.sqrt(rho_bar)
    M_dm_gpf = V_200 * rho_dm_gpf
    M_tot = M_bar + M_dm_gpf
    sigma_theory = np.sqrt(G * M_tot / (5 * R200_kpc))
    return sigma_theory

def objective_function(params):
    sigma_theory = gpf_model_function(params, df['LAMBDA'].values, df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
    residuals = (sigma_theory - df['sigma_v_obs'].values) / df['sigma_v_obs'].values
    return np.sum(residuals**2) / len(residuals)

initial_guess = [300.0, 0.0]
bounds = [(100, 1000), (-0.5, 0.5)]
result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

if result.success:
    A_opt, alpha_opt = result.x
    print(f"✅ A_GPF ottimale: {A_opt:.1f} M_sun^0.5 / kpc^1.5")
    print(f"✅ Alpha scale: {alpha_opt:.3f}")
    print(f"✅ Chi² ridotto: {result.fun:.4f}")
    sigma_opt = gpf_model_function(result.x, df['LAMBDA'].values, df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
    df['sigma_v_gpf_opt'] = sigma_opt
    df['ratio_opt'] = sigma_opt / df['sigma_v_obs']
    print(f"✅ Rapporto medio ottimizzato: {df['ratio_opt'].mean():.3f}")
    print(f"✅ Scatter ottimizzato: {df['ratio_opt'].std():.3f}")
else:
    print("❌ Ottimizzazione fallita")
    A_opt, alpha_opt = 300.0, 0.0

# --- ANALISI CORRELAZIONI ---
print("\n2️⃣ ANALISI CORRELAZIONI")
print("-" * 40)
variables = ['LAMBDA', 'Z_LAMBDA', 'Mbar', 'M_star', 'R200_kpc']
correlations = {}
for var in variables:
    if var in df.columns:
        corr, p_val = pearsonr(df[var].dropna(), df['ratio_gpf'].dropna())
        correlations[var] = {'correlation': corr, 'p_value': p_val}
        print(f"{var}: r = {corr:.3f} (p = {p_val:.3f})")
strongest_corr = max(correlations.items(), key=lambda x: abs(x[1]['correlation']))
print(f"🎯 Correlazione più forte: {strongest_corr[0]} (r = {strongest_corr[1]['correlation']:.3f})")

# --- ANALISI DEI RESIDUI ---
print("\n3️⃣ ANALISI DEI RESIDUI")
print("-" * 40)
df['residuals'] = (df['ratio_gpf'] - 1.0) / df['ratio_gpf'].std()
outliers = df[np.abs(df['residuals']) > 2]
print(f"Outlier identificati: {len(outliers)}/{len(df)} ({100*len(outliers)/len(df):.1f}%)")
if len(outliers) > 0:
    print("Caratteristiche degli outlier:")
    print(f"  Ricchezza media: {outliers['LAMBDA'].mean():.1f} (vs {df['LAMBDA'].mean():.1f})")
    print(f"  Redshift medio: {outliers['Z_LAMBDA'].mean():.3f} (vs {df['Z_LAMBDA'].mean():.3f})")
    print(f"  Ratio medio: {outliers['ratio_gpf'].mean():.3f}")

# --- GRAFICI ---
print("\n4️⃣ GENERAZIONE GRAFICI")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. GPF vs Osservato
ax1 = axes[0, 0]
ax1.scatter(df['sigma_v_obs'], df['sigma_v_gpf'], alpha=0.6, s=30)
ax1.plot([200, 1200], [200, 1200], 'k--', alpha=0.8)
ax1.set_xlabel('σ_v osservato (km/s)')
ax1.set_ylabel('σ_v GPF (km/s)')
ax1.set_title('GPF vs Osservato')
ax1.grid(True, alpha=0.3)

# 2. Ibrido vs Osservato
ax2 = axes[0, 1]
ax2.scatter(df['sigma_v_obs'], df['sigma_v_hybrid'], alpha=0.6, s=30)
ax2.plot([200, 1200], [200, 1200], 'k--', alpha=0.8)
ax2.set_xlabel('σ_v osservato (km/s)')
ax2.set_ylabel('σ_v Ibrido (km/s)')
ax2.set_title('Ibrido vs Osservato')
ax2.grid(True, alpha=0.3)

# 3. Ratio vs Ricchezza
ax3 = axes[0, 2]
ax3.scatter(df['LAMBDA'], df['ratio_gpf'], label='GPF', alpha=0.6, s=30)
ax3.scatter(df['LAMBDA'], df['ratio_hybrid'], label='Ibrido', alpha=0.6, s=30)
ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
ax3.set_xlabel('Ricchezza (λ)')
ax3.set_ylabel('Ratio σ_v')
ax3.set_title('Performance vs Ricchezza')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# 4. Distribuzione ratio
ax4 = axes[1, 0]
ax4.hist(df['ratio_gpf'], bins=20, alpha=0.6, label='GPF', density=True)
ax4.hist(df['ratio_hybrid'], bins=20, alpha=0.6, label='Ibrido', density=True)
ax4.axvline(x=1.0, color='k', linestyle='--', alpha=0.8)
ax4.set_xlabel('Ratio')
ax4.set_ylabel('Densità')
ax4.set_title('Distribuzione Performance')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Residui vs λ
ax5 = axes[1, 1]
scatter5 = ax5.scatter(df['LAMBDA'], df['residuals'], c=df['Z_LAMBDA'], cmap='coolwarm', s=40, alpha=0.7)
ax5.axhline(y=0, color='k', linestyle='-', alpha=0.8)
ax5.axhline(y=2, color='r', linestyle='--', alpha=0.6)
ax5.axhline(y=-2, color='r', linestyle='--', alpha=0.6)
ax5.set_xlabel('Ricchezza (λ)')
ax5.set_ylabel('Residui standardizzati')
ax5.set_title('Residui vs Ricchezza')
ax5.set_xscale('log')
plt.colorbar(scatter5, ax=ax5, label='Redshift')

# 6. Performance vs Redshift
z_bins = np.linspace(df['Z_LAMBDA'].min(), df['Z_LAMBDA'].max(), 6)
z_centers = (z_bins[1:] + z_bins[:-1]) / 2
mean_ratios = [df[(df['Z_LAMBDA'] >= z_bins[i]) & (df['Z_LAMBDA'] < z_bins[i+1])]['ratio_gpf'].mean() for i in range(len(z_bins)-1)]
ax6 = axes[1, 2]
ax6.plot(z_centers, mean_ratios, 'o-', linewidth=2, markersize=8)
ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
ax6.set_xlabel('Redshift')
ax6.set_ylabel('Ratio medio GPF')
ax6.set_title('Performance vs Redshift')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gpf_analysis_complete_final.png", dpi=150, bbox_inches='tight')
plt.show()

# --- SALVATAGGIO ---
df.to_csv("gpf_redmapper_improved_results.csv", index=False)
print("\n✅ Analisi completata!")
print("📊 Risultati salvati in 'gpf_redmapper_improved_results.csv'")
