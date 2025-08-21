import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- COSTANTI FISICHE ---
G = 4.3009e-6  # kpc km² s⁻² M☉⁻¹
c = 299792.458  # km/s
A_GPF = 300.0  # M_sun^0.5 / kpc^1.5
h = 0.7  # Parametro di Hubble normalizzato

# --- CARICA I DATI ---
print("🚀 Caricamento dati...")
df_clusters = pd.read_csv("J_ApJS_224_1_cat_dr8.csv")
df_members = pd.read_csv("J_ApJS_224_1_mmb_dr8.csv")

print("Colonne in clusters:", df_clusters.columns.tolist())
print("Colonne in members:", df_members.columns.tolist())

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

# --- CALCOLO DISPERSIONE VELOCITÀ MIGLIORATO ---
print("📊 Calcolo dispersione di velocità...")
df_members_filtered = df_members[df_members['PMem'] > 0.5]  # Soglia iniziale più bassa

# Calcola dispersione con correzione per outlier (3-sigma clipping)
def robust_velocity_dispersion(group):
    if len(group) < 10:
        return pd.Series({'sigma_z': np.nan, 'n_members': len(group), 'z_mean': np.nan})
    
    z_vals = group['Z'].values
    # Rimuovi eventuali NaN
    z_vals = z_vals[~np.isnan(z_vals)]
    
    if len(z_vals) < 10:
        return pd.Series({'sigma_z': np.nan, 'n_members': len(z_vals), 'z_mean': np.nan})
    
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    
    # 3-sigma clipping solo se abbiamo varianza
    if z_std > 0:
        mask = np.abs(z_vals - z_mean) < 3 * z_std
        z_clean = z_vals[mask]
    else:
        z_clean = z_vals
    
    if len(z_clean) < 5:
        return pd.Series({'sigma_z': np.nan, 'n_members': len(z_clean), 'z_mean': z_mean})
    
    sigma_z = np.std(z_clean)
    if sigma_z <= 0:  # Evita valori non positivi
        return pd.Series({'sigma_z': np.nan, 'n_members': len(z_clean), 'z_mean': z_mean})
    
    return pd.Series({
        'sigma_z': sigma_z,
        'n_members': len(z_clean),
        'z_mean': np.mean(z_clean)
    })

grouped = df_members_filtered.groupby('ID').apply(robust_velocity_dispersion).reset_index()
grouped['sigma_v_obs'] = grouped['sigma_z'] * c  # km/s

# --- UNISCI CON DATI AMMASSO ---
df_analysis = pd.merge(df_clusters, grouped, on='ID', how='inner')

# Filtri più stringenti per evitare problemi numerici
df_analysis = df_analysis[
    (df_analysis['n_members'] >= 10) &
    (~df_analysis['sigma_v_obs'].isna()) &
    (df_analysis['sigma_v_obs'] > 0) &  # Evita valori non positivi
    (df_analysis['sigma_v_obs'] < 2000) &  # Rimuovi valori irrealistici
    (df_analysis['LAMBDA'] > 30) &
    (df_analysis['LAMBDA'] < 1000)  # Evita valori estremi
]

print(f"✅ {len(df_analysis)} ammassi con dati validi")

# --- RELAZIONI SCALING MIGLIORATI ---
print("⚙️ Applicazione relazioni scaling migliorate...")

# R200 con dipendenza dal redshift (Simet et al. 2017)
E_z = np.sqrt(0.3 * (1 + df_analysis['Z_LAMBDA'])**3 + 0.7)
df_analysis['R200_Mpc'] = 1.48 * (df_analysis['LAMBDA'] / 40)**0.2 / E_z / h
df_analysis['R200_kpc'] = df_analysis['R200_Mpc'] * 1000

# Massa barionica migliorata (stellare + gas)
# Massa stellare da lambda con evoluzione in redshift
M_star_pivot = 2.35e13 * h**(-1)  # M_sun
lambda_pivot = 30.0
alpha_lambda = 1.12
beta_z = -0.3

df_analysis['M_star'] = M_star_pivot * (df_analysis['LAMBDA'] / lambda_pivot)**alpha_lambda * \
                       (1 + df_analysis['Z_LAMBDA'])**beta_z

# Massa del gas (frazione cosmica)
f_gas = 0.156 / 0.048  # frazione gas/stellare tipica nei cluster
df_analysis['M_gas'] = df_analysis['M_star'] * f_gas
df_analysis['Mbar'] = df_analysis['M_star'] + df_analysis['M_gas']

# Densità barionica media
V_200 = (4/3) * np.pi * df_analysis['R200_kpc']**3
df_analysis['rho_bar'] = df_analysis['Mbar'] / V_200

# --- APPLICA GPF ---
df_analysis['rho_dm_gpf'] = A_GPF * np.sqrt(df_analysis['rho_bar'])
df_analysis['Mdm_gpf'] = V_200 * df_analysis['rho_dm_gpf']

# --- CALCOLO DISPERSIONE VELOCITÀ TEORICA MIGLIORATO ---
# Per i cluster, uso il teorema del viriale: sigma² ≈ GM_tot/(5*R200)
# Questo è più appropriato dei singoli oggetti orbitanti
df_analysis['M_tot_gpf'] = df_analysis['Mbar'] + df_analysis['Mdm_gpf']
df_analysis['sigma_v_gpf'] = np.sqrt(G * df_analysis['M_tot_gpf'] / (5 * df_analysis['R200_kpc']))

# --- MODELLO IBRIDO (GPF + ΛCDM) ---
print("🔬 Test del modello ibrido...")

# Stima componente ΛCDM mancante
def estimate_lambda_cdm_component(lambda_richness, z):
    """Stima la componente ΛCDM basata su simulazioni"""
    # Massa dell'alone tipica per la richezza (relazione empirica)
    M_halo_200 = 1.0e14 * (lambda_richness / 40)**1.08 * (1 + z)**(-0.3)  # M_sun
    
    # Frazione barionica cosmologica
    f_cosmic = 0.048 / 0.309  # Ω_b / Ω_m
    M_bar_cosmic = M_halo_200 * f_cosmic
    
    # DM primordiale = DM totale - barionico cosmologico
    M_dm_lambda_cdm = M_halo_200 - M_bar_cosmic
    return M_dm_lambda_cdm

df_analysis['Mdm_lambda_cdm'] = estimate_lambda_cdm_component(
    df_analysis['LAMBDA'], df_analysis['Z_LAMBDA']
)

# Modello ibrido: GPF + ΛCDM
df_analysis['Mdm_hybrid'] = df_analysis['Mdm_gpf'] + df_analysis['Mdm_lambda_cdm']
df_analysis['M_tot_hybrid'] = df_analysis['Mbar'] + df_analysis['Mdm_hybrid']
df_analysis['sigma_v_hybrid'] = np.sqrt(G * df_analysis['M_tot_hybrid'] / (5 * df_analysis['R200_kpc']))

# --- CALCOLO RATIOS ---
df_analysis['ratio_gpf'] = df_analysis['sigma_v_gpf'] / df_analysis['sigma_v_obs']
df_analysis['ratio_hybrid'] = df_analysis['sigma_v_hybrid'] / df_analysis['sigma_v_obs']

# --- ANALISI PER RICCHEZZA ---
def analyze_by_richness(df, richness_bins):
    results = []
    for i in range(len(richness_bins)-1):
        mask = (df['LAMBDA'] >= richness_bins[i]) & (df['LAMBDA'] < richness_bins[i+1])
        subset = df[mask]
        
        if len(subset) > 5:  # almeno 5 cluster per bin
            results.append({
                'richness_bin': f"{richness_bins[i]:.0f}-{richness_bins[i+1]:.0f}",
                'n_clusters': len(subset),
                'lambda_mean': subset['LAMBDA'].mean(),
                'ratio_gpf_mean': subset['ratio_gpf'].mean(),
                'ratio_gpf_std': subset['ratio_gpf'].std(),
                'ratio_hybrid_mean': subset['ratio_hybrid'].mean(),
                'ratio_hybrid_std': subset['ratio_hybrid'].std(),
            })
    
    return pd.DataFrame(results)

richness_bins = [30, 50, 80, 120, 200, 500]
richness_analysis = analyze_by_richness(df_analysis, richness_bins)

print("\n" + "="*80)
print("ANALISI PER BINS DI RICCHEZZA")
print("="*80)
print(richness_analysis.to_string(index=False, float_format='%.3f'))

# --- SALVATAGGIO RISULTATI ---
columns_to_save = [
    'ID', 'Z_LAMBDA', 'LAMBDA', 'Mbar', 'M_star', 'M_gas',
    'Mdm_gpf', 'Mdm_lambda_cdm', 'Mdm_hybrid', 'R200_kpc',
    'sigma_v_obs', 'sigma_v_gpf', 'sigma_v_hybrid',
    'ratio_gpf', 'ratio_hybrid', 'n_members'
]
df_results = df_analysis[columns_to_save].copy()
df_results.to_csv("gpf_redmapper_improved_results.csv", index=False)

# --- GRAFICI MIGLIORATI ---
# Verifica che abbiamo dati validi prima di fare i grafici
if len(df_results) == 0:
    print("❌ Nessun dato valido per i grafici!")
else:
    # Verifica che non ci siano NaN nelle colonne chiave
    mask_valid = (
        ~df_results['sigma_v_obs'].isna() &
        ~df_results['sigma_v_gpf'].isna() &
        ~df_results['sigma_v_hybrid'].isna() &
        (df_results['sigma_v_obs'] > 0) &
        (df_results['sigma_v_gpf'] > 0) &
        (df_results['sigma_v_hybrid'] > 0) &
        (df_results['LAMBDA'] > 0)
    )
    
    df_plot = df_results[mask_valid].copy()
    print(f"📊 Dati validi per grafici: {len(df_plot)}/{len(df_results)}")
    
    if len(df_plot) < 10:
        print("⚠️ Troppo pochi dati validi per grafici significativi")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Grafico 1: GPF vs Osservato
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df_plot['sigma_v_obs'], df_plot['sigma_v_gpf'],
                              c=df_plot['LAMBDA'], cmap='viridis', s=50, alpha=0.7)
        min_val = min(df_plot['sigma_v_obs'].min(), df_plot['sigma_v_gpf'].min())
        max_val = max(df_plot['sigma_v_obs'].max(), df_plot['sigma_v_gpf'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, lw=2)
        ax1.set_xlabel('σ_v osservato (km/s)')
        ax1.set_ylabel('σ_v GPF (km/s)')
        ax1.set_title('GPF vs Osservato')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Ricchezza (λ)')

        # Grafico 2: Ibrido vs Osservato
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df_plot['sigma_v_obs'], df_plot['sigma_v_hybrid'],
                              c=df_plot['LAMBDA'], cmap='viridis', s=50, alpha=0.7)
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, lw=2)
        ax2.set_xlabel('σ_v osservato (km/s)')
        ax2.set_ylabel('σ_v Ibrido (km/s)')
        ax2.set_title('Modello Ibrido vs Osservato')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Ricchezza (λ)')

        # Grafico 3: Ratios vs Ricchezza (scala lineare per evitare errori)
        ax3 = axes[1, 0]
        ax3.scatter(df_plot['LAMBDA'], df_plot['ratio_gpf'], 
                   alpha=0.6, label='GPF', s=30)
        ax3.scatter(df_plot['LAMBDA'], df_plot['ratio_hybrid'], 
                   alpha=0.6, label='Ibrido', s=30)
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Ricchezza (λ)')
        ax3.set_ylabel('Ratio σ_v(teoria)/σ_v(oss)')
        ax3.set_title('Performance vs Ricchezza')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # Rimuovi scala logaritmica per evitare errori
        # ax3.set_xscale('log')

        # Grafico 4: Distribuzione ratios (con controllo NaN)
        ax4 = axes[1, 1]
        ratio_gpf_clean = df_plot['ratio_gpf'].dropna()
        ratio_hybrid_clean = df_plot['ratio_hybrid'].dropna()
        
        if len(ratio_gpf_clean) > 0 and len(ratio_hybrid_clean) > 0:
            ax4.hist(ratio_gpf_clean, bins=15, alpha=0.6, label='GPF', density=True)
            ax4.hist(ratio_hybrid_clean, bins=15, alpha=0.6, label='Ibrido', density=True)
            ax4.axvline(x=1.0, color='k', linestyle='--', alpha=0.8)
            ax4.set_xlabel('Ratio σ_v(teoria)/σ_v(oss)')
            ax4.set_ylabel('Densità di probabilità')
            ax4.set_title('Distribuzione Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("gpf_cluster_analysis_improved.png", dpi=150, bbox_inches='tight')
        plt.show()

# --- RIEPILOGO FINALE ---
print("\n" + "="*80)
print("RIEPILOGO FINALE MIGLIORATO")
print("="*80)
print(f"Ammassi analizzati: {len(df_results)}")
print(f"Range ricchezza: {df_results['LAMBDA'].min():.0f} - {df_results['LAMBDA'].max():.0f}")
print(f"Range redshift: {df_results['Z_LAMBDA'].min():.3f} - {df_results['Z_LAMBDA'].max():.3f}")
print()
print("MODELLO GPF PURO:")
print(f"  Rapporto medio: {df_results['ratio_gpf'].mean():.3f} ± {df_results['ratio_gpf'].std():.3f}")
print(f"  Mediana: {df_results['ratio_gpf'].median():.3f}")
print()
print("MODELLO IBRIDO (GPF + ΛCDM):")
print(f"  Rapporto medio: {df_results['ratio_hybrid'].mean():.3f} ± {df_results['ratio_hybrid'].std():.3f}")
print(f"  Mediana: {df_results['ratio_hybrid'].median():.3f}")

# Calcola miglioramento
improvement = np.abs(df_results['ratio_hybrid'] - 1.0) < np.abs(df_results['ratio_gpf'] - 1.0)
print(f"  Miglioramento in {improvement.sum()}/{len(df_results)} cluster ({100*improvement.mean():.1f}%)")

print("\n✅ Analisi migliorata completata!")
print("📊 Risultati salvati in 'gpf_redmapper_improved_results.csv'")
