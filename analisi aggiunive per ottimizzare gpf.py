# ANALISI AGGIUNTIVE PER OTTIMIZZARE GPF
# Esegui questo DOPO lo script principale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import pearsonr

# Carica i risultati dello script precedente
df = pd.read_csv("gpf_redmapper_improved_results.csv")

print("🔬 ANALISI AVANZATE PER OTTIMIZZAZIONE GPF")
print("="*60)

# ==========================================
# 1. OTTIMIZZAZIONE DEL PARAMETRO A_GPF
# ==========================================

def gpf_model_function(params, lambda_vals, z_vals, sigma_obs):
    """Modello GPF con parametri variabili"""
    A_gpf, alpha_scale = params
    
    # Calcola masse e densità con A variabile
    E_z = np.sqrt(0.3 * (1 + z_vals)**3 + 0.7)
    R200_Mpc = 1.48 * (lambda_vals / 40)**0.2 / E_z / 0.7
    R200_kpc = R200_Mpc * 1000
    
    # Massa barionica
    M_star_pivot = 2.35e13 * 0.7**(-1)
    M_star = M_star_pivot * (lambda_vals / 30)**1.12 * (1 + z_vals)**(-0.3)
    f_gas = 0.156 / 0.048
    M_bar = M_star * (1 + f_gas)
    
    # Densità barionica
    V_200 = (4/3) * np.pi * R200_kpc**3
    rho_bar = M_bar / V_200
    
    # GPF con possibile dipendenza dalla scala
    A_effective = A_gpf * (lambda_vals / 40)**alpha_scale
    rho_dm_gpf = A_effective * np.sqrt(rho_bar)
    M_dm_gpf = V_200 * rho_dm_gpf
    
    # Dispersione teorica
    G = 4.3009e-6
    M_tot = M_bar + M_dm_gpf
    sigma_theory = np.sqrt(G * M_tot / (5 * R200_kpc))
    
    return sigma_theory

def objective_function(params):
    """Funzione da minimizzare"""
    sigma_theory = gpf_model_function(params, df['LAMBDA'].values, 
                                    df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
    
    # Chi-quadro ridotto
    residuals = (sigma_theory - df['sigma_v_obs'].values) / df['sigma_v_obs'].values
    chi2 = np.sum(residuals**2) / len(residuals)
    return chi2

print("\n1️⃣ OTTIMIZZAZIONE PARAMETRI GPF")
print("-" * 40)

# Ottimizzazione
initial_guess = [300.0, 0.0]  # A_GPF, alpha_scale
bounds = [(100, 1000), (-0.5, 0.5)]

result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

if result.success:
    A_opt, alpha_opt = result.x
    print(f"✅ A_GPF ottimale: {A_opt:.1f} M_sun^0.5 / kpc^1.5")
    print(f"✅ Alpha scale: {alpha_opt:.3f}")
    print(f"✅ Chi² ridotto: {result.fun:.3f}")
    
    # Calcola performance con parametri ottimizzati
    sigma_opt = gpf_model_function(result.x, df['LAMBDA'].values, 
                                 df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
    df['sigma_v_gpf_opt'] = sigma_opt
    df['ratio_opt'] = df['sigma_v_gpf_opt'] / df['sigma_v_obs']
    
    print(f"✅ Rapporto medio ottimizzato: {df['ratio_opt'].mean():.3f}")
    print(f"✅ Scatter ottimizzato: {df['ratio_opt'].std():.3f}")
else:
    print("❌ Ottimizzazione fallita")
    A_opt, alpha_opt = 300.0, 0.0

# ==========================================
# 2. ANALISI DELLE CORRELAZIONI
# ==========================================

print("\n2️⃣ ANALISI CORRELAZIONI")
print("-" * 40)

# Correlazioni con parametri fisici
correlations = {}
variables = ['LAMBDA', 'Z_LAMBDA', 'Mbar', 'M_star', 'R200_kpc']

for var in variables:
    if var in df.columns:
        corr, p_val = pearsonr(df[var], df['ratio_gpf'])
        correlations[var] = {'correlation': corr, 'p_value': p_val}
        print(f"{var}: r = {corr:.3f} (p = {p_val:.3f})")

# Identifica la correlazione più forte
strongest_corr = max(correlations.items(), key=lambda x: abs(x[1]['correlation']))
print(f"\n🎯 Correlazione più forte: {strongest_corr[0]} (r = {strongest_corr[1]['correlation']:.3f})")

# ==========================================
# 3. MODELLO GPF CON DIPENDENZA DALLA SCALA
# ==========================================

print("\n3️⃣ MODELLO GPF DIPENDENTE DALLA SCALA")
print("-" * 40)

def gpf_scale_dependent(lambda_richness, z, A_base=300.0, lambda_0=40.0, alpha=-0.1):
    """GPF con parametro A dipendente dalla ricchezza"""
    return A_base * (lambda_richness / lambda_0)**alpha

# Test diversi valori di alpha
alpha_values = np.linspace(-0.3, 0.1, 9)
scale_results = []

for alpha in alpha_values:
    # Ricalcola GPF con A dipendente dalla scala
    A_scale = gpf_scale_dependent(df['LAMBDA'], df['Z_LAMBDA'], A_opt, 40.0, alpha)
    
    # Ricalcola densità DM
    V_200 = (4/3) * np.pi * df['R200_kpc']**3
    rho_bar = df['Mbar'] / V_200
    rho_dm_scale = A_scale * np.sqrt(rho_bar)
    M_dm_scale = V_200 * rho_dm_scale
    
    # Dispersione teorica
    G = 4.3009e-6
    M_tot_scale = df['Mbar'] + M_dm_scale
    sigma_scale = np.sqrt(G * M_tot_scale / (5 * df['R200_kpc']))
    
    ratio_scale = sigma_scale / df['sigma_v_obs']
    chi2_scale = np.sum(((ratio_scale - 1)**2)) / len(ratio_scale)
    
    scale_results.append({
        'alpha': alpha,
        'chi2': chi2_scale,
        'mean_ratio': ratio_scale.mean(),
        'std_ratio': ratio_scale.std()
    })

df_scale = pd.DataFrame(scale_results)
best_alpha = df_scale.loc[df_scale['chi2'].idxmin(), 'alpha']
print(f"🎯 Migliore alpha per scala: {best_alpha:.3f}")
print(f"✅ Chi² migliorato: {df_scale['chi2'].min():.3f}")

# ==========================================
# 4. ANALISI DEI RESIDUI
# ==========================================

print("\n4️⃣ ANALISI DEI RESIDUI")
print("-" * 40)

# Calcola residui standardizzati
df['residuals'] = (df['ratio_gpf'] - 1.0) / df['ratio_gpf'].std()

# Identifica outlier (|residuo| > 2σ)
outliers = df[np.abs(df['residuals']) > 2]
print(f"Outlier identificati: {len(outliers)}/{len(df)} ({100*len(outliers)/len(df):.1f}%)")

if len(outliers) > 0:
    print("\nCaratteristiche degli outlier:")
    print(f"  Ricchezza media: {outliers['LAMBDA'].mean():.1f} (vs {df['LAMBDA'].mean():.1f})")
    print(f"  Redshift medio: {outliers['Z_LAMBDA'].mean():.3f} (vs {df['Z_LAMBDA'].mean():.3f})")
    print(f"  Ratio medio: {outliers['ratio_gpf'].mean():.3f}")

# ==========================================
# 5. GRAFICI DIAGNOSTICI
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Parametro A ottimizzato vs originale
ax1 = axes[0, 0]
ax1.scatter(df['sigma_v_obs'], df['sigma_v_gpf'], alpha=0.6, label='A = 300', s=30)
if 'sigma_v_gpf_opt' in df.columns:
    ax1.scatter(df['sigma_v_obs'], df['sigma_v_gpf_opt'], alpha=0.6, label=f'A = {A_opt:.0f}', s=30)
ax1.plot([200, 1200], [200, 1200], 'k--', alpha=0.8)
ax1.set_xlabel('σ_v osservato (km/s)')
ax1.set_ylabel('σ_v GPF (km/s)')
ax1.set_title('Confronto parametri')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Alpha scale optimization
ax2 = axes[0, 1]
ax2.plot(df_scale['alpha'], df_scale['chi2'], 'o-', markersize=6)
ax2.axvline(x=best_alpha, color='r', linestyle='--', alpha=0.8)
ax2.set_xlabel('Alpha (dipendenza scala)')
ax2.set_ylabel('Chi² ridotto')
ax2.set_title('Ottimizzazione dipendenza scala')
ax2.grid(True, alpha=0.3)

# 3. Residui vs ricchezza
ax3 = axes[0, 2]
scatter3 = ax3.scatter(df['LAMBDA'], df['residuals'], c=df['Z_LAMBDA'], 
                      cmap='coolwarm', s=40, alpha=0.7)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.8)
ax3.axhline(y=2, color='r', linestyle='--', alpha=0.6)
ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.6)
ax3.set_xlabel('Ricchezza (λ)')
ax3.set_ylabel('Residui standardizzati')
ax3.set_title('Residui vs Ricchezza')
ax3.set_xscale('log')
plt.colorbar(scatter3, ax=ax3, label='Redshift')

# 4. Distribuzione residui
ax4 = axes[1, 0]
ax4.hist(df['residuals'], bins=20, alpha=0.7, density=True, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8)
x_norm = np.linspace(-3, 3, 100)
y_norm = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_norm**2)
ax4.plot(x_norm, y_norm, 'k--', alpha=0.8, label='Normale')
ax4.set_xlabel('Residui standardizzati')
ax4.set_ylabel('Densità')
ax4.set_title('Distribuzione residui')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Correlazione più forte
strongest_var = strongest_corr[0]
if strongest_var in df.columns:
    ax5 = axes[1, 1]
    ax5.scatter(df[strongest_var], df['ratio_gpf'], alpha=0.6, s=40)
    
    # Fit lineare per visualizzazione
    z = np.polyfit(df[strongest_var], df['ratio_gpf'], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(df[strongest_var].min(), df[strongest_var].max(), 100)
    ax5.plot(x_fit, p(x_fit), "r--", alpha=0.8, linewidth=2)
    
    ax5.set_xlabel(strongest_var)
    ax5.set_ylabel('Ratio GPF')
    ax5.set_title(f'Correlazione: r = {strongest_corr[1]["correlation"]:.3f}')
    ax5.grid(True, alpha=0.3)

# 6. Performance per bins di redshift
ax6 = axes[1, 2]
z_bins = np.linspace(df['Z_LAMBDA'].min(), df['Z_LAMBDA'].max(), 6)
z_centers = (z_bins[1:] + z_bins[:-1]) / 2
mean_ratios = []
std_ratios = []

for i in range(len(z_bins)-1):
    mask = (df['Z_LAMBDA'] >= z_bins[i]) & (df['Z_LAMBDA'] < z_bins[i+1])
    if mask.sum() > 5:
        mean_ratios.append(df[mask]['ratio_gpf'].mean())
        std_ratios.append(df[mask]['ratio_gpf'].std())
    else:
        mean_ratios.append(np.nan)
        std_ratios.append(np.nan)

ax6.errorbar(z_centers, mean_ratios, yerr=std_ratios, 
            marker='o', capsize=5, linewidth=2, markersize=8)
ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
ax6.set_xlabel('Redshift')
ax6.set_ylabel('Ratio medio GPF')
ax6.set_title('Performance vs Redshift')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gpf_advanced_diagnostics.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 6. RACCOMANDAZIONI FINALI
# ==========================================

print("\n" + "="*60)
print("🎯 RACCOMANDAZIONI PER MIGLIORARE GPF")
print("="*60)

print(f"1. PARAMETRO A OTTIMALE: {A_opt:.1f} (vs 300.0 originale)")
if abs(alpha_opt) > 0.05:
    print(f"2. DIPENDENZA SCALA: α = {alpha_opt:.3f} (significativa)")
else:
    print("2. DIPENDENZA SCALA: Non significativa")

print(f"3. CORRELAZIONE PRINCIPALE: {strongest_var} (r = {strongest_corr[1]['correlation']:.3f})")

if len(outliers) > 0:
    print(f"4. OUTLIER: {len(outliers)} cluster richiedono attenzione speciale")
    print(f"   - Principalmente a λ = {outliers['LAMBDA'].mean():.0f}")

# Suggerimenti basati sui risultati
improvement_opt = np.abs(df['ratio_opt'] - 1.0).mean() if 'ratio_opt' in df.columns else 999
improvement_orig = np.abs(df['ratio_gpf'] - 1.0).mean()

if improvement_opt < improvement_orig:
    print(f"\n✅ MIGLIORAMENTO: {100*(improvement_orig-improvement_opt)/improvement_orig:.1f}% con parametri ottimizzati")
else:
    print("\n⚠️  I parametri originali sono già near-ottimali")

print("\n📋 PROSSIMI PASSI SUGGERITI:")
print("   1. Test su campioni indipendenti")
print("   2. Includere errori osservativi nel fitting")
print("   3. Esplorare forme funzionali alternative a √ρ_bar")
print("   4. Analisi di sistemi a massa intermedia (gruppi)")

print("\n✅ Analisi avanzata completata!")
