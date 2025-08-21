import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import linregress

# Carica i risultati
df = pd.read_csv("gpf_redmapper_improved_results.csv")

print("🎯 ANALISI DI RAFFINAMENTO GPF")
print("="*60)
print(f"Cluster analizzati: {len(df)}")
print(f"Performance GPF: {df['ratio_gpf'].mean():.3f} ± {df['ratio_gpf'].std():.3f}")
print(f"Performance Ibrida: {df['ratio_hybrid'].mean():.3f} ± {df['ratio_hybrid'].std():.3f}")

# ============================================
# 1. OTTIMIZZAZIONE DEL PARAMETRO A_GPF
# ============================================

def calculate_gpf_ratio(A_gpf, df_input):
    """Calcola il ratio GPF/osservato per un dato A_GPF"""
    # Costanti
    G = 4.3009e-6
    h = 0.7
    
    # Ricalcola tutto con nuovo A_GPF
    E_z = np.sqrt(0.3 * (1 + df_input['Z_LAMBDA'])**3 + 0.7)
    R200_Mpc = 1.48 * (df_input['LAMBDA'] / 40)**0.2 / E_z / h
    R200_kpc = R200_Mpc * 1000
    
    # Massa barionica (stessa del precedente)
    M_star_pivot = 2.35e13 * h**(-1)
    M_star = M_star_pivot * (df_input['LAMBDA'] / 30)**1.12 * (1 + df_input['Z_LAMBDA'])**(-0.3)
    f_gas = 0.156 / 0.048
    M_bar = M_star * (1 + f_gas)
    
    # GPF con nuovo A
    V_200 = (4/3) * np.pi * R200_kpc**3
    rho_bar = M_bar / V_200
    rho_dm_gpf = A_gpf * np.sqrt(rho_bar)
    M_dm_gpf = V_200 * rho_dm_gpf
    
    # Dispersione teorica
    M_tot = M_bar + M_dm_gpf
    sigma_theory = np.sqrt(G * M_tot / (5 * R200_kpc))
    
    return sigma_theory / df_input['sigma_v_obs']

def objective_A(A_gpf):
    """Minimizza la deviazione da 1.0"""
    ratios = calculate_gpf_ratio(A_gpf, df)
    # Penalizza sia la media diversa da 1 che la dispersione
    chi2 = np.sum((ratios - 1.0)**2) / len(ratios)
    return chi2

print("\n1️⃣ OTTIMIZZAZIONE PARAMETRO A")
print("-" * 40)

# Ottimizzazione A_GPF
A_range = np.linspace(200, 500, 301)
chi2_values = [objective_A(A) for A in A_range]
A_optimal = A_range[np.argmin(chi2_values)]

print(f"A_GPF ottimale: {A_optimal:.1f} M_sun^0.5 / kpc^1.5")
print(f"A_GPF originale: 300.0")
print(f"Miglioramento χ²: {objective_A(300.0):.4f} → {objective_A(A_optimal):.4f}")

# Calcola performance con A ottimizzato
ratios_opt = calculate_gpf_ratio(A_optimal, df)
print(f"Rapporto medio ottimizzato: {ratios_opt.mean():.3f} ± {ratios_opt.std():.3f}")

# ============================================
# 2. ANALISI SISTEMATICA PER RICCHEZZA
# ============================================

print("\n2️⃣ ANALISI SISTEMATICA RICCHEZZA")
print("-" * 40)

# Definisci bins più fini
richness_edges = [30, 40, 50, 65, 80, 100, 130, 200]
richness_centers = []
gpf_means = []
gpf_stds = []
hybrid_means = []
hybrid_stds = []
n_clusters = []

for i in range(len(richness_edges)-1):
    mask = (df['LAMBDA'] >= richness_edges[i]) & (df['LAMBDA'] < richness_edges[i+1])
    subset = df[mask]
    
    if len(subset) >= 5:  # Almeno 5 cluster per bin
        richness_centers.append((richness_edges[i] + richness_edges[i+1]) / 2)
        gpf_means.append(subset['ratio_gpf'].mean())
        gpf_stds.append(subset['ratio_gpf'].std())
        hybrid_means.append(subset['ratio_hybrid'].mean())
        hybrid_stds.append(subset['ratio_hybrid'].std())
        n_clusters.append(len(subset))
        
        print(f"λ = {richness_edges[i]:.0f}-{richness_edges[i+1]:.0f}: "
              f"N={len(subset)}, GPF={subset['ratio_gpf'].mean():.3f}, "
              f"Ibrido={subset['ratio_hybrid'].mean():.3f}")

# Fit lineare del trend con ricchezza
slope_gpf, intercept_gpf, r_gpf, p_gpf, _ = linregress(richness_centers, gpf_means)
slope_hybrid, intercept_hybrid, r_hybrid, p_hybrid, _ = linregress(richness_centers, hybrid_means)

print(f"\nTrend GPF: slope = {slope_gpf:.6f}, r = {r_gpf:.3f}, p = {p_gpf:.3f}")
print(f"Trend Ibrido: slope = {slope_hybrid:.6f}, r = {r_hybrid:.3f}, p = {p_hybrid:.3f}")

# ============================================
# 3. MODELLO GPF CON EVOLUZIONE SCALA
# ============================================

print("\n3️⃣ MODELLO CON EVOLUZIONE SCALA")
print("-" * 40)

def gpf_scale_dependent(lambda_rich, A_base=300.0, alpha=0.0):
    """GPF con parametro A dipendente dalla ricchezza"""
    return A_base * (lambda_rich / 40.0)**alpha

def fit_scale_model(params):
    """Fit del modello con evoluzione scala"""
    A_base, alpha = params
    
    ratios = []
    for _, row in df.iterrows():
        A_eff = gpf_scale_dependent(row['LAMBDA'], A_base, alpha)
        ratio = calculate_gpf_ratio(A_eff, pd.DataFrame([row]))[0]
        ratios.append(ratio)
    
    ratios = np.array(ratios)
    chi2 = np.sum((ratios - 1.0)**2) / len(ratios)
    return chi2

# Ottimizzazione modello scala-dipendente
from scipy.optimize import minimize
result = minimize(fit_scale_model, [300.0, 0.0], bounds=[(200, 500), (-0.3, 0.3)])

if result.success:
    A_scale_opt, alpha_opt = result.x
    print(f"A_base ottimale: {A_scale_opt:.1f}")
    print(f"Alpha ottimale: {alpha_opt:.4f}")
    print(f"Chi² migliorato: {result.fun:.4f}")
    
    # Test se il miglioramento è significativo
    chi2_constant = objective_A(A_optimal)
    chi2_scale = result.fun
    improvement = (chi2_constant - chi2_scale) / chi2_constant * 100
    print(f"Miglioramento: {improvement:.2f}%")
    
    if abs(alpha_opt) > 0.01 and improvement > 5:
        print("✅ Dipendenza dalla scala SIGNIFICATIVA")
    else:
        print("❌ Dipendenza dalla scala non significativa")

# ============================================
# 4. ANALISI DEL MODELLO IBRIDO
# ============================================

print("\n4️⃣ ANALISI MODELLO IBRIDO")
print("-" * 40)

# Frazione di materia oscura da GPF vs ΛCDM
df['f_gpf'] = df['Mdm_gpf'] / (df['Mdm_gpf'] + df['Mdm_lambda_cdm'])
df['f_lambda_cdm'] = df['Mdm_lambda_cdm'] / (df['Mdm_gpf'] + df['Mdm_lambda_cdm'])

print(f"Frazione GPF media: {df['f_gpf'].mean():.3f} ± {df['f_gpf'].std():.3f}")
print(f"Frazione ΛCDM media: {df['f_lambda_cdm'].mean():.3f} ± {df['f_lambda_cdm'].std():.3f}")

# Correlazione frazione GPF con ricchezza
corr_gpf_lambda = np.corrcoef(df['LAMBDA'], df['f_gpf'])[0,1]
print(f"Correlazione f_GPF - λ: {corr_gpf_lambda:.3f}")

# ============================================
# 5. GRAFICI DIAGNOSTICI
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Ottimizzazione A_GPF
ax1 = axes[0, 0]
ax1.plot(A_range, chi2_values, 'b-', linewidth=2)
ax1.axvline(x=A_optimal, color='r', linestyle='--', alpha=0.8, label=f'A_opt = {A_optimal:.0f}')
ax1.axvline(x=300, color='k', linestyle=':', alpha=0.8, label='A_orig = 300')
ax1.set_xlabel('Parametro A (M_sun^0.5 / kpc^1.5)')
ax1.set_ylabel('χ² ridotto')
ax1.set_title('Ottimizzazione parametro A')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Performance vs ricchezza
ax2 = axes[0, 1]
ax2.errorbar(richness_centers, gpf_means, yerr=gpf_stds, 
            marker='o', label='GPF', capsize=5, linewidth=2, markersize=6)
ax2.errorbar(richness_centers, hybrid_means, yerr=hybrid_stds, 
            marker='s', label='Ibrido', capsize=5, linewidth=2, markersize=6)

# Fit lineari
x_fit = np.linspace(min(richness_centers), max(richness_centers), 100)
y_gpf_fit = slope_gpf * x_fit + intercept_gpf
y_hybrid_fit = slope_hybrid * x_fit + intercept_hybrid
ax2.plot(x_fit, y_gpf_fit, '--', alpha=0.7, color='C0')
ax2.plot(x_fit, y_hybrid_fit, '--', alpha=0.7, color='C1')

ax2.axhline(y=1.0, color='k', linestyle='-', alpha=0.5)
ax2.set_xlabel('Ricchezza (λ)')
ax2.set_ylabel('Rapporto σ_theory/σ_obs')
ax2.set_title(f'Trend con ricchezza\n(GPF: r={r_gpf:.3f}, Ibrido: r={r_hybrid:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Frazione GPF vs ΛCDM
ax3 = axes[0, 2]
ax3.scatter(df['LAMBDA'], df['f_gpf'], alpha=0.6, s=40, c=df['Z_LAMBDA'], cmap='plasma')
ax3.set_xlabel('Ricchezza (λ)')
ax3.set_ylabel('Frazione GPF (M_DM_GPF / M_DM_tot)')
ax3.set_title(f'Bilanciamento GPF vs ΛCDM\n(r = {corr_gpf_lambda:.3f})')
cb3 = plt.colorbar(ax3.collections[0], ax=ax3)
cb3.set_label('Redshift')
ax3.grid(True, alpha=0.3)

# 4. Distribuzione ratios dettagliata
ax4 = axes[1, 0]
bins = np.linspace(0.3, 1.8, 25)
ax4.hist(df['ratio_gpf'], bins=bins, alpha=0.6, label=f'GPF (μ={df["ratio_gpf"].mean():.3f})', 
         density=True, edgecolor='black', linewidth=0.5)
ax4.hist(df['ratio_hybrid'], bins=bins, alpha=0.6, label=f'Ibrido (μ={df["ratio_hybrid"].mean():.3f})', 
         density=True, edgecolor='black', linewidth=0.5)
ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Perfetto accordo')
ax4.set_xlabel('Rapporto σ_theory/σ_obs')
ax4.set_ylabel('Densità di probabilità')
ax4.set_title('Distribuzione performance')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Residui vs parametri osservativi
ax5 = axes[1, 1]
residui = df['ratio_gpf'] - 1.0
scatter5 = ax5.scatter(df['Z_LAMBDA'], residui, c=df['LAMBDA'], 
                      cmap='viridis', s=50, alpha=0.7)
ax5.axhline(y=0, color='k', linestyle='-', alpha=0.8)
ax5.set_xlabel('Redshift')
ax5.set_ylabel('Residui GPF (ratio - 1)')
ax5.set_title('Residui vs Redshift')
cb5 = plt.colorbar(scatter5, ax=ax5)
cb5.set_label('Ricchezza (λ)')
ax5.grid(True, alpha=0.3)

# 6. Miglioramento ibrido per cluster
ax6 = axes[1, 2]
improvement = df['ratio_hybrid'] - df['ratio_gpf']
colors = ['red' if imp < 0 else 'green' for imp in improvement]
ax6.scatter(df['LAMBDA'], improvement, c=colors, alpha=0.7, s=40)
ax6.axhline(y=0, color='k', linestyle='-', alpha=0.8)
ax6.set_xlabel('Ricchezza (λ)')
ax6.set_ylabel('Miglioramento Ibrido - GPF')
ax6.set_title(f'Efficacia modello ibrido\n({np.sum(improvement > 0)}/{len(df)} migliorati)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gpf_refinement_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 6. CONCLUSIONI E RACCOMANDAZIONI
# ============================================

print("\n" + "="*80)
print("🎯 CONCLUSIONI E RACCOMANDAZIONI")
print("="*80)

print(f"1. PARAMETRO A OTTIMALE: {A_optimal:.0f} (vs 300 originale)")

if abs(slope_gpf) > 1e-4 and abs(r_gpf) > 0.3:
    print(f"2. TREND SIGNIFICATIVO CON RICCHEZZA: {slope_gpf:.2e} per unità λ")
else:
    print("2. NESSUN TREND SIGNIFICATIVO CON RICCHEZZA")

if result.success and abs(alpha_opt) > 0.01:
    print(f"3. DIPENDENZA SCALA RILEVATA: α = {alpha_opt:.3f}")
else:
    print("3. NESSUNA DIPENDENZA SCALA SIGNIFICATIVA")

improvement_frac = np.sum(df['ratio_hybrid'] > df['ratio_gpf']) / len(df)
print(f"4. MODELLO IBRIDO: Migliora {100*improvement_frac:.1f}% dei cluster")

print(f"5. BILANCIAMENTO: {100*df['f_gpf'].mean():.1f}% GPF, {100*df['f_lambda_cdm'].mean():.1f}% ΛCDM")

print("\n📋 PROSSIMI PASSI:")
print("   • Testare su campioni indipendenti")
print("   • Esplorare alternative a ρ_DM ∝ √ρ_bar")
print("   • Includere sistemi a massa intermedia")
print("   • Analisi della dipendenza cosmologica")

print("\n✅ Analisi di raffinamento completata!")
