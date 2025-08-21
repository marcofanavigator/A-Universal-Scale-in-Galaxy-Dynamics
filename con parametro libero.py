# 🔬 ANALISI AVANZATA: OTTIMIZZAZIONE COMPLETA DI A E α
# Modello: ρ_dm = A * ρ_bar^α
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Carica i risultati dello script precedente
df = pd.read_csv("gpf_redmapper_improved_results.csv")
print("🔬 ANALISI AVANZATA: OTTIMIZZAZIONE DI A E α")
print("=" * 60)

# ==========================================
# 1. OTTIMIZZAZIONE DI A E α INSIEME
# ==========================================
def gpf_model_free_alpha(params, lambda_vals, z_vals, sigma_obs):
    """Modello GPF con A e α liberi: ρ_dm ∝ ρ_bar^α"""
    A_gpf, alpha = params
    # Costanti
    G = 4.3009e-6  # kpc km² s⁻² M☉⁻¹
    h = 0.7
    # Fattore di crescita cosmologica
    E_z = np.sqrt(0.3 * (1 + z_vals)**3 + 0.7)
    
    # Raggio R200 [kpc]
    R200_Mpc = 1.48 * (lambda_vals / 40)**0.2 / E_z / h
    R200_kpc = R200_Mpc * 1000
    
    # Volume
    V_200 = (4/3) * np.pi * R200_kpc**3
    
    # Massa barionica (stellare + gas)
    M_star_pivot = 2.35e13 * h**(-1)
    M_star = M_star_pivot * (lambda_vals / 30)**1.12 * (1 + z_vals)**(-0.3)
    f_gas = 0.156 / 0.048
    M_bar = M_star * (1 + f_gas)
    
    # Densità barionica media
    rho_bar = M_bar / V_200
    
    # Materia oscura GPF: ρ_dm = A * ρ_bar^α
    rho_dm_gpf = A_gpf * (rho_bar ** alpha)
    M_dm_gpf = V_200 * rho_dm_gpf
    
    # Massa totale e dispersione di velocità
    M_tot = M_bar + M_dm_gpf
    sigma_theory = np.sqrt(G * M_tot / (5 * R200_kpc))
    
    return sigma_theory

def objective_free_alpha(params):
    """Funzione chi-quadro da minimizzare"""
    A_gpf, alpha = params
    if A_gpf <= 0 or alpha <= 0:  # Vincoli fisici
        return 1e6
    try:
        sigma_theory = gpf_model_free_alpha(params, df['LAMBDA'].values, 
                                          df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
        residuals = (sigma_theory - df['sigma_v_obs'].values) / df['sigma_v_obs'].values
        chi2 = np.sum(residuals**2) / len(residuals)
        return chi2
    except:
        return 1e6

print("1️⃣ OTTIMIZZAZIONE DI A E α (modello: ρ_dm ∝ ρ_bar^α)")
print("-" * 50)

# Valore iniziale: A=300, α=0.5
initial_guess = [300.0, 0.5]
# Bounds: A tra 100 e 1000, α tra 0.1 e 1.0
bounds = [(100, 1000), (0.1, 1.0)]

result = minimize(objective_free_alpha, initial_guess, bounds=bounds, method='L-BFGS-B')

if result.success:
    A_opt, alpha_opt = result.x
    print(f"✅ A_GPF ottimale: {A_opt:.1f} M_sun^0.5 / kpc^1.5")
    print(f"✅ α ottimale: {alpha_opt:.3f}")
    print(f"✅ Chi² ridotto: {result.fun:.4f}")
    
    # Calcola i nuovi valori teorici
    sigma_opt = gpf_model_free_alpha([A_opt, alpha_opt], df['LAMBDA'].values, 
                                   df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
    ratio_opt = sigma_opt / df['sigma_v_obs']
    print(f"✅ Rapporto medio: {ratio_opt.mean():.3f}")
    print(f"✅ Scatter: {ratio_opt.std():.3f}")
else:
    print("❌ Ottimizzazione fallita")
    A_opt, alpha_opt = 300.0, 0.5

# Aggiungi al DataFrame
df['sigma_v_gpf_free'] = sigma_opt
df['ratio_gpf_free'] = ratio_opt

# ==========================================
# 2. CONFRONTO CON MODELLO α = 0.5
# ==========================================
print("\n2️⃣ CONFRONTO CON MODELLO FISSO α = 0.5")
print("-" * 50)

# Calcola il modello con α = 0.5 e A = 300
sigma_fixed = gpf_model_free_alpha([300.0, 0.5], df['LAMBDA'].values, 
                                 df['Z_LAMBDA'].values, df['sigma_v_obs'].values)
ratio_fixed = sigma_fixed / df['sigma_v_obs']
chi2_fixed = np.mean(((ratio_fixed - 1))**2)

print(f"Modello α = 0.5, A = 300:")
print(f"  Chi² ridotto: {chi2_fixed:.4f}")
print(f"  Rapporto medio: {ratio_fixed.mean():.3f}")
print(f"  Scatter: {ratio_fixed.std():.3f}")

# Miglioramento
delta_chi2 = chi2_fixed - result.fun
print(f"🎯 Miglioramento Δχ²: {delta_chi2:.4f}")

# ==========================================
# 3. ANALISI CORRELAZIONI CON RICCHEZZA
# ==========================================
print("\n3️⃣ ANALISI CORRELAZIONI")
print("-" * 50)

corr_free, p_free = pearsonr(df['LAMBDA'], ratio_opt)
corr_fixed, p_fixed = pearsonr(df['LAMBDA'], ratio_fixed)

print(f"Correlazione ratio vs λ (α libero): r = {corr_free:.3f} (p = {p_free:.3f})")
print(f"Correlazione ratio vs λ (α = 0.5): r = {corr_fixed:.3f} (p = {p_fixed:.3f})")

# ==========================================
# 4. GRAFICI DI CONFRONTO
# ==========================================
print("\n4️⃣ GENERAZIONE GRAFICI")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Grafico 1: GPF libero vs osservato
ax1 = axes[0, 0]
ax1.scatter(df['sigma_v_obs'], df['sigma_v_gpf_free'], c=df['LAMBDA'], cmap='plasma', s=40, alpha=0.7)
ax1.plot([200, 1200], [200, 1200], 'k--', lw=2, alpha=0.8)
ax1.set_xlabel('σ_v osservato (km/s)')
ax1.set_ylabel('σ_v GPF (α libero)')
ax1.set_title(f'GPF (α = {alpha_opt:.2f}) vs Osservato')
ax1.grid(True, alpha=0.3)
plt.colorbar(ax1.collections[0], ax=ax1, label='Ricchezza (λ)')

# Grafico 2: GPF fisso vs osservato
ax2 = axes[0, 1]
ax2.scatter(df['sigma_v_obs'], sigma_fixed, c=df['LAMBDA'], cmap='plasma', s=40, alpha=0.7)
ax2.plot([200, 1200], [200, 1200], 'k--', lw=2, alpha=0.8)
ax2.set_xlabel('σ_v osservato (km/s)')
ax2.set_ylabel('σ_v GPF (α = 0.5)')
ax2.set_title('GPF (α = 0.5) vs Osservato')
ax2.grid(True, alpha=0.3)
plt.colorbar(ax2.collections[0], ax=ax2, label='Ricchezza (λ)')

# Grafico 3: Ratio vs Ricchezza
ax3 = axes[1, 0]
ax3.scatter(df['LAMBDA'], ratio_opt, label=f'α = {alpha_opt:.2f}', alpha=0.7, s=30)
ax3.scatter(df['LAMBDA'], ratio_fixed, label='α = 0.5', alpha=0.7, s=30, color='orange')
ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
ax3.set_xlabel('Ricchezza (λ)')
ax3.set_ylabel('Ratio σ_v')
ax3.set_title('Performance vs Ricchezza')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# Grafico 4: Distribuzione ratio
ax4 = axes[1, 1]
ax4.hist(ratio_opt, bins=20, alpha=0.6, label=f'α libero (σ = {ratio_opt.std():.3f})', density=True)
ax4.hist(ratio_fixed, bins=20, alpha=0.6, label=f'α = 0.5 (σ = {ratio_fixed.std():.3f})', density=True)
ax4.axvline(x=1.0, color='k', linestyle='--', alpha=0.8)
ax4.set_xlabel('Ratio σ_v(teoria)/σ_v(oss)')
ax4.set_ylabel('Densità')
ax4.set_title('Distribuzione dei ratio')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gpf_alpha_free_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 5. RIEPILOGO FINALE
# ==========================================
print("\n" + "="*60)
print("🎯 RIEPILOGO: MODELLO CON α LIBERO")
print("="*60)
print(f"Parametri ottimali: A = {A_opt:.1f}, α = {alpha_opt:.3f}")
print(f"Chi² ridotto: {result.fun:.4f} (vs {chi2_fixed:.4f} per α=0.5)")
print(f"Miglioramento chi²: {delta_chi2:.4f}")
print(f"Scatter ridotto del: {100*(ratio_fixed.std() - ratio_opt.std())/ratio_fixed.std():.1f}%")
print(f"Residui vs λ: correlazione più debole con α libero (r = {corr_free:.3f})")

if abs(corr_free) < abs(corr_fixed):
    print("✅ Il modello con α libero riduce il trend sistemico con la ricchezza.")
else:
    print("⚠️  Il trend con λ è leggermente peggiore, ma lo scatter è minore.")

print("\n✅ Analisi completata! Risultati salvati in grafico e DataFrame.")

# Salva i nuovi risultati
df.to_csv("gpf_redmapper_improved_results_alpha_free.csv", index=False)
