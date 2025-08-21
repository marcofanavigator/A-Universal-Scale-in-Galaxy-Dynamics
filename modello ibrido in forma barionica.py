# 🔬 MODELLO IBRIDO IN FORMA BARIONICA
# ρ_dm = A * √ρ_bar + B * ρ_bar^γ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Carica i risultati dello script precedente
df = pd.read_csv("gpf_redmapper_improved_results.csv")
print("🔬 MODELLO IBRIDO IN FORMA BARIONICA")
print("=" * 60)

# Costanti
G = 4.3009e-6  # kpc km² s⁻² M☉⁻¹
h = 0.7
E_z = np.sqrt(0.3 * (1 + df['Z_LAMBDA'])**3 + 0.7)

# Calcola R200 [kpc]
R200_Mpc = 1.48 * (df['LAMBDA'] / 40)**0.2 / E_z / h
R200_kpc = R200_Mpc * 1000
V_200 = (4/3) * np.pi * R200_kpc**3

# Massa barionica (stellare + gas)
M_star_pivot = 2.35e13 * h**(-1)
M_star = M_star_pivot * (df['LAMBDA'] / 30)**1.12 * (1 + df['Z_LAMBDA'])**(-0.3)
f_gas = 0.156 / 0.048
M_bar = M_star * (1 + f_gas)

# Densità barionica media
rho_bar = M_bar / V_200  # M_sun / kpc³

# ==========================================
# 1. MODELLO IBRIDO: ρ_dm = A√ρ_bar + B ρ_bar^γ
# ==========================================
def hybrid_barionic_model(params, rho_bar, R200_kpc, sigma_obs):
    A, B, gamma = params
    if A <= 0 or B < 0 or gamma <= 0:
        return np.inf * np.ones_like(sigma_obs)
    
    # Densità di materia oscura
    rho_dm = A * np.sqrt(rho_bar) + B * (rho_bar ** gamma)
    M_dm = rho_dm * V_200
    
    # Massa totale e dispersione
    M_tot = M_bar + M_dm
    sigma_theory = np.sqrt(G * M_tot / (5 * R200_kpc))
    
    return sigma_theory

def chi2_hybrid(params):
    sigma_theory = hybrid_barionic_model(params, rho_bar, R200_kpc, df['sigma_v_obs'].values)
    residuals = (sigma_theory - df['sigma_v_obs'].values) / df['sigma_v_obs'].values
    return np.sum(residuals**2) / len(residuals)

print("1️⃣ OTTIMIZZAZIONE MODELLO IBRIDO BARIONICO")
print("-" * 50)
print("Forma: ρ_dm = A √ρ_bar + B ρ_bar^γ")

# Valore iniziale: A=300, B piccolo, γ≈1
initial_guess = [300.0, 10.0, 1.0]
bounds = [
    (100, 1000),    # A
    (0.1, 1000),    # B
    (0.5, 2.0)      # gamma
]

result = minimize(chi2_hybrid, initial_guess, bounds=bounds, method='L-BFGS-B')

if result.success:
    A_opt, B_opt, gamma_opt = result.x
    print(f"✅ A (emergente): {A_opt:.1f} M_sun^0.5 / kpc^1.5")
    print(f"✅ B (primordiale): {B_opt:.2f}")
    print(f"✅ γ (esponente): {gamma_opt:.3f}")
    print(f"✅ Chi² ridotto: {result.fun:.4f}")
else:
    print("❌ Ottimizzazione fallita")
    A_opt, B_opt, gamma_opt = 300.0, 10.0, 1.0

# Calcola risultati
sigma_hybrid_bar = hybrid_barionic_model([A_opt, B_opt, gamma_opt], rho_bar, R200_kpc, df['sigma_v_obs'])
ratio_hybrid_bar = sigma_hybrid_bar / df['sigma_v_obs']
print(f"✅ Rapporto medio: {ratio_hybrid_bar.mean():.3f}")
print(f"✅ Scatter: {ratio_hybrid_bar.std():.3f}")

# Salva nel DataFrame
df['sigma_v_hybrid_bar'] = sigma_hybrid_bar
df['ratio_hybrid_bar'] = ratio_hybrid_bar

# ==========================================
# 2. CONFRONTO CON MODELLI PRECEDENTI
# ==========================================
print("\n2️⃣ CONFRONTO CON MODELLI PRECEDENTI")
print("-" * 50)

# GPF puro (da dati esistenti)
chi2_gpf = np.mean(((df['ratio_gpf'] - 1))**2)
chi2_hybrid_old = np.mean(((df['ratio_hybrid'] - 1))**2)
chi2_hybrid_bar = result.fun

print(f"Chi² GPF puro: {chi2_gpf:.4f}")
print(f"Chi² Ibrido classico: {chi2_hybrid_old:.4f}")
print(f"Chi² Ibrido barionico: {chi2_hybrid_bar:.4f}")

delta_vs_gpf = chi2_gpf - chi2_hybrid_bar
delta_vs_old = chi2_hybrid_old - chi2_hybrid_bar

print(f"🎯 Miglioramento vs GPF: {delta_vs_gpf:.4f}")
print(f"🎯 Miglioramento vs ibrido: {delta_vs_old:.4f}")

# ==========================================
# 3. ANALISI CORRELAZIONI
# ==========================================
print("\n3️⃣ ANALISI CORRELAZIONI")
print("-" * 50)

corr_bar, p_bar = pearsonr(df['LAMBDA'], ratio_hybrid_bar)
corr_gpf, p_gpf = pearsonr(df['LAMBDA'], df['ratio_gpf'])
corr_old, p_old = pearsonr(df['LAMBDA'], df['ratio_hybrid'])

print(f"Correlazione ratio vs λ (ibrido barionico): r = {corr_bar:.3f} (p = {p_bar:.3f})")
print(f"Correlazione ratio vs λ (GPF puro): r = {corr_gpf:.3f} (p = {p_gpf:.3f})")
print(f"Correlazione ratio vs λ (ibrido classico): r = {corr_old:.3f} (p = {p_old:.3f})")

# ==========================================
# 4. GRAFICI DI CONFRONTO
# ==========================================
print("\n4️⃣ GENERAZIONE GRAFICI")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Grafico 1: Ibrido barionico vs osservato
ax1 = axes[0, 0]
ax1.scatter(df['sigma_v_obs'], df['sigma_v_hybrid_bar'], c=df['LAMBDA'], cmap='plasma', s=40, alpha=0.7)
ax1.plot([200, 1200], [200, 1200], 'k--', lw=2, alpha=0.8)
ax1.set_xlabel('σ_v osservato (km/s)')
ax1.set_ylabel('σ_v (ibrido barionico)')
ax1.set_title(f'Ibrido barionico (A={A_opt:.0f}, B={B_opt:.1f}, γ={gamma_opt:.2f})')
ax1.grid(True, alpha=0.3)
plt.colorbar(ax1.collections[0], ax=ax1, label='Ricchezza (λ)')

# Grafico 2: Confronto tra modelli
ax2 = axes[0, 1]
ax2.scatter(df['LAMBDA'], df['ratio_gpf'], label='GPF puro', alpha=0.6, s=30)
ax2.scatter(df['LAMBDA'], df['ratio_hybrid'], label='Ibrido classico', alpha=0.6, s=30)
ax2.scatter(df['LAMBDA'], df['ratio_hybrid_bar'], label='Ibrido barionico', alpha=0.8, s=30)
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.8)
ax2.set_xlabel('Ricchezza (λ)')
ax2.set_ylabel('Ratio σ_v')
ax2.set_title('Confronto modelli')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Grafico 3: Distribuzione ratio
ax3 = axes[1, 0]
ax3.hist(df['ratio_gpf'], bins=20, alpha=0.6, label='GPF puro', density=True)
ax3.hist(df['ratio_hybrid'], bins=20, alpha=0.6, label='Ibrido classico', density=True)
ax3.hist(ratio_hybrid_bar, bins=20, alpha=0.6, label='Ibrido barionico', density=True)
ax3.axvline(x=1.0, color='k', linestyle='--', alpha=0.8)
ax3.set_xlabel('Ratio σ_v')
ax3.set_ylabel('Densità')
ax3.set_title('Distribuzione performance')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Grafico 4: Contributo delle componenti DM
ax4 = axes[1, 1]
rho_dm_emergent = A_opt * np.sqrt(rho_bar)
rho_dm_primordial = B_opt * (rho_bar ** gamma_opt)
ratio_component = rho_dm_primordial / (rho_dm_emergent + rho_dm_primordial)
ax4.scatter(df['LAMBDA'], ratio_component, c=df['Z_LAMBDA'], cmap='cool', s=40, alpha=0.7)
ax4.set_xlabel('Ricchezza (λ)')
ax4.set_ylabel('Frazione DM primordiale')
ax4.set_title('Transizione emergente → ΛCDM')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
plt.colorbar(ax4.collections[0], ax=ax4, label='Redshift')

plt.tight_layout()
plt.savefig("gpf_hybrid_barionic_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 5. RIEPILOGO
# ==========================================
print("\n" + "="*60)
print("🎯 RIEPILOGO: MODELLO IBRIDO BARIONICO")
print("="*60)
print(f"Parametri ottimali: A = {A_opt:.1f}, B = {B_opt:.2f}, γ = {gamma_opt:.3f}")
print(f"Chi² ridotto: {chi2_hybrid_bar:.4f}")
print(f"Scatter ratio: {ratio_hybrid_bar.std():.3f}")
print(f"Correlazione con λ: r = {corr_bar:.3f}")

if gamma_opt < 1.1:
    print("💡 Nota: γ ≈ 1 suggerisce che la componente ΛCDM è proporzionale alla massa barionica (f_cosmic).")
elif gamma_opt > 1.3:
    print("⚠️  γ > 1 suggerisce una crescita più rapida della DM primordiale in ammassi massicci.")

print("\n✅ Analisi completata! Risultati salvati.")

# Salva il DataFrame aggiornato
df.to_csv("gpf_redmapper_improved_results_hybrid_barionic.csv", index=False)
