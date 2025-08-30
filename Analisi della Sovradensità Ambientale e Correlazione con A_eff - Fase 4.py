import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. CARICA I DATI DELLA FASE 2
    print("Caricamento dati dalla fase 2...")
    try:
        clusters = pd.read_csv('clusters_con_delta_e_norm.csv')
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        print("Assicurati di aver eseguito prima la seconda fase dell'analisi")
        return
    
    # 2. CONVALIDA DELLA RELAZIONE TEORICA
    print("Convalida della relazione teorica...")
    
    # Fissa A0 al valore medio di A_eff_norm e stima solo beta
    A0_fixed = clusters['A_eff_norm'].mean()
    
    def theoretical_relation_fixed_A0(delta, beta):
        return A0_fixed * (1 + delta) ** beta
    
    # Dividi in bin di delta e calcola i valori medi
    delta_bins = pd.qcut(clusters['delta'], q=20, duplicates='drop')
    delta_means = []
    a_eff_means = []
    a_eff_std = []
    
    for bin in delta_bins.unique():
        mask = delta_bins == bin
        delta_means.append(clusters.loc[mask, 'delta'].mean())
        a_eff_means.append(clusters.loc[mask, 'A_eff_norm'].mean())
        a_eff_std.append(clusters.loc[mask, 'A_eff_norm'].std())
    
    # Rimuovi eventuali valori NaN
    valid_idx = ~np.isnan(delta_means) & ~np.isnan(a_eff_means)
    delta_means = np.array(delta_means)[valid_idx]
    a_eff_means = np.array(a_eff_means)[valid_idx]
    a_eff_std = np.array(a_eff_std)[valid_idx]
    
    try:
        popt_fixed, pcov_fixed = curve_fit(theoretical_relation_fixed_A0, 
                                         delta_means, 
                                         a_eff_means,
                                         sigma=a_eff_std,
                                         absolute_sigma=True,
                                         p0=[-0.5])
        perr_fixed = np.sqrt(np.diag(pcov_fixed))
        print(f"Parametri con A0 fissato a {A0_fixed:.3f}: beta = {popt_fixed[0]:.3f} ± {perr_fixed[0]:.3f}")
        fit_success_fixed = True
    except Exception as e:
        print(f"Fit con A0 fissato fallito: {e}")
        fit_success_fixed = False
    
    # 3. ESPLORAZIONE DI FORME FUNZIONALI ALTERNATIVE
    print("Esplorazione di forme funzionali alternative...")
    
    # Funzione lineare
    def linear_func(delta, a, b):
        return a + b * delta
    
    # Funzione logaritmica
    def log_func(delta, a, b):
        return a + b * np.log1p(delta)
    
    # Funzione esponenziale
    def exp_func(delta, a, b):
        return a * np.exp(b * delta)
    
    functions = {
        'Lineare': linear_func,
        'Logaritmica': log_func,
        'Esponenziale': exp_func,
        'Teorica (A0 fisso)': theoretical_relation_fixed_A0
    }
    
    results = {}
    
    for name, func in functions.items():
        try:
            if name == 'Teorica (A0 fisso)':
                popt, pcov = curve_fit(func, delta_means, a_eff_means, 
                                      sigma=a_eff_std, absolute_sigma=True, p0=[-0.5])
                y_pred = func(delta_means, *popt)
            else:
                popt, pcov = curve_fit(func, delta_means, a_eff_means, 
                                      sigma=a_eff_std, absolute_sigma=True)
                y_pred = func(delta_means, *popt)
            
            r2 = r2_score(a_eff_means, y_pred)
            mse = mean_squared_error(a_eff_means, y_pred)
            results[name] = {'r2': r2, 'mse': mse, 'params': popt}
            print(f"{name}: R² = {r2:.4f}, MSE = {mse:.4f}")
        except Exception as e:
            print(f"Fit {name} fallito: {e}")
    
    # 4. ANALISI APPROFONDITA PER BIN DI RICCHEZZA
    print("Analisi approfondita per bin di ricchezza...")
    
    # Dividi il campione in base alla ricchezza degli ammassi
    richness_bins = pd.qcut(clusters['lambda'], q=4, labels=['Bassa', 'Medio-bassa', 'Medio-alta', 'Alta'])
    richness_analysis = []
    
    for bin_name, bin_data in clusters.groupby(richness_bins):
        if len(bin_data) > 100:
            # Calcola i valori medi per bin di delta
            delta_bins_bin = pd.qcut(bin_data['delta'], q=10, duplicates='drop')
            delta_means_bin = []
            a_eff_means_bin = []
            
            for bin_delta in delta_bins_bin.unique():
                mask = delta_bins_bin == bin_delta
                delta_means_bin.append(bin_data.loc[mask, 'delta'].mean())
                a_eff_means_bin.append(bin_data.loc[mask, 'A_eff_norm'].mean())
            
            # Rimuovi NaN
            valid_idx = ~np.isnan(delta_means_bin) & ~np.isnan(a_eff_means_bin)
            delta_means_bin = np.array(delta_means_bin)[valid_idx]
            a_eff_means_bin = np.array(a_eff_means_bin)[valid_idx]
            
            # Fit della relazione teorica
            try:
                A0_bin = bin_data['A_eff_norm'].mean()
                popt_bin, pcov_bin = curve_fit(theoretical_relation_fixed_A0, 
                                             delta_means_bin, 
                                             a_eff_means_bin,
                                             p0=[-0.5])
                
                richness_analysis.append({
                    'bin': bin_name,
                    'n_clusters': len(bin_data),
                    'lambda_mean': bin_data['lambda'].mean(),
                    'A0': A0_bin,
                    'beta': popt_bin[0],
                    'beta_err': np.sqrt(np.diag(pcov_bin))[0] if len(pcov_bin) > 0 else np.nan
                })
            except Exception as e:
                print(f"Fit fallito per bin {bin_name}: {e}")
                richness_analysis.append({
                    'bin': bin_name,
                    'n_clusters': len(bin_data),
                    'lambda_mean': bin_data['lambda'].mean(),
                    'A0': bin_data['A_eff_norm'].mean(),
                    'beta': np.nan,
                    'beta_err': np.nan
                })
    
    # 5. ANALISI DELLA RELAZIONE TRA BETA E RICCHEZZA
    print("Analisi della relazione tra beta e ricchezza...")
    
    richness_df = pd.DataFrame(richness_analysis)
    
    if len(richness_df) > 1 and not np.all(np.isnan(richness_df['beta'])):
        try:
            # Fit lineare tra beta e lambda_mean
            valid_beta = ~np.isnan(richness_df['beta'])
            if np.sum(valid_beta) > 1:
                popt_beta, pcov_beta = np.polyfit(richness_df.loc[valid_beta, 'lambda_mean'], 
                                                richness_df.loc[valid_beta, 'beta'], 
                                                1, cov=True)
                beta_slope, beta_intercept = popt_beta
                beta_slope_err, beta_intercept_err = np.sqrt(np.diag(pcov_beta))
                
                print(f"Relazione beta-lambda: beta = {beta_intercept:.3f} ± {beta_intercept_err:.3f} + {beta_slope:.3f} ± {beta_slope_err:.3f} * lambda")
        except Exception as e:
            print(f"Analisi relazione beta-lambda fallita: {e}")
    
    # 6. VISUALIZZAZIONI
    print("Creazione visualizzazioni...")
    
    plt.figure(figsize=(18, 12))
    
    # Grafico 1: Confronto forme funzionali
    plt.subplot(3, 3, 1)
    plt.errorbar(delta_means, a_eff_means, yerr=a_eff_std, fmt='o', alpha=0.7, label='Dati')
    
    x_fit = np.linspace(min(delta_means), max(delta_means), 100)
    for name, result in results.items():
        if name == 'Teorica (A0 fisso)':
            y_fit = theoretical_relation_fixed_A0(x_fit, *result['params'])
        else:
            y_fit = functions[name](x_fit, *result['params'])
        plt.plot(x_fit, y_fit, label=f'{name} (R²={result["r2"]:.3f})')
    
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato (medio)')
    plt.title('Confronto forme funzionali')
    plt.legend()
    
    # Grafico 2: Relazione teorica per bin di ricchezza
    plt.subplot(3, 3, 2)
    for _, row in richness_df.iterrows():
        if not np.isnan(row['beta']):
            y_fit = row['A0'] * (1 + x_fit) ** row['beta']
            plt.plot(x_fit, y_fit, label=f'{row["bin"]} (λ={row["lambda_mean"]:.1f}, β={row["beta"]:.3f})')
    
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('Relazione teorica per bin di ricchezza')
    plt.legend()
    
    # Grafico 3: Beta vs ricchezza media
    plt.subplot(3, 3, 3)
    if len(richness_df) > 1 and not np.all(np.isnan(richness_df['beta'])):
        plt.errorbar(richness_df['lambda_mean'], richness_df['beta'], 
                    yerr=richness_df['beta_err'], fmt='o', capsize=5)
        
        if 'popt_beta' in locals():
            x_fit_beta = np.linspace(min(richness_df['lambda_mean']), max(richness_df['lambda_mean']), 100)
            y_fit_beta = np.polyval(popt_beta, x_fit_beta)
            plt.plot(x_fit_beta, y_fit_beta, 'r--', 
                    label=f'β = {beta_intercept:.3f} + {beta_slope:.3f}·λ')
            plt.legend()
        
        plt.xlabel('Ricchezza media (λ)')
        plt.ylabel('β')
        plt.title('Relazione tra β e ricchezza')
    
    # Grafico 4: Distribuzione di A_eff_norm per bin di ricchezza
    plt.subplot(3, 3, 4)
    for bin_name, bin_data in clusters.groupby(richness_bins):
        plt.hist(bin_data['A_eff_norm'], bins=30, alpha=0.5, label=str(bin_name))
    plt.xlabel('A_eff normalizzato')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione di A_eff per bin di ricchezza')
    plt.legend()
    
    # Grafico 5: Distribuzione di delta per bin di ricchezza
    plt.subplot(3, 3, 5)
    for bin_name, bin_data in clusters.groupby(richness_bins):
        plt.hist(bin_data['delta'], bins=30, alpha=0.5, label=str(bin_name))
    plt.xlabel('Sovraddensità (δ)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione di δ per bin di ricchezza')
    plt.legend()
    
    # Grafico 6: A_eff_norm vs delta per bin di ricchezza
    plt.subplot(3, 3, 6)
    for bin_name, bin_data in clusters.groupby(richness_bins):
        plt.scatter(bin_data['delta'], bin_data['A_eff_norm'], alpha=0.1, label=str(bin_name))
    plt.xlabel('Sovraddensità (δ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('A_eff vs δ per bin di ricchezza')
    plt.legend()
    
    # Grafico 7: Valori di A0 per bin di ricchezza
    plt.subplot(3, 3, 7)
    if len(richness_df) > 0:
        plt.bar(range(len(richness_df)), richness_df['A0'])
        plt.xticks(range(len(richness_df)), richness_df['bin'], rotation=45)
        plt.ylabel('A0')
        plt.title('A0 per bin di ricchezza')
    
    # Grafico 8: Valori di beta per bin di ricchezza
    plt.subplot(3, 3, 8)
    if len(richness_df) > 0 and not np.all(np.isnan(richness_df['beta'])):
        plt.errorbar(range(len(richness_df)), richness_df['beta'], 
                    yerr=richness_df['beta_err'], fmt='o', capsize=5)
        plt.xticks(range(len(richness_df)), richness_df['bin'], rotation=45)
        plt.ylabel('β')
        plt.title('β per bin di ricchezza')
    
    # Grafico 9: Relazione tra A0 e beta
    plt.subplot(3, 3, 9)
    if len(richness_df) > 1 and not np.all(np.isnan(richness_df['beta'])):
        plt.scatter(richness_df['A0'], richness_df['beta'])
        for i, row in richness_df.iterrows():
            plt.annotate(row['bin'], (row['A0'], row['beta']))
        plt.xlabel('A0')
        plt.ylabel('β')
        plt.title('Relazione tra A0 e β')
    
    plt.tight_layout()
    plt.savefig('analisi_fase4.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. SALVA I RISULTATI
    richness_df.to_csv('risultati_approfonditi_bin_ricchezza.csv', index=False)
    
    # 8. RIEPILOGO RISULTATI
    print("\nRIEPILOGO RISULTATI FASE 4:")
    if fit_success_fixed:
        print(f"1. Parametri con A0 fissato: beta = {popt_fixed[0]:.3f} ± {perr_fixed[0]:.3f}")
    
    print("2. Confronto forme funzionali:")
    for name, result in results.items():
        print(f"   {name}: R² = {result['r2']:.4f}")
    
    print("3. Risultati per bin di ricchezza:")
    for _, row in richness_df.iterrows():
        if not np.isnan(row['beta']):
            print(f"   {row['bin']}: A0 = {row['A0']:.3f}, β = {row['beta']:.3f} ± {row['beta_err']:.3f}")
        else:
            print(f"   {row['bin']}: A0 = {row['A0']:.3f}, β = NaN")

if __name__ == "__main__":
    main()
