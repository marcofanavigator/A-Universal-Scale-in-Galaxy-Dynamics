import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
    
    # 2. ANALISI DELLA RELAZIONE FUNZIONALE
    print("Analisi della relazione funzionale...")
    
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
    
    # Fit della relazione teorica: A_eff = A_0 * (1 + δ)^β
    def theoretical_relation(delta, A0, beta):
        return A0 * (1 + delta) ** beta
    
    try:
        popt_theo, pcov_theo = curve_fit(theoretical_relation, 
                                       delta_means, 
                                       a_eff_means,
                                       sigma=a_eff_std,
                                       absolute_sigma=True,
                                       p0=[1, -0.5])
        perr_theo = np.sqrt(np.diag(pcov_theo))
        print(f"Parametri teorici: A0 = {popt_theo[0]:.3f} ± {perr_theo[0]:.3f}, beta = {popt_theo[1]:.3f} ± {perr_theo[1]:.3f}")
        fit_success_theo = True
    except Exception as e:
        print(f"Fit teorico fallito: {e}")
        fit_success_theo = False
    
    # 3. ANALISI CON REGRESSIONE POLINOMIALE
    print("Analisi con regressione polinomiale...")
    
    # Prova una regressione polinomiale
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(clusters[['delta']])
    
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X_poly, clusters['A_eff_norm'], test_size=0.3, random_state=42
    )
    
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train_poly)
    y_pred_poly = lr_poly.predict(X_test_poly)
    
    r2_poly = r2_score(y_test_poly, y_pred_poly)
    print(f"R2 con regressione polinomiale (grado 2): {r2_poly:.4f}")
    print(f"Coefficienti: {lr_poly.coef_}")
    
    # 4. ANALISI PER SOTTOCAMPIONI (BIN DI RICCHEZZA)
    print("Analisi per sottocampioni (bin di ricchezza)...")
    
    # Dividi il campione in base alla ricchezza degli ammassi
    richness_bins = pd.qcut(clusters['lambda'], q=4)
    richness_results = []
    
    for bin_name, bin_data in clusters.groupby(richness_bins):
        X_bin = bin_data[['delta']]
        y_bin = bin_data['A_eff_norm']
        
        if len(X_bin) > 100:
            X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
                X_bin, y_bin, test_size=0.3, random_state=42
            )
            rf_bin = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_bin.fit(X_train_bin, y_train_bin)
            
            y_pred_bin = rf_bin.predict(X_test_bin)
            r2_bin = r2_score(y_test_bin, y_pred_bin)
            
            # Calcola l'importanza della feature
            importance_bin = rf_bin.feature_importances_[0] if len(rf_bin.feature_importances_) > 0 else 0
            
            richness_results.append({
                'bin': bin_name,
                'n_clusters': len(bin_data),
                'r2': r2_bin,
                'delta_importance': importance_bin,
                'lambda_mean': bin_data['lambda'].mean(),
                'delta_mean': bin_data['delta'].mean()
            })
    
    # 5. ANALISI DELL'INTERAZIONE TRA VARIABILI
    print("Analisi dell'interazione tra variabili...")
    
    # Esplora l'interazione tra lambda e delta
    clusters['lambda_delta'] = clusters['lambda'] * clusters['delta']
    
    X_int = clusters[['lambda', 'delta', 'lambda_delta']]
    y_int = clusters['A_eff_norm']
    
    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_int, y_int, test_size=0.3, random_state=42
    )
    
    rf_int = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_int.fit(X_train_int, y_train_int)
    
    y_pred_int = rf_int.predict(X_test_int)
    r2_int = r2_score(y_test_int, y_pred_int)
    
    print(f"R2 con interazione: {r2_int:.4f}")
    print("Feature importance con interazione:")
    feature_importance_int = pd.DataFrame({
        'feature': X_int.columns,
        'importance': rf_int.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance_int)
    
    # 6. VISUALIZZAZIONI
    print("Creazione visualizzazioni...")
    
    plt.figure(figsize=(18, 12))
    
    # Grafico 1: Relazione teorica
    plt.subplot(3, 3, 1)
    plt.errorbar(delta_means, a_eff_means, yerr=a_eff_std, fmt='o', alpha=0.7, label='Dati')
    if fit_success_theo:
        x_fit = np.linspace(min(delta_means), max(delta_means), 100)
        y_fit = theoretical_relation(x_fit, *popt_theo)
        plt.plot(x_fit, y_fit, 'r-', label=f'A_eff = {popt_theo[0]:.3f} · (1+δ)$^{{{popt_theo[1]:.3f}}}$')
        plt.legend()
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato (medio)')
    plt.title('Relazione teorica tra δ e A_eff')
    
    # Grafico 2: Feature importance con interazione
    plt.subplot(3, 3, 2)
    plt.barh(feature_importance_int['feature'], feature_importance_int['importance'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance con interazione')
    
    # Grafico 3: R2 per bin di ricchezza
    plt.subplot(3, 3, 3)
    bin_labels = [str(r['bin']) for r in richness_results]
    r2_values = [r['r2'] for r in richness_results]
    plt.bar(range(len(richness_results)), r2_values)
    plt.xticks(range(len(richness_results)), bin_labels, rotation=45, ha='right')
    plt.ylabel('R²')
    plt.title('R² per bin di ricchezza')
    
    # Grafico 4: Importanza delta per bin di ricchezza
    plt.subplot(3, 3, 4)
    delta_importance_values = [r['delta_importance'] for r in richness_results]
    plt.bar(range(len(richness_results)), delta_importance_values)
    plt.xticks(range(len(richness_results)), bin_labels, rotation=45, ha='right')
    plt.ylabel('Importanza di δ')
    plt.title('Importanza di δ per bin di ricchezza')
    
    # Grafico 5: Distribuzione di lambda
    plt.subplot(3, 3, 5)
    plt.hist(clusters['lambda'], bins=50, alpha=0.7)
    plt.xlabel('Ricchezza (λ)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione della ricchezza degli ammassi')
    
    # Grafico 6: Distribuzione di delta
    plt.subplot(3, 3, 6)
    plt.hist(clusters['delta'], bins=50, alpha=0.7)
    plt.xlabel('Sovraddensità (δ)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione della sovraddensità')
    
    # Grafico 7: A_eff_norm vs lambda
    plt.subplot(3, 3, 7)
    plt.scatter(clusters['lambda'], clusters['A_eff_norm'], alpha=0.1, s=5)
    plt.xlabel('Ricchezza (λ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('A_eff normalizzato vs Ricchezza')
    
    # Grafico 8: A_eff_norm vs delta (tutti i dati)
    plt.subplot(3, 3, 8)
    plt.scatter(clusters['delta'], clusters['A_eff_norm'], alpha=0.1, s=5)
    plt.xlabel('Sovraddensità (δ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('A_eff normalizzato vs Sovraddensità')
    
    # Grafico 9: Lambda vs delta
    plt.subplot(3, 3, 9)
    plt.scatter(clusters['delta'], clusters['lambda'], alpha=0.1, s=5)
    plt.xlabel('Sovraddensità (δ)')
    plt.ylabel('Ricchezza (λ)')
    plt.title('Ricchezza vs Sovraddensità')
    
    plt.tight_layout()
    plt.savefig('analisi_fase3.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. SALVA I RISULTATI
    # Crea un dataframe con i risultati dei bin di ricchezza
    richness_df = pd.DataFrame(richness_results)
    richness_df.to_csv('risultati_bin_ricchezza.csv', index=False)
    
    # 8. RIEPILOGO RISULTATI
    print("\nRIEPILOGO RISULTATI FASE 3:")
    print(f"1. R2 con regressione polinomiale: {r2_poly:.4f}")
    print(f"2. R2 con interazione: {r2_int:.4f}")
    
    if fit_success_theo:
        print(f"3. Parametri teorici: A0 = {popt_theo[0]:.3f} ± {perr_theo[0]:.3f}, beta = {popt_theo[1]:.3f} ± {perr_theo[1]:.3f}")
    
    print("4. Risultati per bin di ricchezza:")
    for result in richness_results:
        print(f"   Bin {result['bin']}: R² = {result['r2']:.4f}, Importanza δ = {result['delta_importance']:.4f}")

if __name__ == "__main__":
    main()
