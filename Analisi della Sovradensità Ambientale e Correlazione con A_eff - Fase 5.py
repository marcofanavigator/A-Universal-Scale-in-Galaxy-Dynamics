import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    
    # 2. ESPLORAZIONE DELLA RELAZIONE LOGARITMICA
    print("Esplorazione della relazione logaritmica...")
    
    # Funzione logaritmica
    def log_func(delta, a, b):
        return a + b * np.log1p(delta)
    
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
    
    # Fit della relazione logaritmica
    try:
        popt_log, pcov_log = curve_fit(log_func, delta_means, a_eff_means,
                                      sigma=a_eff_std, absolute_sigma=True)
        perr_log = np.sqrt(np.diag(pcov_log))
        print(f"Parametri logaritmici: a = {popt_log[0]:.3f} ± {perr_log[0]:.3f}, b = {popt_log[1]:.3f} ± {perr_log[1]:.3f}")
        
        # Calcola R²
        y_pred_log = log_func(delta_means, *popt_log)
        r2_log = r2_score(a_eff_means, y_pred_log)
        print(f"R² per relazione logaritmica: {r2_log:.4f}")
    except Exception as e:
        print(f"Fit logaritmico fallito: {e}")
    
    # 3. INVESTIGAZIONE DELLA RELAZIONE TEORICA PROPOSTA
    print("Investigazione della relazione teorica proposta...")
    
    # Relazione teorica: A_eff = A0 * (1 + δ)^β
    def theoretical_relation(delta, A0, beta):
        return A0 * (1 + delta) ** beta
    
    # Analisi per sottocampioni basati sulla ricchezza
    richness_bins = pd.qcut(clusters['lambda'], q=4, labels=['Bassa', 'Medio-bassa', 'Medio-alta', 'Alta'])
    theoretical_results = []
    
    for bin_name, bin_data in clusters.groupby(richness_bins):
        if len(bin_data) > 100:
            # Calcola i valori medi per bin di delta
            delta_bins_bin = pd.qcut(bin_data['delta'], q=10, duplicates='drop')
            delta_means_bin = []
            a_eff_means_bin = []
            a_eff_std_bin = []
            
            for bin_delta in delta_bins_bin.unique():
                mask = delta_bins_bin == bin_delta
                delta_means_bin.append(bin_data.loc[mask, 'delta'].mean())
                a_eff_means_bin.append(bin_data.loc[mask, 'A_eff_norm'].mean())
                a_eff_std_bin.append(bin_data.loc[mask, 'A_eff_norm'].std())
            
            # Rimuovi NaN
            valid_idx = ~np.isnan(delta_means_bin) & ~np.isnan(a_eff_means_bin)
            delta_means_bin = np.array(delta_means_bin)[valid_idx]
            a_eff_means_bin = np.array(a_eff_means_bin)[valid_idx]
            a_eff_std_bin = np.array(a_eff_std_bin)[valid_idx]
            
            # Fit della relazione teorica
            try:
                popt_theo, pcov_theo = curve_fit(theoretical_relation,
                                               delta_means_bin,
                                               a_eff_means_bin,
                                               sigma=a_eff_std_bin,
                                               absolute_sigma=True,
                                               p0=[1, -0.5])
                perr_theo = np.sqrt(np.diag(pcov_theo))
                
                # Calcola R²
                y_pred_theo = theoretical_relation(delta_means_bin, *popt_theo)
                r2_theo = r2_score(a_eff_means_bin, y_pred_theo)
                
                theoretical_results.append({
                    'bin': bin_name,
                    'n_clusters': len(bin_data),
                    'lambda_mean': bin_data['lambda'].mean(),
                    'A0': popt_theo[0],
                    'A0_err': perr_theo[0],
                    'beta': popt_theo[1],
                    'beta_err': perr_theo[1],
                    'r2': r2_theo
                })
            except Exception as e:
                print(f"Fit teorico fallito per bin {bin_name}: {e}")
    
    # 4. MODELLO CON INTERAZIONE TRA RICCHEZZA E SOVRADDENSITÀ
    print("Modello con interazione tra ricchezza e sovraddensità...")
    
    # Prepara i dati
    X = clusters[['lambda', 'delta']].copy()
    X['lambda_delta'] = clusters['lambda'] * clusters['delta']
    y = clusters['A_eff_norm']
    
    # Crea una pipeline con standardizzazione e Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Cross-validazione
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    print(f"R² medio dalla cross-validazione: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Addestra il modello su tutto il dataset
    pipeline.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': pipeline.named_steps['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance con interazione:")
    print(feature_importance)
    
    # 5. VALIDAZIONE DELLA RELAZIONE LOGARITMICA
    print("Validazione della relazione logaritmica...")
    
    # Aggiungi la trasformazione logaritmica di delta
    X_log = clusters[['lambda']].copy()
    X_log['log_delta'] = np.log1p(clusters['delta'])
    X_log['lambda_log_delta'] = clusters['lambda'] * np.log1p(clusters['delta'])
    y_log = clusters['A_eff_norm']
    
    # Cross-validazione per il modello logaritmico
    pipeline_log = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    cv_scores_log = cross_val_score(pipeline_log, X_log, y_log, cv=kf, scoring='r2')
    
    print(f"R² medio per modello logaritmico: {cv_scores_log.mean():.4f} (±{cv_scores_log.std():.4f})")
    
    # Addestra il modello logaritmico su tutto il dataset
    pipeline_log.fit(X_log, y_log)
    
    # Feature importance per il modello logaritmico
    feature_importance_log = pd.DataFrame({
        'feature': X_log.columns,
        'importance': pipeline_log.named_steps['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance per modello logaritmico:")
    print(feature_importance_log)
    
    # 6. VISUALIZZAZIONI
    print("Creazione visualizzazioni...")
    
    plt.figure(figsize=(18, 12))
    
    # Grafico 1: Relazione logaritmica
    plt.subplot(3, 3, 1)
    plt.errorbar(delta_means, a_eff_means, yerr=a_eff_std, fmt='o', alpha=0.7, label='Dati')
    
    x_fit = np.linspace(min(delta_means), max(delta_means), 100)
    y_fit_log = log_func(x_fit, *popt_log)
    plt.plot(x_fit, y_fit_log, 'r-', label=f'Logaritmica: y = {popt_log[0]:.3f} + {popt_log[1]:.3f}·ln(1+δ)')
    
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato (medio)')
    plt.title('Relazione logaritmica tra δ e A_eff')
    plt.legend()
    
    # Grafico 2: Relazione teorica per bin di ricchezza
    plt.subplot(3, 3, 2)
    theoretical_df = pd.DataFrame(theoretical_results)
    
    for _, row in theoretical_df.iterrows():
        y_fit_theo = theoretical_relation(x_fit, row['A0'], row['beta'])
        plt.plot(x_fit, y_fit_theo, label=f'{row["bin"]} (λ={row["lambda_mean"]:.1f}, β={row["beta"]:.3f})')
    
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('Relazione teorica per bin di ricchezza')
    plt.legend()
    
    # Grafico 3: Confronto R² per bin di ricchezza
    plt.subplot(3, 3, 3)
    if len(theoretical_df) > 0:
        plt.bar(range(len(theoretical_df)), theoretical_df['r2'])
        plt.xticks(range(len(theoretical_df)), theoretical_df['bin'], rotation=45)
        plt.ylabel('R²')
        plt.title('R² della relazione teorica per bin di ricchezza')
    
    # Grafico 4: Feature importance con interazione
    plt.subplot(3, 3, 4)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance con interazione')
    
    # Grafico 5: Feature importance per modello logaritmico
    plt.subplot(3, 3, 5)
    plt.barh(feature_importance_log['feature'], feature_importance_log['importance'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance per modello logaritmico')
    
    # Grafico 6: Confronto cross-validazione
    plt.subplot(3, 3, 6)
    models = ['Con interazione', 'Logaritmico']
    means = [cv_scores.mean(), cv_scores_log.mean()]
    stds = [cv_scores.std(), cv_scores_log.std()]
    
    plt.bar(models, means, yerr=stds, capsize=5)
    plt.ylabel('R² (cross-validazione)')
    plt.title('Confronto modelli con cross-validazione')
    
    # Grafico 7: Beta vs ricchezza media
    plt.subplot(3, 3, 7)
    if len(theoretical_df) > 0:
        plt.errorbar(theoretical_df['lambda_mean'], theoretical_df['beta'],
                    yerr=theoretical_df['beta_err'], fmt='o', capsize=5)
        plt.xlabel('Ricchezza media (λ)')
        plt.ylabel('β')
        plt.title('Relazione tra β e ricchezza')
    
    # Grafico 8: A0 vs ricchezza media
    plt.subplot(3, 3, 8)
    if len(theoretical_df) > 0:
        plt.errorbar(theoretical_df['lambda_mean'], theoretical_df['A0'],
                    yerr=theoretical_df['A0_err'], fmt='o', capsize=5)
        plt.xlabel('Ricchezza media (λ)')
        plt.ylabel('A0')
        plt.title('Relazione tra A0 e ricchezza')
    
    # Grafico 9: Relazione tra A0 e beta
    plt.subplot(3, 3, 9)
    if len(theoretical_df) > 0:
        plt.errorbar(theoretical_df['A0'], theoretical_df['beta'],
                    xerr=theoretical_df['A0_err'], yerr=theoretical_df['beta_err'],
                    fmt='o', capsize=5)
        plt.xlabel('A0')
        plt.ylabel('β')
        plt.title('Relazione tra A0 e β')
    
    plt.tight_layout()
    plt.savefig('analisi_fase5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. SALVA I RISULTATI
    theoretical_df.to_csv('risultati_relazione_teorica.csv', index=False)
    
    # 8. RIEPILOGO RISULTATI
    print("\nRIEPILOGO RISULTATI FASE 5:")
    print(f"1. R² per relazione logaritmica: {r2_log:.4f}")
    print(f"2. R² medio dalla cross-validazione (modello con interazione): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"3. R² medio dalla cross-validazione (modello logaritmico): {cv_scores_log.mean():.4f} (±{cv_scores_log.std():.4f})")
    
    print("4. Risultati relazione teorica per bin di ricchezza:")
    for _, row in theoretical_df.iterrows():
        print(f"   {row['bin']}: A0 = {row['A0']:.3f} ± {row['A0_err']:.3f}, β = {row['beta']:.3f} ± {row['beta_err']:.3f}, R² = {row['r2']:.4f}")

if __name__ == "__main__":
    main()
