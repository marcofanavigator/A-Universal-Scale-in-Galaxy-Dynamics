import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings('ignore')

# Configurazione cosmologia (usando i valori Planck)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def main():
    # 1. CARICA I DATI GIÀ PROCESSATI
    print("Caricamento dati già processati...")
    try:
        clusters = pd.read_csv('clusters_con_delta.csv')
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        print("Assicurati di aver eseguito prima la prima fase dell'analisi")
        return
    
    # 2. NORMALIZZAZIONE COSMOLOGICA DI A_EFF
    print("Normalizzazione cosmologica di A_eff...")
    
    # Trova la relazione tra A_eff_proxy e zlambda
    z = clusters['zlambda']
    a_eff = clusters['A_eff_proxy']
    
    # Fit polinomiale per trovare la dipendenza da z
    coeffs = np.polyfit(z, a_eff, 2)
    z_fit = np.linspace(z.min(), z.max(), 100)
    a_eff_fit = np.polyval(coeffs, z_fit)
    
    # Normalizza A_eff rimuovendo la dipendenza da z
    clusters['A_eff_norm'] = a_eff / np.polyval(coeffs, z)
    
    # 3. ANALISI CON A_EFF NORMALIZZATO
    print("Analisi con A_eff normalizzato...")
    
    # Regressione con A_eff normalizzato
    X = clusters[['lambda', 'delta']]
    y = clusters['A_eff_norm']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_norm = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_norm.fit(X_train, y_train)
    
    y_pred = rf_norm.predict(X_test)
    r2_norm = r2_score(y_test, y_pred)
    
    print(f"R2 con A_eff normalizzato: {r2_norm:.4f}")
    print("Feature importance con A_eff normalizzato:")
    feature_importance_norm = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_norm.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance_norm)
    
    # 4. ANALISI DELLA RELAZIONE FUNZIONALE
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
    
    # Fit della relazione
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    # Rimuovi eventuali valori NaN
    valid_idx = ~np.isnan(delta_means) & ~np.isnan(a_eff_means)
    delta_means = np.array(delta_means)[valid_idx]
    a_eff_means = np.array(a_eff_means)[valid_idx]
    a_eff_std = np.array(a_eff_std)[valid_idx]
    
    try:
        popt, pcov = curve_fit(exp_func, delta_means, a_eff_means, 
                              sigma=a_eff_std, absolute_sigma=True, p0=[1, -0.1])
        x_fit = np.linspace(min(delta_means), max(delta_means), 100)
        y_fit = exp_func(x_fit, *popt)
        fit_success = True
    except Exception as e:
        print(f"Fit esponenziale fallito: {e}")
        fit_success = False
    
    # 5. ANALISI PER SCALE DIVERSE
    print("Analisi per scale diverse...")
    
    # Prova diversi raggi per il calcolo di delta
    radii = [5, 10, 15, 20]  # Mpc
    results = []
    
    # Ricarica i dati membri per calcolare delta con raggi diversi
    try:
        member_cols = ['ID', 'RAJ2000', 'DEJ2000', 'PMem']
        members = pd.read_csv('J_ApJS_224_1_mmb_dr8.csv', usecols=member_cols)
        members = members.merge(clusters[['ID', 'zlambda']], on='ID', how='inner')
        
        # Calcola coordinate per i membri
        def ra_dec_z_to_cartesian(ra, dec, z):
            coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=cosmo.comoving_distance(z))
            return coord.cartesian.xyz.value
        
        coords = []
        for _, row in members.iterrows():
            x, y, z = ra_dec_z_to_cartesian(row['RAJ2000'], row['DEJ2000'], row['zlambda'])
            coords.append([x, y, z])
        
        members[['x', 'y', 'z']] = pd.DataFrame(coords, index=members.index)
        
        all_member_positions = members[['x', 'y', 'z']].values
        tree = KDTree(all_member_positions)
        
        for radius in radii:
            print(f"Calcolo delta per raggio {radius} Mpc...")
            indici_vicini = tree.query_ball_point(clusters[['x', 'y', 'z']].values, r=radius)
            n_vicini = [len(indici) for indici in indici_vicini]
            
            # Calcola δ per questo raggio
            volume_sfera = (4/3) * np.pi * radius**3
            x_min, x_max = members['x'].min(), members['x'].max()
            y_min, y_max = members['y'].min(), members['y'].max()
            z_min, z_max = members['z'].min(), members['z'].max()
            volume_totale = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
            densità_media = len(members) / volume_totale
            
            delta_radius = (np.array(n_vicini) / volume_sfera) / densità_media - 1
            
            # Regressione con questo delta
            X_radius = pd.DataFrame({'lambda': clusters['lambda'], 'delta': delta_radius})
            y_radius = clusters['A_eff_norm']
            
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_radius, y_radius, test_size=0.3, random_state=42
            )
            
            rf_radius = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_radius.fit(X_train_r, y_train_r)
            
            y_pred_r = rf_radius.predict(X_test_r)
            r2_radius = r2_score(y_test_r, y_pred_r)
            
            results.append({
                'radius': radius,
                'r2': r2_radius,
                'delta_importance': rf_radius.feature_importances_[1]
            })
    
    except Exception as e:
        print(f"Errore nel calcolo per scale diverse: {e}")
        results = []
    
    # 6. VISUALIZZAZIONI
    print("Creazione visualizzazioni...")
    
    plt.figure(figsize=(15, 12))
    
    # Grafico 1: Dipendenza da redshift
    plt.subplot(3, 2, 1)
    plt.scatter(clusters['zlambda'], clusters['A_eff_proxy'], alpha=0.1, s=5)
    plt.plot(z_fit, a_eff_fit, 'r-', linewidth=2)
    plt.xlabel('Redshift (z)')
    plt.ylabel('A_eff_proxy')
    plt.title('Dipendenza di A_eff dal redshift')
    
    # Grafico 2: A_eff normalizzato vs delta
    plt.subplot(3, 2, 2)
    plt.scatter(clusters['delta'], clusters['A_eff_norm'], alpha=0.1, s=5)
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff normalizzato')
    plt.title('A_eff normalizzato vs Sovraddensità')
    
    # Grafico 3: Relazione funzionale
    plt.subplot(3, 2, 3)
    plt.errorbar(delta_means, a_eff_means, yerr=a_eff_std, fmt='o', alpha=0.7)
    if fit_success:
        plt.plot(x_fit, y_fit, 'r-', label=f'y = {popt[0]:.3f} exp({popt[1]:.3f} x)')
        plt.legend()
    plt.xlabel('Delta (sovraddensità)')
    plt.ylabel('A_eff normalizzato (medio)')
    plt.title('Relazione tra delta e A_eff normalizzato')
    
    # Grafico 4: Feature importance con A_eff normalizzato
    plt.subplot(3, 2, 4)
    plt.barh(feature_importance_norm['feature'], feature_importance_norm['importance'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance con A_eff normalizzato')
    
    # Grafico 5: Performance per diversi raggi
    if results:
        plt.subplot(3, 2, 5)
        radii = [r['radius'] for r in results]
        r2_scores = [r['r2'] for r in results]
        delta_importances = [r['delta_importance'] for r in results]
        
        plt.plot(radii, r2_scores, 'o-', label='R²')
        plt.plot(radii, delta_importances, 's-', label='Importanza delta')
        plt.xlabel('Raggio (Mpc)')
        plt.ylabel('Valore')
        plt.legend()
        plt.title('Performance per diversi raggi')
    
    # Grafico 6: Distribuzione di A_eff normalizzato
    plt.subplot(3, 2, 6)
    plt.hist(clusters['A_eff_norm'], bins=50, alpha=0.7)
    plt.xlabel('A_eff normalizzato')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione di A_eff normalizzato')
    
    plt.tight_layout()
    plt.savefig('analisi_fase2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. SALVA I RISULTATI
    clusters.to_csv('clusters_con_delta_e_norm.csv', index=False)
    
    print("Analisi completata. Risultati salvati in:")
    print("- clusters_con_delta_e_norm.csv")
    print("- analisi_fase2.png")
    
    # 8. RIEPILOGO RISULTATI
    print("\nRIEPILOGO RISULTATI:")
    print(f"1. R2 con A_eff normalizzato: {r2_norm:.4f}")
    print(f"2. Importanza delle features con A_eff normalizzato:")
    for _, row in feature_importance_norm.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    if fit_success:
        print(f"3. Relazione funzionale: A_eff_norm = {popt[0]:.3f} * exp({popt[1]:.3f} * delta)")
    
    if results:
        print("4. Risultati per diversi raggi:")
        for result in results:
            print(f"   Raggio {result['radius']} Mpc: R² = {result['r2']:.4f}, Importanza delta = {result['delta_importance']:.4f}")

if __name__ == "__main__":
    main()
