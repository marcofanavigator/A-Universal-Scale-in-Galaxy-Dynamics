import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord

# Configurazione cosmologia (usando i valori Planck)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def main():
    # 1. CARICAMENTO E PREPROCESSAMENTO DEI DATI
    print("Caricamento dati ammassi...")
    # Carica il catalogo degli ammassi (solo colonne necessarie)
    cluster_cols = ['ID', 'RAJ2000', 'DEJ2000', 'zlambda', 'lambda', 'S']
    clusters = pd.read_csv('J_ApJS_224_1_cat_dr8.csv', usecols=cluster_cols)
    
    # Filtra per qualità e redshift
    clusters = clusters[
        (clusters['lambda'] > 20) & 
        (clusters['zlambda'] > 0.08) & 
        (clusters['zlambda'] < 0.6)
    ].copy()
    
    print(f"Numero di ammassi dopo filtraggio: {len(clusters)}")
    
    # 2. CALCOLO DELLA MASSA BARIONICA (PROXY) E A_EFF
    # Supponendo che S sia la dispersione di velocità σ
    # Usiamo lambda come proxy per la massa barionica: M_bar ∝ λ
    clusters['A_eff_proxy'] = (clusters['S']**2) / np.sqrt(clusters['lambda'])
    
    # 3. CARICAMENTO E PREPARAZIONE DEI DATI DEI MEMBRI
    print("Caricamento dati membri...")
    # Carica solo le colonne necessarie dal file membri
    member_cols = ['ID', 'RAJ2000', 'DEJ2000', 'PMem']
    members = pd.read_csv('J_ApJS_224_1_mmb_dr8.csv', usecols=member_cols)
    
    # Unisci i redshift degli ammassi ai membri
    members = members.merge(
        clusters[['ID', 'zlambda']], 
        on='ID', 
        how='inner'
    )
    
    # 4. CALCOLO DELLE COORDINATE COMOVENTI
    print("Calcolo coordinate comoventi...")
    
    # Funzione per convertire RA, Dec, z in coordinate cartesiane comoventi
    def ra_dec_z_to_cartesian(ra, dec, z):
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=cosmo.comoving_distance(z))
        return coord.cartesian.xyz.value  # restituisce (x, y, z) in Mpc
    
    # Calcola coordinate per gli ammassi (vettorialmente)
    coords = []
    for _, row in clusters.iterrows():
        x, y, z = ra_dec_z_to_cartesian(row['RAJ2000'], row['DEJ2000'], row['zlambda'])
        coords.append([x, y, z])
    
    clusters[['x', 'y', 'z']] = pd.DataFrame(coords, index=clusters.index)
    
    # Calcola coordinate per i membri (vettorialmente)
    coords = []
    for _, row in members.iterrows():
        x, y, z = ra_dec_z_to_cartesian(row['RAJ2000'], row['DEJ2000'], row['zlambda'])
        coords.append([x, y, z])
    
    members[['x', 'y', 'z']] = pd.DataFrame(coords, index=members.index)
    
    # 5. CALCOLO DELLA SOVRADDENSITÀ AMBIENTALE (δ)
    print("Calcolo sovraddensità ambientale...")
    
    # Crea un KDTree con tutte le posizioni dei membri
    all_member_positions = members[['x', 'y', 'z']].values
    tree = KDTree(all_member_positions)
    
    # Calcola la densità media dell'intero campione
    # Trova il volume approssimativo del campione
    x_min, x_max = members['x'].min(), members['x'].max()
    y_min, y_max = members['y'].min(), members['y'].max()
    z_min, z_max = members['z'].min(), members['z'].max()
    
    volume_totale = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    n_membri_totale = len(members)
    densità_media = n_membri_totale / volume_totale
    
    # Per ogni ammasso, calcola il numero di membri in un raggio di 10 Mpc
    raggio = 10  # Mpc
    
    # MODIFICA: query_radius restituisce una lista di array, non possiamo usare count_only
    # Calcoliamo la lunghezza di ogni array risultante
    indici_vicini = tree.query_radius(clusters[['x', 'y', 'z']].values, r=raggio)
    clusters['n_vicini'] = [len(indici) for indici in indici_vicini]
    
    # Calcola δ = (densità_locale / densità_media) - 1
    volume_sfera = (4/3) * np.pi * raggio**3
    clusters['delta'] = (clusters['n_vicini'] / volume_sfera) / densità_media - 1
    
    # 6. ANALISI DI REGRESSIONE
    print("Analisi di regressione...")
    
    # Prepara i dati per la regressione
    X = clusters[['lambda', 'zlambda', 'delta']]
    y = clusters['A_eff_proxy']
    
    # Rimuovi eventuali valori NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Dividi in train e test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Addestra il modello Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Valuta il modello
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R2 score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance)
    
    # 7. VISUALIZZAZIONI
    plt.figure(figsize=(10, 6))
    
    # Grafico A_eff vs delta
    plt.subplot(1, 2, 1)
    plt.scatter(clusters['delta'], clusters['A_eff_proxy'], alpha=0.5)
    plt.xlabel('Sovraddensità ambientale (δ)')
    plt.ylabel('A_eff proxy')
    plt.title('A_eff vs Sovraddensità')
    
    # Grafico delle feature importance
    plt.subplot(1, 2, 2)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importanza')
    plt.title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('risultati_analisi.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. SALVA I RISULTATI
    clusters.to_csv('clusters_con_delta.csv', index=False)
    print("Analisi completata. Risultati salvati in 'clusters_con_delta.csv'")

if __name__ == "__main__":
    main()
