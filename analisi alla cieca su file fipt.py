#!/usr/bin/env python3
"""
Test di predizione cieca del framework GPF non-locale sui dati ACT-DR5
Basato sul paper: The Cosmic Tension Field: A Non-Local Generalized Poisson Framework

Implementa il test cruciale per verificare la robustezza del modello logaritmico:
A_norm_eff = 3.146 - 0.367 * ln(1 + δ)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.spatial import cKDTree
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Cosmologia come nel paper originale
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class ACT_DR5_Analyzer:
    def __init__(self):
        self.clusters = None
        self.mask = None
        self.multiple_systems = None
        
    def load_fits_files(self, catalog_file, mask_file, multiple_file):
        """Carica i file FITS ACT-DR5"""
        print("Caricamento dati ACT-DR5...")
        
        # Catalog principale
        with fits.open(catalog_file) as hdul:
            self.clusters = hdul[1].data
            print(f"Colonne catalogo: {self.clusters.columns.names}")
            print(f"Numero cluster: {len(self.clusters)}")
        
        # Mask dell'area di survey
        with fits.open(mask_file) as hdul:
            self.mask = hdul[1].data
            print(f"Mask shape: {self.mask.shape if hasattr(self.mask, 'shape') else 'scalar'}")
        
        # Sistemi multipli
        with fits.open(multiple_file) as hdul:
            self.multiple_systems = hdul[1].data
            print(f"Sistemi multipli: {len(self.multiple_systems) if self.multiple_systems is not None else 0}")
    
    def explore_data_structure(self):
        """Esplora la struttura dei dati"""
        print("\n=== STRUTTURA DATI ACT-DR5 ===")
        
        # Info sul catalogo principale
        print(f"\nCatalogo cluster:")
        print(f"Entries: {len(self.clusters)}")
        
        # Stampa alcune colonne chiave
        key_cols = []
        for col in self.clusters.columns.names:
            if any(keyword in col.lower() for keyword in ['ra', 'dec', 'redshift', 'z', 'mass', 'snr', 'rich']):
                key_cols.append(col)
        
        print(f"Colonne chiave identificate: {key_cols}")
        
        # Statistiche base
        if 'redshift' in [col.lower() for col in self.clusters.columns.names]:
            z_col = next(col for col in self.clusters.columns.names if col.lower() == 'redshift')
            print(f"Range redshift: {np.min(self.clusters[z_col]):.3f} - {np.max(self.clusters[z_col]):.3f}")
        
        return key_cols
    
    def calculate_environmental_overdensity(self, k=5, radius_mpc=10):
        """
        Calcola l'overdensity ambientale δ per ogni cluster
        Usando la stessa metodologia del paper originale
        """
        print(f"\nCalcolo overdensity ambientale (k={k}, R={radius_mpc} Mpc)...")
        
        # Identifica colonne coordinate
        ra_col = None
        dec_col = None
        z_col = None
        
        for col in self.clusters.columns.names:
            if 'ra' in col.lower() and ra_col is None:
                ra_col = col
            if 'dec' in col.lower() and dec_col is None:
                dec_col = col
            if any(x in col.lower() for x in ['redshift', 'z']) and 'err' not in col.lower():
                z_col = col
        
        print(f"Usando colonne: RA={ra_col}, DEC={dec_col}, Z={z_col}")
        
        # Coordinate
        ra = self.clusters[ra_col]
        dec = self.clusters[dec_col]
        z = self.clusters[z_col]
        
        # Converti a coordinate comoventi
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        
        # Distanza comotiva
        r_comoving = cosmo.comoving_distance(z).value  # Mpc
        
        # Converti a coordinate cartesiane comoventi
        x = r_comoving * np.cos(coords.dec.radian) * np.cos(coords.ra.radian)
        y = r_comoving * np.cos(coords.dec.radian) * np.sin(coords.ra.radian)
        z_cart = r_comoving * np.sin(coords.dec.radian)
        
        positions = np.column_stack([x, y, z_cart])
        
        # Costruisci k-NN tree
        tree = cKDTree(positions)
        
        # Per ogni cluster, trova densità locale
        overdensities = []
        rho_crit = cosmo.critical_density(0).to(u.Msun/u.Mpc**3).value
        
        for i, pos in enumerate(positions):
            # Trova vicini entro radius_mpc
            indices = tree.query_ball_point(pos, radius_mpc)
            n_neighbors = len(indices) - 1  # Escludi se stesso
            
            # Volume della sfera
            volume = (4/3) * np.pi * radius_mpc**3
            
            # Densità locale (assumiamo massa tipica cluster ~ 1e14 Msun)
            rho_local = n_neighbors * 1e14 / volume
            
            # Overdensity
            delta = rho_local / rho_crit - 1
            overdensities.append(delta)
            
            if i % 1000 == 0:
                print(f"Processato {i}/{len(positions)} cluster...")
        
        return np.array(overdensities)
    
    def predict_aeff_blind(self, overdensities):
        """
        Applica il modello logaritmico del paper per predizione cieca
        A_norm_eff = 3.146 - 0.367 * ln(1 + δ)
        """
        print("\n=== PREDIZIONE CIECA ===")
        print("Applicando modello logaritmico dal paper RedMaPPer...")
        
        # Parametri dal paper
        a = 3.146
        b = -0.367
        
        # Predizioni
        ln_term = np.log(1 + np.maximum(overdensities, -0.99))  # Evita log di numeri negativi
        A_pred = a + b * ln_term
        
        print(f"Parametri usati: a={a}, b={b}")
        print(f"Range overdensity: {np.min(overdensities):.3f} - {np.max(overdensities):.3f}")
        print(f"Range A_eff predetto: {np.min(A_pred):.3f} - {np.max(A_pred):.3f}")
        
        return A_pred
    
    def save_blind_predictions(self, overdensities, predictions, filename='act_dr5_blind_predictions.csv'):
        """Salva le predizioni cieche PRIMA di calcolare A_eff osservato"""
        
        # Identifica colonne ID
        id_cols = [col for col in self.clusters.columns.names if 'id' in col.lower() or 'name' in col.lower()]
        
        df_pred = pd.DataFrame({
            'cluster_index': range(len(self.clusters)),
            'overdensity_delta': overdensities,
            'A_eff_predicted': predictions,
            'model_used': 'logarithmic_redmapper',
            'timestamp': pd.Timestamp.now()
        })
        
        # Aggiungi coordinate per identificazione
        ra_col = next(col for col in self.clusters.columns.names if 'ra' in col.lower())
        dec_col = next(col for col in self.clusters.columns.names if 'dec' in col.lower())
        z_col = next(col for col in self.clusters.columns.names if any(x in col.lower() for x in ['redshift', 'z']))
        
        df_pred['RA'] = self.clusters[ra_col]
        df_pred['DEC'] = self.clusters[dec_col] 
        df_pred['redshift'] = self.clusters[z_col]
        
        df_pred.to_csv(filename, index=False)
        print(f"\n✓ PREDIZIONI CIECHE salvate in: {filename}")
        print(f"  IMPORTANTE: Non accedere ai dati spettroscopici fino a dopo aver pubblicato queste predizioni!")
        
        return df_pred
    
    def plot_predictions(self, overdensities, predictions):
        """Visualizza le predizioni cieche"""
        plt.figure(figsize=(12, 8))
        
        # Plot principale
        plt.subplot(2, 2, 1)
        plt.scatter(overdensities, predictions, alpha=0.6, s=20)
        plt.xlabel('Environmental overdensity δ')
        plt.ylabel('Predicted A_eff')
        plt.title('Blind Predictions: A_eff vs δ')
        plt.grid(True, alpha=0.3)
        
        # Distribuzione overdensity
        plt.subplot(2, 2, 2)
        plt.hist(overdensities, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Environmental overdensity δ')
        plt.ylabel('Count')
        plt.title('Distribution of δ in ACT-DR5')
        plt.grid(True, alpha=0.3)
        
        # Distribuzione predizioni
        plt.subplot(2, 2, 3)
        plt.hist(predictions, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Predicted A_eff')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted A_eff')
        plt.grid(True, alpha=0.3)
        
        # Modello teorico
        plt.subplot(2, 2, 4)
        delta_theory = np.linspace(np.min(overdensities), np.max(overdensities), 100)
        A_theory = 3.146 - 0.367 * np.log(1 + np.maximum(delta_theory, -0.99))
        plt.plot(delta_theory, A_theory, 'r-', linewidth=2, label='Logarithmic model')
        plt.scatter(overdensities, predictions, alpha=0.3, s=10, label='ACT predictions')
        plt.xlabel('Environmental overdensity δ')
        plt.ylabel('A_eff')
        plt.title('Theoretical Model vs Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('act_dr5_blind_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Script principale per il test di predizione cieca"""
    
    print("=== TEST PREDIZIONE CIECA GPF NON-LOCALE ===")
    print("Implementazione del test cruciale per verificare il framework")
    
    # Inizializza analyzer
    analyzer = ACT_DR5_Analyzer()
    
    # Carica dati (modifica i path secondo la tua directory)
    catalog_file = "DR5_cluster-catalog_v1.1.fits"
    mask_file = "DR5_cluster-search-area-mask_v1.0.fits"
    multiple_file = "DR5_multiple-systems_v1.0-1.fits"
    
    try:
        analyzer.load_fits_files(catalog_file, mask_file, multiple_file)
        
        # Esplora struttura
        key_columns = analyzer.explore_data_structure()
        
        # Calcola overdensity ambientale
        overdensities = analyzer.calculate_environmental_overdensity(k=5, radius_mpc=10)
        
        # Predizioni cieche usando il modello dal paper
        predictions = analyzer.predict_aeff_blind(overdensities)
        
        # Salva predizioni (FONDAMENTALE!)
        df_predictions = analyzer.save_blind_predictions(overdensities, predictions)
        
        # Visualizza
        analyzer.plot_predictions(overdensities, predictions)
        
        print("\n=== PROSSIMI PASSI ===")
        print("1. ✓ Predizioni cieche generate e salvate")
        print("2. → Pubblicare/registrare queste predizioni PRIMA di continuare")
        print("3. → Raccogliere dati spettroscopici per i cluster ACT")
        print("4. → Calcolare A_eff osservato")
        print("5. → Confrontare con predizioni per test finale")
        
        print(f"\nStatistiche predizioni:")
        print(f"Media δ: {np.mean(overdensities):.3f}")
        print(f"Std δ: {np.std(overdensities):.3f}")
        print(f"Media A_eff pred: {np.mean(predictions):.3f}")
        print(f"Std A_eff pred: {np.std(predictions):.3f}")
        print(f"Range A_eff: {np.min(predictions):.3f} - {np.max(predictions):.3f}")
        
    except FileNotFoundError as e:
        print(f"\nERRORE: File non trovato - {e}")
        print("Assicurati che i file FITS siano nella directory corrente:")
        print("- DR5_cluster-catalog_v1.1.fits")
        print("- DR5_cluster-search-area-mask_v1.0.fits") 
        print("- DR5_multiple-systems_v1.0-1.fits")
    
    except Exception as e:
        print(f"\nERRORE durante l'analisi: {e}")
        print("Controllando struttura dati...")
        
        # Debug: stampa info sui file
        try:
            with fits.open(catalog_file) as hdul:
                hdul.info()
                if len(hdul) > 1:
                    print(f"\nColonne disponibili: {hdul[1].data.columns.names}")
        except:
            print("Impossibile aprire il catalogo principale")

def analyze_column_structure(fits_file):
    """Funzione helper per esplorare la struttura di un file FITS"""
    print(f"\n=== ANALISI STRUTTURA: {fits_file} ===")
    try:
        with fits.open(fits_file) as hdul:
            hdul.info()
            
            for i, ext in enumerate(hdul):
                print(f"\nExtension {i}: {ext.name}")
                if hasattr(ext, 'data') and ext.data is not None:
                    if hasattr(ext.data, 'columns'):
                        print(f"Colonne: {ext.data.columns.names}")
                        print(f"Numero righe: {len(ext.data)}")
                        
                        # Mostra primi record per colonne chiave
                        for col in ext.data.columns.names[:5]:
                            sample_values = ext.data[col][:3]
                            print(f"  {col}: {sample_values}")
                    else:
                        print(f"Shape: {ext.data.shape}")
                        
    except Exception as e:
        print(f"Errore nell'analisi: {e}")

if __name__ == "__main__":
    print("Opzioni:")
    print("1. Analizza struttura files FITS")
    print("2. Esegui test predizione cieca completo")
    
    choice = input("\nScegli opzione (1 o 2): ")
    
    if choice == "1":
        # Analizza struttura
        files = ["DR5_cluster-catalog_v1.1.fits", 
                "DR5_cluster-search-area-mask_v1.0.fits",
                "DR5_multiple-systems_v1.0-1.fits"]
        
        for f in files:
            analyze_column_structure(f)
    
    elif choice == "2":
        # Test completo
        main()
    
    else:
        print("Opzione non valida")

# Requisiti:
# pip install astropy numpy matplotlib pandas scikit-learn scipy
