#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Advanced CME Detection System
Multi-Algorithm Approach for Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data

Based on scientific models from sceintifcmodelstomake.txt:
- Pearson and Spearman correlations
- Mutual Information analysis
- Granger Causality for temporal relationships
- Advanced composite scoring
- Cross-validation with Richardson-Cane and CACTUS databases
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import signal, stats
from scipy.stats import zscore, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Import hello functionality
try:
    from hello import say_hello
except ImportError:
    def say_hello(name=None):
        """Fallback hello function if hello.py is not available"""
        return "Hello! Welcome to the ISRO CME Detection System."

# Configuration
OUTPUT_DIR = "cme_detection_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("ISRO HACKATHON PS10 - ADVANCED CME DETECTION SYSTEM")
print("Multi-Algorithm Approach with Scientific Correlation Models")
print("="*80)

class CMEDetectionSystem:
    """Advanced CME Detection System using multiple scientific algorithms"""
    
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.models = {}
        self.feature_importance = {}
        self.thresholds = {}
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")
        
        try:
            # Load main SWIS data
            self.swis_data = pd.read_csv(os.path.join(self.output_dir, "integrated_swis_data.csv"))
            self.swis_data['datetime'] = pd.to_datetime(self.swis_data['timestamp'])
            print(f"✓ SWIS data: {len(self.swis_data):,} points")
            
            # Load CACTUS CME events
            self.cactus_events = pd.read_csv(os.path.join(self.output_dir, "cactus_cme_events.csv"))
            self.cactus_events['datetime'] = pd.to_datetime(self.cactus_events['datetime'])
            print(f"✓ CACTUS events: {len(self.cactus_events)} events")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def load_richardson_cane_data(self):
        """Parse Richardson-Cane ICME catalogue"""
        print("\\nLoading Richardson-Cane ICME catalogue...")
        
        try:
            # Read the Richardson-Cane file
            with open("richardsoncanehalo.txt", 'r') as f:
                lines = f.readlines()
            
            # Parse the data (skip headers and find data lines)
            rc_events = []
            for line in lines:
                if line.strip() and not line.startswith(('Near-Earth', 'Compiled', 'Revised', 
                                                        'Note', 'Disturbance', '(1)', '(2)', '(3)')):
                    # Parse Richardson-Cane format
                    parts = line.strip().split('\\t')
                    if len(parts) >= 3 and '/' in parts[0]:
                        try:
                            # Extract disturbance time
                            dist_time_str = parts[0].split()[0]  # Get Y/M/D part
                            if len(dist_time_str.split('/')) == 3:
                                year, month, day = dist_time_str.split('/')
                                if len(year) == 4:  # Valid year format
                                    rc_events.append({
                                        'datetime': pd.to_datetime(f"{year}-{month}-{day}"),
                                        'source': 'Richardson-Cane',
                                        'raw_line': line.strip()
                                    })
                        except:
                            continue
            
            self.richardson_cane = pd.DataFrame(rc_events)
            print(f"✓ Richardson-Cane events: {len(self.richardson_cane)} ICMEs")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading Richardson-Cane data: {e}")
            return False
    
    def load_cactus_catalogue(self):
        """Parse CACTUS CME catalogue"""
        print("\\nLoading CACTUS CME catalogue...")
        
        try:
            with open("cactuscmeevents.txt", 'r') as f:
                lines = f.readlines()
            
            cactus_events = []
            for line in lines:
                if line.strip() and '|' in line and not line.startswith(('#', ':', ' CME')):
                    try:
                        # Parse CACTUS format: CME | t0 | dt0| pa | da | v | dv | minv| maxv| halo?
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 9:
                            time_str = parts[1].strip()
                            velocity = int(parts[5].strip())
                            angular_width = int(parts[4].strip())
                            halo_type = parts[9].strip() if len(parts) > 9 else ''
                            
                            # Parse datetime
                            dt = pd.to_datetime(time_str)
                            
                            # Classify halo type
                            is_halo = angular_width >= 180 or 'III' in halo_type or 'IV' in halo_type
                            is_partial_halo = angular_width >= 90 or 'II' in halo_type
                            
                            cactus_events.append({
                                'datetime': dt,
                                'velocity': velocity,
                                'angular_width': angular_width,
                                'halo_type': halo_type,
                                'is_halo': is_halo,
                                'is_partial_halo': is_partial_halo,
                                'is_fast': velocity >= 1000,
                                'source': 'CACTUS-Full'
                            })
                    except:
                        continue
            
            self.cactus_full = pd.DataFrame(cactus_events)
            print(f"✓ CACTUS catalogue: {len(self.cactus_full)} CMEs")
            print(f"  - Halo CMEs: {self.cactus_full['is_halo'].sum()}")
            print(f"  - Partial halo CMEs: {self.cactus_full['is_partial_halo'].sum()}")
            print(f"  - Fast CMEs (≥1000 km/s): {self.cactus_full['is_fast'].sum()}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading CACTUS catalogue: {e}")
            return False
    
    def compute_correlation_features(self):
        """Compute correlation-based features using scientific methods"""
        print("\\nComputing correlation-based features...")
        
        params = ['proton_density', 'proton_velocity', 'proton_temperature']
        
        # Pearson correlations (linear relationships)
        print("  Computing Pearson correlations...")
        pearson_corrs = {}
        for i, param1 in enumerate(params):
            for j, param2 in enumerate(params):
                if i < j:
                    corr, p_val = pearsonr(self.swis_data[param1].dropna(), 
                                         self.swis_data[param2].dropna())
                    pearson_corrs[f'{param1}_{param2}_pearson'] = corr
                    self.swis_data[f'{param1}_{param2}_pearson'] = corr
        
        # Spearman correlations (monotonic relationships)
        print("  Computing Spearman correlations...")
        spearman_corrs = {}
        for i, param1 in enumerate(params):
            for j, param2 in enumerate(params):
                if i < j:
                    corr, p_val = spearmanr(self.swis_data[param1].dropna(), 
                                          self.swis_data[param2].dropna())
                    spearman_corrs[f'{param1}_{param2}_spearman'] = corr
                    self.swis_data[f'{param1}_{param2}_spearman'] = corr
        
        # Mutual Information (non-linear dependencies)
        print("  Computing Mutual Information...")
        for param in params:
            mi_score = mutual_info_regression(
                self.swis_data[params].fillna(0), 
                self.swis_data[param].fillna(0)
            ).mean()
            self.swis_data[f'{param}_mutual_info'] = mi_score
        
        return pearson_corrs, spearman_corrs
    
    def compute_physics_features(self):
        """Compute physics-based derived parameters"""
        print("\\nComputing physics-based features...")
        
        # Dynamic pressure (ρv²)
        self.swis_data['dynamic_pressure'] = (
            self.swis_data['proton_density'] * 
            self.swis_data['proton_velocity']**2
        )
        
        # Kinetic energy proxy
        self.swis_data['kinetic_energy'] = (
            0.5 * self.swis_data['proton_density'] * 
            self.swis_data['proton_velocity']**2
        )
        
        # Beta parameter (thermal vs magnetic pressure)
        self.swis_data['beta_proxy'] = (
            self.swis_data['proton_temperature'] / 
            self.swis_data['proton_velocity']
        )
        
        # Enhancement factors
        self.swis_data['velocity_enhancement'] = self.swis_data['proton_velocity'] / 400.0
        self.swis_data['density_enhancement'] = self.swis_data['proton_density'] / 5.0
        self.swis_data['temp_enhancement'] = self.swis_data['proton_temperature'] / 50.0
        
        # Combined enhancement score
        self.swis_data['combined_enhancement'] = np.sqrt(
            self.swis_data['velocity_enhancement']**2 + 
            self.swis_data['density_enhancement']**2 + 
            self.swis_data['temp_enhancement']**2
        )
        
        print("  ✓ Physics-based features computed")
    
    def compute_temporal_features(self):
        """Compute temporal features for time series analysis"""
        print("\\nComputing temporal features...")
        
        # Sort by time
        self.swis_data = self.swis_data.sort_values('datetime').reset_index(drop=True)
        
        params = ['proton_density', 'proton_velocity', 'proton_temperature', 
                 'dynamic_pressure', 'combined_enhancement']
        
        for param in params:
            if param in self.swis_data.columns:
                # Moving averages (different time windows)
                self.swis_data[f'{param}_ma_1h'] = self.swis_data[param].rolling(12).mean()
                self.swis_data[f'{param}_ma_6h'] = self.swis_data[param].rolling(72).mean()
                self.swis_data[f'{param}_ma_24h'] = self.swis_data[param].rolling(288).mean()
                
                # Gradients (rate of change)
                self.swis_data[f'{param}_gradient'] = self.swis_data[param].diff()
                self.swis_data[f'{param}_gradient_6h'] = self.swis_data[param].diff(72)
                
                # Z-scores (anomaly detection)
                self.swis_data[f'{param}_zscore'] = (
                    self.swis_data[param] - self.swis_data[f'{param}_ma_24h']
                ) / self.swis_data[param].rolling(288).std()
                
                # Percentile ranking
                self.swis_data[f'{param}_percentile'] = (
                    self.swis_data[param].rolling(288).rank(pct=True)
                )
        
        print("  ✓ Temporal features computed")
    
    def compute_granger_causality_proxy(self):
        """Compute Granger causality-like features"""
        print("\\nComputing temporal precedence features...")
        
        # Simplified Granger causality proxy using lagged correlations
        params = ['proton_density', 'proton_velocity', 'proton_temperature']
        
        for param in params:
            # Correlation with CME events at different lags
            for lag in [1, 6, 12, 24]:  # 5min, 30min, 1h, 2h lags
                if len(self.swis_data) > lag:
                    lagged_param = self.swis_data[param].shift(lag)
                    corr = lagged_param.corr(self.swis_data['cme_event'])
                    self.swis_data[f'{param}_lag_{lag}_corr'] = corr
        
        print("  ✓ Temporal precedence features computed")
    
    def build_composite_score(self):
        """Build composite CME probability score using multiple methods"""
        print("\\nBuilding composite CME probability score...")
        
        # Select key features for scoring
        score_features = [
            'velocity_enhancement', 'density_enhancement', 'temp_enhancement',
            'dynamic_pressure', 'combined_enhancement',
            'proton_velocity_zscore', 'proton_density_zscore', 'proton_temperature_zscore'
        ]
        
        # Normalize features to 0-1 scale
        feature_scores = pd.DataFrame()
        for feature in score_features:
            if feature in self.swis_data.columns:
                values = self.swis_data[feature].fillna(0)
                # Clip extreme values and normalize
                values_clipped = np.clip(values, 
                                       values.quantile(0.01), 
                                       values.quantile(0.99))
                normalized = (values_clipped - values_clipped.min()) / (values_clipped.max() - values_clipped.min())
                feature_scores[feature] = normalized
        
        # Weighted composite score
        weights = {
            'velocity_enhancement': 0.25,
            'density_enhancement': 0.15,
            'temp_enhancement': 0.15,
            'dynamic_pressure': 0.20,
            'combined_enhancement': 0.25
        }
        
        composite_score = np.zeros(len(self.swis_data))
        for feature, weight in weights.items():
            if feature in feature_scores.columns:
                composite_score += weight * feature_scores[feature]
        
        self.swis_data['cme_probability_score'] = composite_score
        
        print("  ✓ Composite probability score computed")
    
    def train_ml_models(self):
        """Train multiple ML models for CME detection"""
        print("\\nTraining machine learning models...")
        
        # Prepare features
        feature_cols = [col for col in self.swis_data.columns 
                       if col not in ['timestamp', 'datetime', 'cme_event', 'cme_event_extended',
                                     'cme_type', 'cme_speed', 'file_date', 'source_file']]
        
        # Remove non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if self.swis_data[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        print(f"  Found {len(numeric_features)} numeric features")
        
        # Limit features to prevent memory issues
        if len(numeric_features) > 50:
            print(f"  Limiting to top 50 features to prevent memory issues...")
            # Use basic correlation to select top features
            correlations = []
            for col in numeric_features:
                try:
                    corr = abs(self.swis_data[col].corr(self.swis_data['cme_event']))
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass
            
            # Sort by correlation and take top 50
            correlations.sort(key=lambda x: x[1], reverse=True)
            numeric_features = [col for col, _ in correlations[:50]]
            print(f"  Selected top {len(numeric_features)} features")
        
        # Prepare data
        X = self.swis_data[numeric_features].fillna(0)
        y = self.swis_data['cme_event']
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Sample data if too large (keep it manageable)
        if len(X) > 50000:
            print(f"  Sampling {min(50000, len(X))} points to prevent memory issues...")
            sample_idx = np.random.choice(len(X), min(50000, len(X)), replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
        
        print(f"  Training data shape: {X.shape}")
        print(f"  CME events in training: {y.sum()} ({y.mean()*100:.2f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        print("  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (simplified)
        print("  Training Random Forest (optimized)...")
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=10,     # Limit depth
            min_samples_split=10,  # Increase min samples
            random_state=42, 
            class_weight='balanced',
            n_jobs=1  # Single thread to prevent hanging
        )
        rf_model.fit(X_train, y_train)
        print("    ✓ Random Forest trained")
        
        # Train Logistic Regression
        print("  Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=500
        )
        lr_model.fit(X_train_scaled, y_train)
        print("    ✓ Logistic Regression trained")
        
        # Train Isolation Forest (simplified)
        print("  Training Isolation Forest...")
        # Use smaller sample for Isolation Forest
        iso_sample_size = min(5000, len(X_train_scaled))
        iso_sample_idx = np.random.choice(len(X_train_scaled), iso_sample_size, replace=False)
        
        iso_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=50  # Reduced from default
        )
        iso_forest.fit(X_train_scaled[iso_sample_idx])
        print("    ✓ Isolation Forest trained")
        
        # Store models
        self.models = {
            'random_forest': rf_model,
            'logistic_regression': lr_model,
            'isolation_forest': iso_forest,
            'scaler': scaler
        }
        
        # Store feature names
        self.feature_names = numeric_features
        
        # Evaluate models
        self.evaluate_models(X_test, X_test_scaled, y_test, numeric_features)
        
        print("  ✓ Machine learning models trained")
    
    def evaluate_models(self, X_test, X_test_scaled, y_test, feature_names):
        """Evaluate trained models"""
        print("\\nEvaluating models...")
        
        results = {}
        
        # Random Forest evaluation
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_prob = self.models['random_forest'].predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_prob)
        
        results['Random Forest'] = {
            'AUC': rf_auc,
            'Accuracy': (rf_pred == y_test).mean(),
            'Predictions': rf_pred,
            'Probabilities': rf_prob
        }
        
        # Logistic Regression evaluation
        lr_pred = self.models['logistic_regression'].predict(X_test_scaled)
        lr_prob = self.models['logistic_regression'].predict_proba(X_test_scaled)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_prob)
        
        results['Logistic Regression'] = {
            'AUC': lr_auc,
            'Accuracy': (lr_pred == y_test).mean(),
            'Predictions': lr_pred,
            'Probabilities': lr_prob
        }
        
        # Feature importance from Random Forest
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        # Print results
        print("\\nModel Performance:")
        print("-" * 50)
        for model, metrics in results.items():
            print(f"{model:20s} AUC: {metrics['AUC']:.3f}, Accuracy: {metrics['Accuracy']:.3f}")
        
        # Save results
        self.model_results = results
        feature_importance.to_csv(
            os.path.join(self.output_dir, "ml_feature_importance.csv"), 
            index=False
        )
        
        return results
    
    def determine_optimal_thresholds(self):
        """Determine optimal thresholds using ROC analysis"""
        print("\\nDetermining optimal thresholds...")
        
        # Use Random Forest probabilities for threshold optimization
        y_true = self.swis_data['cme_event']
        
        # Get model predictions using stored feature names
        if hasattr(self, 'feature_names'):
            features = self.feature_names
        else:
            features = [col for col in self.swis_data.columns 
                       if col not in ['timestamp', 'datetime', 'cme_event', 'cme_event_extended',
                                     'cme_type', 'cme_speed', 'file_date', 'source_file'] 
                       and self.swis_data[col].dtype in ['int64', 'float64']]
        
        X = self.swis_data[features].fillna(0).replace([np.inf, -np.inf], 0)
        
        try:
            rf_prob = self.models['random_forest'].predict_proba(X)[:, 1]
            
            # ROC analysis
            fpr, tpr, thresholds = roc_curve(y_true, rf_prob)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Thresholds for different use cases
            self.thresholds = {
                'optimal': optimal_threshold,
                'high_sensitivity': thresholds[np.where(tpr >= 0.95)[0][0]] if any(tpr >= 0.95) else optimal_threshold,
                'high_specificity': thresholds[np.where(fpr <= 0.05)[0][-1]] if any(fpr <= 0.05) else optimal_threshold,
                'composite_score': np.percentile(self.swis_data['cme_probability_score'], 95)
            }
            
        except Exception as e:
            print(f"  Warning: Error in threshold determination: {e}")
            # Use fallback thresholds
            self.thresholds = {
                'optimal': 0.5,
                'high_sensitivity': 0.3,
                'high_specificity': 0.7,
                'composite_score': np.percentile(self.swis_data['cme_probability_score'], 95)
            }
        
        print(f"  Optimal threshold (Youden's J): {self.thresholds['optimal']:.3f}")
        print(f"  High sensitivity threshold: {self.thresholds['high_sensitivity']:.3f}")
        print(f"  High specificity threshold: {self.thresholds['high_specificity']:.3f}")
        print(f"  Composite score threshold (95th percentile): {self.thresholds['composite_score']:.3f}")
        
        return self.thresholds
    
    def validate_with_catalogues(self):
        """Cross-validate results with Richardson-Cane and CACTUS catalogues"""
        print("\\nValidating with external catalogues...")
        
        if hasattr(self, 'richardson_cane') and len(self.richardson_cane) > 0:
            print("  Validating with Richardson-Cane ICME catalogue...")
            try:
                # Find overlapping time periods
                rc_dates = self.richardson_cane['datetime']
                swis_dates = self.swis_data['datetime']
                
                # Count matches within ±2 days
                matches = 0
                for rc_date in rc_dates:
                    time_diff = np.abs((swis_dates - rc_date).dt.total_seconds() / 86400)  # days
                    if np.any(time_diff <= 2):
                        matches += 1
                
                print(f"    Richardson-Cane matches: {matches}/{len(rc_dates)} ({matches/len(rc_dates)*100:.1f}%)")
            except Exception as e:
                print(f"    Richardson-Cane validation error: {e}")
        else:
            print("  No Richardson-Cane data available for validation")
        
        if hasattr(self, 'cactus_full') and len(self.cactus_full) > 0:
            print("  Validating with full CACTUS catalogue...")
            try:
                # Focus on halo CMEs in our time range
                swis_start = self.swis_data['datetime'].min()
                swis_end = self.swis_data['datetime'].max()
                
                relevant_cactus = self.cactus_full[
                    (self.cactus_full['datetime'] >= swis_start) & 
                    (self.cactus_full['datetime'] <= swis_end) &
                    (self.cactus_full['is_halo'] == True)
                ]
                
                print(f"    Relevant CACTUS halo CMEs: {len(relevant_cactus)}")
                
                # Check detection performance
                detected_events = self.swis_data[
                    self.swis_data['cme_probability_score'] > self.thresholds['composite_score']
                ]['datetime']
                
                catalogue_matches = 0
                for cactus_date in relevant_cactus['datetime']:
                    time_diff = np.abs((detected_events - cactus_date).dt.total_seconds() / 86400)
                    if len(time_diff) > 0 and np.any(time_diff <= 2):
                        catalogue_matches += 1
                
                print(f"    CACTUS halo CME detection rate: {catalogue_matches}/{len(relevant_cactus)} ({catalogue_matches/len(relevant_cactus)*100:.1f}%)")
            except Exception as e:
                print(f"    CACTUS validation error: {e}")
        else:
            print("  No CACTUS catalogue data available for validation")
        
        # Use our integrated CACTUS events from step 1
        try:
            print("  Using integrated CACTUS events from data preparation...")
            cme_events = self.swis_data[self.swis_data['cme_event'] == 1]
            detected_events = self.swis_data[
                self.swis_data['cme_probability_score'] > self.thresholds['composite_score']
            ]
            
            print(f"    Known CME events: {len(cme_events)}")
            print(f"    Detected events: {len(detected_events)}")
            
            # Calculate overlap
            overlap = len(detected_events[detected_events['cme_event'] == 1])
            if len(cme_events) > 0:
                detection_rate = overlap / len(cme_events) * 100
                print(f"    Detection rate: {overlap}/{len(cme_events)} ({detection_rate:.1f}%)")
            
        except Exception as e:
            print(f"    Error in integrated validation: {e}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\\nCreating visualizations...")
        
        # 1. Feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Features for CME Detection')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=300)
        # plt.show()  # Disabled to prevent hanging
        
        # 2. ROC curves
        plt.figure(figsize=(10, 8))
        for model_name, results in self.model_results.items():
            y_true = self.swis_data['cme_event'][:len(results['Probabilities'])]
            fpr, tpr, _ = roc_curve(y_true, results['Probabilities'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={results['AUC']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for CME Detection Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"), dpi=300)
        # plt.show()  # Disabled to prevent hanging
        
        # 3. Time series of detection scores
        plt.figure(figsize=(16, 10))
        
        # Plot composite score
        plt.subplot(3, 1, 1)
        plt.plot(self.swis_data['datetime'], self.swis_data['cme_probability_score'], 
                color='blue', alpha=0.7, linewidth=0.5)
        plt.axhline(y=self.thresholds['composite_score'], color='red', linestyle='--', 
                   label=f"Threshold ({self.thresholds['composite_score']:.3f})")
        plt.ylabel('CME Probability Score')
        plt.title('Advanced CME Detection System Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot key parameters
        plt.subplot(3, 1, 2)
        plt.plot(self.swis_data['datetime'], self.swis_data['proton_velocity'], 
                color='red', alpha=0.7, linewidth=0.5, label='Velocity')
        plt.ylabel('Velocity (km/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.swis_data['datetime'], self.swis_data['proton_density'], 
                color='green', alpha=0.7, linewidth=0.5, label='Density')
        plt.ylabel('Density (#/cm³)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark detected events
        detected_times = self.swis_data[
            self.swis_data['cme_probability_score'] > self.thresholds['composite_score']
        ]['datetime']
        
        for i in range(3):
            plt.subplot(3, 1, i+1)
            for dt in detected_times[:50]:  # Limit to first 50 for visibility
                plt.axvline(x=dt, color='red', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "detection_results.png"), dpi=300)
        # plt.show()  # Disabled to prevent hanging
        
        print("  ✓ Visualizations created")
    
    def generate_report(self):
        """Generate comprehensive detection report"""
        print("\\nGenerating detection report...")
        
        # Count detections
        total_points = len(self.swis_data)
        detected_points = (self.swis_data['cme_probability_score'] > self.thresholds['composite_score']).sum()
        detection_rate = detected_points / total_points * 100
        
        # Get detection periods
        detected_events = self.swis_data[
            self.swis_data['cme_probability_score'] > self.thresholds['composite_score']
        ]
        
        # Statistics
        if len(detected_events) > 0:
            avg_velocity = detected_events['proton_velocity'].mean()
            avg_density = detected_events['proton_density'].mean()
            avg_temperature = detected_events['proton_temperature'].mean()
        else:
            avg_velocity = avg_density = avg_temperature = 0
        
        # Create report
        report = f"""
ISRO HACKATHON PS10 - ADVANCED CME DETECTION SYSTEM REPORT
=============================================================

Analysis Period: {self.swis_data['datetime'].min()} to {self.swis_data['datetime'].max()}
Total Data Points: {total_points:,}
Analysis Duration: {(self.swis_data['datetime'].max() - self.swis_data['datetime'].min()).days} days

DETECTION SUMMARY:
-----------------
Detected CME Points: {detected_points:,} ({detection_rate:.2f}%)
Detection Threshold: {self.thresholds['composite_score']:.3f}

DETECTED EVENT CHARACTERISTICS:
------------------------------
Average Velocity: {avg_velocity:.1f} km/s
Average Density: {avg_density:.1f} #/cm³  
Average Temperature: {avg_temperature:.1f} km/s

MODEL PERFORMANCE:
-----------------
Random Forest AUC: {self.model_results['Random Forest']['AUC']:.3f}
Logistic Regression AUC: {self.model_results['Logistic Regression']['AUC']:.3f}

TOP 10 MOST IMPORTANT FEATURES:
------------------------------
"""
        
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
            report += f"{i+1:2d}. {row['feature']:30s} {row['importance']:.4f}\\n"
        
        report += f"""
THRESHOLDS:
----------
Optimal (Youden's J): {self.thresholds['optimal']:.3f}
High Sensitivity: {self.thresholds['high_sensitivity']:.3f}
High Specificity: {self.thresholds['high_specificity']:.3f}
Composite Score: {self.thresholds['composite_score']:.3f}

SCIENTIFIC METHODS APPLIED:
--------------------------
1. Pearson Correlation Analysis (linear relationships)
2. Spearman Correlation Analysis (monotonic relationships)  
3. Mutual Information (non-linear dependencies)
4. Physics-based Feature Engineering
5. Temporal Analysis with Granger Causality Proxies
6. Random Forest Feature Importance
7. Composite Scoring Algorithm
8. Cross-validation with External Catalogues

RECOMMENDATIONS:
---------------
1. Use composite score > {self.thresholds['composite_score']:.3f} for operational detection
2. Monitor top features: {', '.join(self.feature_importance.head(3)['feature'].tolist())}
3. Cross-reference with Richardson-Cane catalogue for validation
4. Consider ensemble approach for critical applications

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(os.path.join(self.output_dir, "advanced_cme_detection_report.txt"), 'w') as f:
            f.write(report)
        
        print(report)
        print("\\n✓ Report saved to advanced_cme_detection_report.txt")
    
    def save_detection_results(self):
        """Save final detection results"""
        print("\\nSaving detection results...")
        
        # Create detection dataset
        detection_data = self.swis_data[['datetime', 'proton_density', 'proton_velocity', 
                                        'proton_temperature', 'cme_probability_score', 
                                        'cme_event']].copy()
        
        # Add detection flags
        detection_data['detected_by_composite'] = (
            detection_data['cme_probability_score'] > self.thresholds['composite_score']
        ).astype(int)
        
        # Add model predictions (with error handling)
        try:
            if hasattr(self, 'feature_names'):
                features = self.feature_names
            else:
                features = [col for col in self.swis_data.columns 
                           if col not in ['timestamp', 'datetime', 'cme_event', 'cme_event_extended',
                                         'cme_type', 'cme_speed', 'file_date', 'source_file'] 
                           and self.swis_data[col].dtype in ['int64', 'float64']]
            
            X = self.swis_data[features].fillna(0).replace([np.inf, -np.inf], 0)
            detection_data['rf_probability'] = self.models['random_forest'].predict_proba(X)[:, 1]
            detection_data['detected_by_rf'] = (
                detection_data['rf_probability'] > self.thresholds['optimal']
            ).astype(int)
            
        except Exception as e:
            print(f"  Warning: Could not generate RF predictions: {e}")
            detection_data['rf_probability'] = 0.0
            detection_data['detected_by_rf'] = 0
        
        # Save results
        detection_data.to_csv(
            os.path.join(self.output_dir, "cme_detection_results.csv"), 
            index=False
        )
        
        # Save thresholds
        thresholds_df = pd.DataFrame(list(self.thresholds.items()), 
                                   columns=['threshold_type', 'value'])
        thresholds_df.to_csv(
            os.path.join(self.output_dir, "detection_thresholds.csv"), 
            index=False
        )
        
        print("  ✓ Detection results saved")
        print(f"  ✓ Files: cme_detection_results.csv, detection_thresholds.csv")

def main():
    """Main execution function"""
    # Check for hello command
    if len(sys.argv) > 1 and sys.argv[1].lower() == "hello":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(say_hello(name))
        return
    
    print("Initializing Advanced CME Detection System...")
    
    # Initialize system
    detector = CMEDetectionSystem()
    
    # Load data
    if not detector.load_data():
        return
    
    # Load external catalogues
    detector.load_richardson_cane_data()
    detector.load_cactus_catalogue()
    
    # Feature engineering
    print("\\n" + "="*60)
    print("PHASE 1: SCIENTIFIC FEATURE ENGINEERING")
    print("="*60)
    
    detector.compute_correlation_features()
    detector.compute_physics_features()
    detector.compute_temporal_features()
    detector.compute_granger_causality_proxy()
    detector.build_composite_score()
    
    # Machine learning
    print("\\n" + "="*60)
    print("PHASE 2: MACHINE LEARNING MODELS")
    print("="*60)
    
    detector.train_ml_models()
    detector.determine_optimal_thresholds()
    
    # Validation
    print("\\n" + "="*60)
    print("PHASE 3: VALIDATION & REPORTING")
    print("="*60)
    
    detector.validate_with_catalogues()
    detector.create_visualizations()
    detector.generate_report()
    detector.save_detection_results()
    
    print("\\n" + "="*80)
    print("ADVANCED CME DETECTION SYSTEM - COMPLETE!")
    print("✓ Multi-algorithm approach implemented")
    print("✓ Scientific correlation methods applied")  
    print("✓ Machine learning models trained")
    print("✓ External catalogue validation performed")
    print("✓ Comprehensive reporting generated")
    print("="*80)

if __name__ == "__main__":
    main()
