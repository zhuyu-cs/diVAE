import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import logging
import os
import warnings
import umap

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveClassifierEvaluator:
    """Comprehensive classifier evaluator"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.ct_to_hour_mapping = self._create_ct_mapping()
        self.classifiers = self._get_comprehensive_classifiers()
        
    def _create_ct_mapping(self):
        """Create mapping from CT time to 24-hour labels"""
        mapping = {}
        # Hour 0: 1-5
        for i in range(1, 6):
            mapping[i] = 0
        # Hours 1-23: interval of 60, 5 consecutive numbers per group
        for hour in range(1, 24):
            base = 60 * hour + 1
            for i in range(5):
                mapping[base + i] = hour
        return mapping
    
    def _ct_to_hour_label(self, ct_time):
        """Convert CT time to 24-hour label (0-23)"""
        return self.ct_to_hour_mapping.get(ct_time, -1)
    
    def _get_comprehensive_classifiers(self):
        """Get comprehensive set of classifiers"""
        classifiers = {
            # Support Vector Machines
            'SVM_Linear': SVC(kernel='linear', C=1.0, random_state=42),
            'SVM_RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'SVM_Poly': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
            'SVM_Sigmoid': SVC(kernel='sigmoid', C=1.0, random_state=42),
            
            # Ensemble methods
            'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Extra_Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
            
            # Linear classifiers
            'Logistic_Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'Ridge_Classifier': RidgeClassifier(alpha=1.0, random_state=42),
            'SGD_Classifier': SGDClassifier(loss='hinge', alpha=0.01, random_state=42, max_iter=1000),
            
            # Nearest neighbors
            'KNN_3': KNeighborsClassifier(n_neighbors=3),
            'KNN_5': KNeighborsClassifier(n_neighbors=5),
            'KNN_7': KNeighborsClassifier(n_neighbors=7),
            'KNN_Weighted': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            
            # Bayesian classifiers
            'Gaussian_NB': GaussianNB(),
            'Multinomial_NB': MultinomialNB(),
            
            # Decision trees
            'Decision_Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Decision_Tree_Gini': DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42),
            'Decision_Tree_Entropy': DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42),
            
            # Neural networks
            'MLP_Small': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
            'MLP_Medium': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'MLP_Large': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=42),
            
            # Discriminant analysis
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
        }
        
        return classifiers
    
    def load_latent_variables(self, pkl_path, feature_type):
        """
        Load latent variable data from a pkl file.
        
        Args:
            pkl_path: path to the pkl file
            feature_type: feature type, e.g. 'pld', 'ild'
        
        Returns:
            data_dict: processed data dictionary
        """
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        logger.info(f"Loaded {pkl_path} with feature type: {feature_type}")
        return data_dict
    
    def extract_features_and_labels(self, data_dict, feature_type):
        """
        Extract features and labels from the data dictionary.
        
        Args:
            data_dict: data dictionary
            feature_type: feature type
        
        Returns:
            X_train, X_test, y_train, y_test: training and testing data
        """
        train_mice = ['SCN1', 'SCN2', 
                      'SCN3', 'SCN4']
        test_mice = ['SCN5', 'SCN6']
        
        def extract_from_mice(mice_list, split_key):
            features = []
            labels = []
            
            for mouse in mice_list:
                if mouse not in data_dict:
                    continue
                
                mouse_data = data_dict[mouse]
                if split_key not in mouse_data:
                    continue
                
                sessions = mouse_data[split_key]
                for session_name, session_data in sessions.items():
                    ct_time = int(session_name.split('.')[0])
                    hour_label = self._ct_to_hour_label(ct_time)
                    
                    if hour_label == -1:
                        continue
                    
                    if feature_type not in session_data:
                        continue
                    
                    feature_data = session_data[feature_type]
                    
                    # Handle feature data of different shapes
                    if len(feature_data.shape) > 1:
                        feature_data = np.mean(feature_data, axis=0)
                    
                    features.append(feature_data)
                    labels.append(hour_label)
            
            return np.array(features), np.array(labels)
        
        # Extract training and testing data
        X_train, y_train = extract_from_mice(train_mice, 'train')
        X_test, y_test = extract_from_mice(test_mice, 'val')
        
        logger.info(f"Extracted features - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def create_baseline_features(self, data_dict, target_dim):
        """
        Create baseline features from raw calcium signals.
        
        Args:
            data_dict: data dictionary containing cal_data
            target_dim: target dimensionality
        
        Returns:
            baseline_results: dictionary containing PCA, ICA, UMAP results
        """
        baseline_results = {}
        
        # Baseline method configuration
        baseline_methods = {
            'PCA': PCA(n_components=target_dim, random_state=42),
            'ICA': FastICA(n_components=target_dim, random_state=42, max_iter=1000),
            'UMAP': umap.UMAP(n_components=target_dim, random_state=42, n_neighbors=15, min_dist=0.1)
        }
        
        # Process each mouse separately
        train_mice = ['SCN1', 'SCN2', 
                      'SCN3', 'SCN4']
        test_mice = ['SCN5', 'SCN6']
        
        for method_name, method in baseline_methods.items():
            logger.info(f"Creating {method_name} baseline features...")
            
            all_train_features = []
            all_test_features = []
            train_labels = []
            test_labels = []
            
            # Process each mouse
            all_mice = train_mice + test_mice
            for mouse in all_mice:
                if mouse not in data_dict:
                    continue
                
                # Collect all calcium signals for this mouse
                split_key = 'train' if mouse in train_mice else 'val'
                mouse_data = data_dict[mouse][split_key]
                
                calcium_signals = []
                mouse_labels = []
                
                for session_name, session_data in mouse_data.items():
                    if 'cal_data' not in session_data:
                        continue
                        
                    ct_time = int(session_name.split('.')[0])
                    hour_label = self._ct_to_hour_label(ct_time)
                    
                    if hour_label == -1:
                        continue
                    
                    cal_signal = session_data['cal_data']  # (neurons, time_points)
                    flattened_signal = cal_signal.flatten()
                    
                    calcium_signals.append(flattened_signal)
                    mouse_labels.append(hour_label)
                
                if not calcium_signals:
                    continue
                
                # Apply dimensionality reduction to this mouse's data
                X_mouse = np.array(calcium_signals)
                
                # Ensure sufficient dimensions
                if X_mouse.shape[1] > target_dim and X_mouse.shape[0] > 1:
                    try:
                        if method_name == 'UMAP' and X_mouse.shape[0] < 15:
                            # UMAP needs adjusted n_neighbors
                            method_copy = umap.UMAP(n_components=target_dim, random_state=42, 
                                                  n_neighbors=min(5, X_mouse.shape[0]-1), min_dist=0.1)
                            X_reduced = method_copy.fit_transform(X_mouse)
                        else:
                            X_reduced = method.fit_transform(X_mouse)
                    except Exception as e:
                        logger.warning(f"Error with {method_name} for {mouse}: {e}, using PCA fallback")
                        fallback_pca = PCA(n_components=min(target_dim, X_mouse.shape[1]-1), random_state=42)
                        X_reduced = fallback_pca.fit_transform(X_mouse)
                else:
                    # Insufficient dimensions, truncate or pad
                    if X_mouse.shape[1] >= target_dim:
                        X_reduced = X_mouse[:, :target_dim]
                    else:
                        padding = np.zeros((X_mouse.shape[0], target_dim - X_mouse.shape[1]))
                        X_reduced = np.hstack([X_mouse, padding])
                
                # Assign to training or testing set
                if mouse in train_mice:
                    all_train_features.extend(X_reduced)
                    train_labels.extend(mouse_labels)
                else:
                    all_test_features.extend(X_reduced)
                    test_labels.extend(mouse_labels)
            
            baseline_results[method_name] = {
                'X_train': np.array(all_train_features),
                'X_test': np.array(all_test_features),
                'y_train': np.array(train_labels),
                'y_test': np.array(test_labels)
            }
            
            logger.info(f"{method_name} features - Train: {baseline_results[method_name]['X_train'].shape}, "
                       f"Test: {baseline_results[method_name]['X_test'].shape}")
        
        return baseline_results
    
    def evaluate_method_with_all_classifiers(self, X_train, X_test, y_train, y_test, method_name):
        """
        Evaluate performance of a single method using all classifiers.
        
        Args:
            X_train, X_test, y_train, y_test: training and testing data
            method_name: name of the method
        
        Returns:
            results: dictionary containing results for each classifier
        """
        results = {}
        
        # Standardize features
        scaler = StandardScaler()
        
        # For MultinomialNB, features must be non-negative
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create non-negative features for MultinomialNB
        X_train_nonneg = X_train_scaled - X_train_scaled.min() + 1e-8
        X_test_nonneg = X_test_scaled - X_test_scaled.min() + 1e-8
        
        for clf_name, clf in self.classifiers.items():
            try:
                logger.info(f"  Training {clf_name}...")
                
                # Use non-negative features for MultinomialNB
                if clf_name == 'Multinomial_NB':
                    X_train_use = X_train_nonneg
                    X_test_use = X_test_nonneg
                else:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                
                # Train model
                clf.fit(X_train_use, y_train)
                
                # Test set prediction
                y_pred = clf.predict(X_test_use)
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                
                # 5-fold cross-validation
                cv_scores = cross_val_score(clf, X_train_use, y_train, cv=5, scoring='accuracy')
                
                results[clf_name] = {
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"    {clf_name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test={test_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"    Error with {clf_name}: {e}")
                results[clf_name] = {
                    'test_accuracy': 0.0,
                    'test_f1': 0.0,
                    'cv_mean': 0.0,
                    'cv_std': 0.0
                }
        
        return results

def run_comprehensive_comparison(config):
    """
    Run comprehensive comparison experiments.
    
    Args:
        config: configuration dictionary containing pkl file paths and method names
    
    Returns:
        comparison_results: comparison results
    """
    evaluator = ComprehensiveClassifierEvaluator()
    all_results = {}
    
    # Determine target dimension (from first latent variable method)
    first_method = next(iter(config['latent_methods'].values()))
    sample_data = evaluator.load_latent_variables(first_method['pkl_path'], first_method['feature_type'])
    sample_mouse = next(iter(sample_data.keys()))
    sample_session = next(iter(sample_data[sample_mouse]['train'].keys()))
    sample_features = sample_data[sample_mouse]['train'][sample_session][first_method['feature_type']]
    
    if len(sample_features.shape) > 1:
        target_dim = sample_features.shape[1]
    else:
        target_dim = len(sample_features)
    
    logger.info(f"Target dimension: {target_dim}")
    
    # 1. Evaluate latent variable methods
    for method_name, method_config in config['latent_methods'].items():
        logger.info(f"Evaluating latent method: {method_name}")
        
        data_dict = evaluator.load_latent_variables(method_config['pkl_path'], method_config['feature_type'])
        X_train, X_test, y_train, y_test = evaluator.extract_features_and_labels(data_dict, method_config['feature_type'])
        
        if X_train.size == 0 or X_test.size == 0:
            logger.warning(f"No data found for {method_name}")
            continue
        
        results = evaluator.evaluate_method_with_all_classifiers(X_train, X_test, y_train, y_test, method_name)
        all_results[method_name] = results
    
    # 2. Evaluate baseline methods (based on cal_data from first pkl file)
    if config.get('include_baselines', True):
        logger.info("Creating and evaluating baseline methods...")
        baseline_data_dict = evaluator.load_latent_variables(
            first_method['pkl_path'], first_method['feature_type']
        )
        
        baseline_results = evaluator.create_baseline_features(baseline_data_dict, target_dim)
        
        for method_name, method_data in baseline_results.items():
            logger.info(f"Evaluating baseline method: {method_name}")
            results = evaluator.evaluate_method_with_all_classifiers(
                method_data['X_train'], method_data['X_test'],
                method_data['y_train'], method_data['y_test'],
                method_name
            )
            all_results[method_name] = results
    
    return all_results


def load_and_prepare_data(pkl_path):
    """Load and prepare data, computing SEM"""
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Create summary data
    summary_data = []
    
    for method_name, method_results in all_results.items():
        # Compute average performance of this method across all classifiers
        cv_means = [result['cv_mean'] for result in method_results.values()]
        cv_stds = [result['cv_std'] for result in method_results.values()]
        test_accs = [result['test_accuracy'] for result in method_results.values()]
        
        avg_cv_mean = np.mean(cv_means)
        avg_cv_std = np.std(cv_means)  # Use cross-classifier standard deviation
        avg_test_acc = np.mean(test_accs)
        
        # Compute SEM (Standard Error of Mean)
        # SEM = std / sqrt(n), where n is the number of classifiers
        n_classifiers = len(cv_means)
        sem = avg_cv_std / np.sqrt(n_classifiers)
        
        # Find the best classifier
        best_clf = max(method_results.keys(), key=lambda k: method_results[k]['cv_mean'])
        best_performance = method_results[best_clf]
        best_cv_std = best_performance['cv_std']
        
        summary_data.append({
            'Method': method_name,
            'Avg_CV_Performance': f"{avg_cv_mean:.3f}±{avg_cv_std:.3f}",
            'Avg_Test_Accuracy': f"{avg_test_acc:.3f}",
            'Best_Classifier': best_clf,
            'Best_CV_Performance': f"{best_performance['cv_mean']:.3f}±{best_cv_std:.3f}",
            'Best_Test_Accuracy': f"{best_performance['test_accuracy']:.3f}",
            'Avg_CV_Mean_Numeric': avg_cv_mean,
            'Avg_CV_Std_Numeric': avg_cv_std,
            'SEM': sem,  # Add SEM
            'Best_CV_Mean_Numeric': best_performance['cv_mean'],
            'Best_CV_Std_Numeric': best_cv_std
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Avg_CV_Mean_Numeric', ascending=False)
    
    return summary_df

def prepare_plot_data(summary_df):
    """Prepare plotting data"""
    # Reorder, placing diVAE first
    df_reordered = summary_df.copy()
    if 'diVAE' in df_reordered['Method'].values:
        divae_row = df_reordered[df_reordered['Method'] == 'diVAE']
        other_rows = df_reordered[df_reordered['Method'] != 'diVAE']
        df_reordered = pd.concat([divae_row, other_rows]).reset_index(drop=True)
    
    methods = df_reordered['Method'].values
    scores = df_reordered['Avg_CV_Mean_Numeric'].values
    stds = df_reordered['Avg_CV_Std_Numeric'].values
    sems = df_reordered['SEM'].values  # Add SEM
    
    # Ensure values are in range [0, 1]
    scores = np.clip(scores, 0, 1)
    
    return methods, scores, stds, sems

def get_colors(methods):
    """Define color scheme - updated to support new piVAE variants"""
    colors = []
    for method in methods:
        if method == 'diVAE':
            colors.append('#e74c3c')  # Red - diVAE (Best)
        elif any(vae_method in method for vae_method in ['VAE', 'piVAE']):
            # Includes all VAE variants: VAE, piVAE(position), piVAE(time), etc.
            colors.append('#f39c12')  # Orange - VAE methods
        else:
            colors.append('#3498db')  # Blue - Traditional methods
    return colors

def add_chance_level(ax, num_classes=24):
    """Add chance level reference line"""
    chance_level = 1.0 / num_classes
    
    # Add light gray reference line
    ax.axhline(y=chance_level, color='lightgray', linestyle='--', 
               alpha=0.7, linewidth=2, zorder=1)
    
    return chance_level

def create_clean_with_sem_visualization(summary_df, save_dir='./visualization_results'):
    """Create bar chart visualization with SEM error bars"""
    
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    methods, scores, stds, sems = prepare_plot_data(summary_df)
    colors = get_colors(methods)
    
    # Convert to percentage
    scores_percent = scores * 100
    sems_percent = sems * 100
    chance_level_percent = (1.0 / 24) * 100  # ~4.17%
    
    # Add chance level (percentage version)
    ax.axhline(y=chance_level_percent, color='lightgray', linestyle='--', 
               alpha=0.7, linewidth=2, zorder=1)
    
    # Create bar chart with SEM error bars
    bars = ax.bar(range(len(methods)), scores_percent, 
                 width=0.6, 
                  yerr=sems_percent,
                  color=colors, alpha=0.9, edgecolor='white', linewidth=1, 
                  capsize=5, error_kw={'linewidth': 2, 'alpha': 0.8}, zorder=3)
    
    # Set labels
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=13, rotation=0)
    ax.set_ylabel('CT Classification Accuracy (%)', fontsize=14)
    ax.set_title('Average Circadian Time Classification Performance Across 25 Classifiers', 
                 fontsize=16, pad=20)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=13)
    
    # Add value labels (show average only)
    for i, (bar, score_pct) in enumerate(zip(bars, scores_percent)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sems_percent[i] + 1,
                f'{score_pct:.1f}%', ha='center', va='bottom', 
                fontsize=12)
    
    # Update legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#e74c3c', alpha=0.9, label='diVAE (Best)'),
        plt.Rectangle((0,0),1,1, facecolor='#f39c12', alpha=0.9, label='VAE methods'),
        plt.Rectangle((0,0),1,1, facecolor='#3498db', alpha=0.9, label='Traditional methods'),
        plt.Line2D([0], [0], color='lightgray', linestyle='--', alpha=0.7, label='Chance level (4.2%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Clean borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clean_with_sem_bars_visualization.png', dpi=300, bbox_inches='tight')

    

def create_all_visualizations(pkl_path):
    """Create all visualization schemes"""
    
    print("🚀 Starting visualization...")
    print(f"📁 Loading data from {pkl_path}...")
    
    summary_df = load_and_prepare_data(pkl_path)
    
    chance_level = 1.0 / 24
    
    print(f"\n📊 Data summary:")
    print(f"Total methods: {len(summary_df)}")
    print(f"Method list: {', '.join(summary_df['Method'].values)}")
    print(f"Chance level (1/24): {chance_level:.3f} ({chance_level*100:.1f}%)")
    print("\n" + "="*60)
    
    create_clean_with_sem_visualization(summary_df)
    print()
    
    print("🎯 Performance summary (with SEM):")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        score = row['Avg_CV_Mean_Numeric']
        sem = row['SEM']
        fold_improvement = score / chance_level
        print(f"{row['Method']}: {score:.3f}±{sem:.3f} ({fold_improvement:.0f}x chance level)")
    
    print(f"\n🎉 Visualization with SEM saved to ./visualization_results/ directory")
    
    return summary_df

def main():
    """Main function"""
    
    config = {
        'latent_methods': {
            'VAE': {
                'pkl_path': './latents/vae.pkl',
                'feature_type': 'pld'
            },
            'piVAE (spatial)': {
                'pkl_path': './latents/pivae_s.pkl',
                'feature_type': 'pld'
            },
            'piVAE (temporal)': {
                'pkl_path': './latents/pivae_t.pkl',
                'feature_type': 'pld'
            },
            'diVAE': {
                'pkl_path': './latents/divae.pkl',
                'feature_type': 'latent_variable'
            }
        },
        'include_baselines': True
    }
    
    logger.info("Starting comprehensive classification comparison...")
    logger.info(f"Using {len(ComprehensiveClassifierEvaluator()._get_comprehensive_classifiers())} classifiers")
    
    all_results = run_comprehensive_comparison(config)
    
    logger.info("Creating publication-style visualizations...")
    
    with open('./visualization_results/complete_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    summary_df = create_all_visualizations('./visualization_results/complete_results.pkl')
    
    
if __name__ == '__main__':
    main()