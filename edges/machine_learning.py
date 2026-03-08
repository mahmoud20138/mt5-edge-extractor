"""
Machine Learning Edge Detection Module.

Identifies edges using ML methods:
- Feature importance analysis
- Classification models
- Regime detection (clustering)
- Anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass
import warnings

from .time_based import EdgeResult

warnings.filterwarnings('ignore')


@dataclass
class MLFeatureImportance:
    """Container for feature importance results."""
    feature_name: str
    importance: float
    rank: int


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for edge detection.
    
    Uses various methods to identify which features are
    most predictive of future returns.
    """
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        pass
    
    def mutual_information_analysis(self, df: pd.DataFrame,
                                   target: str = 'fwd_return_10',
                                   top_n: int = 20) -> List[MLFeatureImportance]:
        """
        Calculate mutual information between features and target.
        
        Args:
            df: DataFrame with features
            target: Target column name
            top_n: Number of top features to return
            
        Returns:
            List of MLFeatureImportance
        """
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target and not c.startswith('fwd_')]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return []
        
        # Calculate mutual information
        mi = mutual_info_regression(X, y, random_state=42)
        
        # Create results
        results = []
        for i, (col, importance) in enumerate(zip(feature_cols, mi)):
            results.append(MLFeatureImportance(
                feature_name=col,
                importance=importance,
                rank=0
            ))
        
        # Sort and rank
        results.sort(key=lambda x: x.importance, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        
        return results[:top_n]
    
    def correlation_analysis(self, df: pd.DataFrame,
                            target: str = 'fwd_return_10',
                            top_n: int = 20) -> List[MLFeatureImportance]:
        """
        Calculate correlation between features and target.
        
        Args:
            df: DataFrame with features
            target: Target column
            top_n: Number of top features
            
        Returns:
            List of MLFeatureImportance
        """
        feature_cols = [c for c in df.columns if c != target and not c.startswith('fwd_')]
        
        correlations = []
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                corr = df[col].corr(df[target])
                if not pd.isna(corr):
                    correlations.append(MLFeatureImportance(
                        feature_name=col,
                        importance=abs(corr),
                        rank=0
                    ))
        
        correlations.sort(key=lambda x: x.importance, reverse=True)
        for i, r in enumerate(correlations):
            r.rank = i + 1
        
        return correlations[:top_n]


class ClassificationEdgeDetector:
    """
    Use classification models to detect edges.
    
    Tests whether ML models can predict market direction
    better than random.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize classification edge detector."""
        self.significance_level = significance_level
    
    def random_forest_direction(self, df: pd.DataFrame,
                               target: str = 'fwd_return_10',
                               test_size: float = 0.3,
                               n_estimators: int = 100) -> EdgeResult:
        """
        Test Random Forest for direction prediction.
        
        Args:
            df: DataFrame with features
            target: Target column
            test_size: Fraction of data for testing
            n_estimators: Number of trees
            
        Returns:
            EdgeResult with model performance
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score
        except ImportError:
            return None
        
        # Prepare features
        feature_cols = [c for c in df.columns if c != target and not c.startswith('fwd_')]
        
        X = df[feature_cols].copy()
        
        # Create target: 1 if positive return, 0 otherwise
        y = (df[target] > 0).astype(int)
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 200:
            return None
        
        # Train/test split (respecting time order for time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=5,
                                       random_state=42,
                                       n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Test if accuracy is significantly better than 50%
        n_correct = (y_pred == y_test).sum()
        n_total = len(y_test)
        
        # Binomial test
        from scipy.stats import binom_test
        try:
            p_val = binom_test(n_correct, n_total, 0.5, alternative='greater')
        except:
            p_val = stats.binom_test(n_correct, n_total, 0.5, alternative='greater')
        
        # Calculate returns using model signals
        test_returns = df[target].iloc[split_idx:].loc[y_test.index]
        strategy_returns = test_returns * (2 * y_pred - 1)  # Convert 0/1 to -1/1
        
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = mean_ret / std_ret * np.sqrt(252 / 10) if std_ret > 0 else 0
        
        return EdgeResult(
            name="Random Forest Direction",
            edge_type="ml_classification",
            sample_size=n_total,
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=(accuracy - 0.5) / np.sqrt(0.25 / n_total),
            p_value=p_val,
            win_rate=accuracy,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level and accuracy > 0.52,
            details={
                'accuracy': accuracy,
                'n_estimators': n_estimators,
                'test_size': test_size
            }
        )
    
    def gradient_boosting_direction(self, df: pd.DataFrame,
                                   target: str = 'fwd_return_10',
                                   test_size: float = 0.3) -> EdgeResult:
        """
        Test Gradient Boosting for direction prediction.
        
        Args:
            df: DataFrame with features
            target: Target column
            test_size: Test fraction
            
        Returns:
            EdgeResult with model performance
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score
        except ImportError:
            return None
        
        # Prepare features
        feature_cols = [c for c in df.columns if c != target and not c.startswith('fwd_')]
        
        X = df[feature_cols].copy()
        y = (df[target] > 0).astype(int)
        
        # Handle non-numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 200:
            return None
        
        # Time-ordered split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Test significance
        n_correct = (y_pred == y_test).sum()
        n_total = len(y_test)
        
        try:
            p_val = stats.binom_test(n_correct, n_total, 0.5, alternative='greater')
        except:
            p_val = 1.0
        
        # Strategy returns
        test_returns = df[target].iloc[split_idx:].loc[y_test.index]
        strategy_returns = test_returns * (2 * y_pred - 1)
        
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = mean_ret / std_ret * np.sqrt(252 / 10) if std_ret > 0 else 0
        
        return EdgeResult(
            name="Gradient Boosting Direction",
            edge_type="ml_classification",
            sample_size=n_total,
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=(accuracy - 0.5) / np.sqrt(0.25 / n_total),
            p_value=p_val,
            win_rate=accuracy,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level and accuracy > 0.52,
            details={'accuracy': accuracy}
        )


class RegimeDetector:
    """
    Detect market regimes using clustering.
    
    Identifies distinct market states that may have
    different optimal strategies.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
        """
        self.n_regimes = n_regimes
    
    def kmeans_regime_detection(self, df: pd.DataFrame,
                               features: List[str] = None) -> pd.Series:
        """
        Detect regimes using K-Means clustering.
        
        Args:
            df: DataFrame with features
            features: Features to use for clustering
            
        Returns:
            Series with regime labels
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return pd.Series(index=df.index, dtype=int)
        
        if features is None:
            # Default features for regime detection
            features = ['vol_std_20', 'adx', 'return_20']
            features = [f for f in features if f in df.columns]
        
        if not features:
            return pd.Series(index=df.index, dtype=int)
        
        X = df[features].dropna()
        
        if len(X) < 100:
            return pd.Series(index=df.index, dtype=int)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        return pd.Series(labels, index=X.index, name='regime')
    
    def regime_returns_analysis(self, df: pd.DataFrame,
                               regime_series: pd.Series,
                               lookahead: int = 10) -> Dict[int, EdgeResult]:
        """
        Analyze returns by regime.
        
        Args:
            df: DataFrame with prices
            regime_series: Regime labels
            lookahead: Forward return period
            
        Returns:
            Dictionary mapping regime to EdgeResult
        """
        fwd_returns = df['close'].pct_change(lookahead).shift(-lookahead)
        
        results = {}
        
        for regime in regime_series.unique():
            regime_mask = regime_series == regime
            regime_returns = fwd_returns[regime_mask].dropna()
            
            if len(regime_returns) < 30:
                continue
            
            t_stat, p_val = stats.ttest_1samp(regime_returns, 0)
            
            mean_ret = regime_returns.mean()
            std_ret = regime_returns.std()
            win_rate = (regime_returns > 0).mean()
            sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
            
            results[regime] = EdgeResult(
                name=f"Regime {regime}",
                edge_type="ml_regime",
                sample_size=len(regime_returns),
                mean_return=mean_ret,
                std_return=std_ret,
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                is_significant=p_val < 0.05,
                details={'regime': regime}
            )
        
        return results


class AnomalyDetector:
    """
    Detect market anomalies for edge identification.
    
    Identifies unusual market conditions that may offer
    trading opportunities.
    """
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
    
    def isolation_forest_anomalies(self, df: pd.DataFrame,
                                   features: List[str] = None) -> pd.Series:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            df: DataFrame with features
            features: Features for anomaly detection
            
        Returns:
            Series with anomaly scores (-1 = anomaly, 1 = normal)
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return pd.Series(index=df.index, dtype=int)
        
        if features is None:
            features = ['return_1', 'vol_std_20', 'range', 'tick_volume']
            features = [f for f in features if f in df.columns]
        
        if not features:
            return pd.Series(index=df.index, dtype=int)
        
        X = df[features].dropna()
        
        if len(X) < 100:
            return pd.Series(index=df.index, dtype=int)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        predictions = iso.fit_predict(X_scaled)
        
        return pd.Series(predictions, index=X.index, name='anomaly')
    
    def anomaly_returns_analysis(self, df: pd.DataFrame,
                                anomaly_series: pd.Series,
                                lookahead: int = 10) -> EdgeResult:
        """
        Analyze returns following anomalies.
        
        Args:
            df: DataFrame with prices
            anomaly_series: Anomaly predictions
            lookahead: Forward return period
            
        Returns:
            EdgeResult with anomaly analysis
        """
        fwd_returns = df['close'].pct_change(lookahead).shift(-lookahead)
        
        # Anomaly returns
        anomaly_mask = anomaly_series == -1
        anomaly_returns = fwd_returns[anomaly_mask].dropna()
        
        # Normal returns
        normal_mask = anomaly_series == 1
        normal_returns = fwd_returns[normal_mask].dropna()
        
        if len(anomaly_returns) < 30:
            return None
        
        # Compare anomaly vs normal returns
        t_stat, p_val = stats.ttest_ind(anomaly_returns.abs(), normal_returns.abs())
        
        mean_ret = anomaly_returns.mean()
        std_ret = anomaly_returns.std()
        win_rate = (anomaly_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name="Anomaly Detection",
            edge_type="ml_anomaly",
            sample_size=len(anomaly_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < 0.05,
            details={
                'anomaly_count': len(anomaly_returns),
                'normal_count': len(normal_returns),
                'avg_anomaly_return': anomaly_returns.mean(),
                'avg_normal_return': normal_returns.mean()
            }
        )


class MLEdgeDetector:
    """
    Main class for ML-based edge detection.
    
    Combines feature importance, classification, regime detection,
    and anomaly detection.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize ML edge detector."""
        self.significance_level = significance_level
        self.feature_importance = FeatureImportanceAnalyzer()
        self.classification = ClassificationEdgeDetector(significance_level)
        self.regime = RegimeDetector()
        self.anomaly = AnomalyDetector()
    
    def run_all_ml_edges(self, df: pd.DataFrame,
                        target: str = 'fwd_return_10') -> Dict[str, EdgeResult]:
        """
        Run all ML-based edge tests.
        
        Args:
            df: DataFrame with features
            target: Target column
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # Classification models
        rf_result = self.classification.random_forest_direction(df, target)
        if rf_result:
            all_results['random_forest'] = rf_result
        
        gb_result = self.classification.gradient_boosting_direction(df, target)
        if gb_result:
            all_results['gradient_boosting'] = gb_result
        
        # Regime detection
        regime_series = self.regime.kmeans_regime_detection(df)
        if not regime_series.empty:
            regime_results = self.regime.regime_returns_analysis(df, regime_series)
            for regime, result in regime_results.items():
                all_results[f'regime_{regime}'] = result
        
        # Anomaly detection
        anomaly_series = self.anomaly.isolation_forest_anomalies(df)
        if not anomaly_series.empty:
            anomaly_result = self.anomaly.anomaly_returns_analysis(df, anomaly_series)
            if anomaly_result:
                all_results['anomaly'] = anomaly_result
        
        return all_results
    
    def get_top_features(self, df: pd.DataFrame,
                        target: str = 'fwd_return_10',
                        top_n: int = 20) -> List[MLFeatureImportance]:
        """
        Get top predictive features.
        
        Args:
            df: DataFrame with features
            target: Target column
            top_n: Number of features
            
        Returns:
            List of top features
        """
        # Combine mutual information and correlation
        mi_features = self.feature_importance.mutual_information_analysis(df, target, top_n)
        corr_features = self.feature_importance.correlation_analysis(df, target, top_n)
        
        # Combine and re-rank
        feature_scores = {}
        for f in mi_features:
            feature_scores[f.feature_name] = f.importance
        for f in corr_features:
            if f.feature_name in feature_scores:
                feature_scores[f.feature_name] += f.importance
            else:
                feature_scores[f.feature_name] = f.importance
        
        # Sort
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (name, score) in enumerate(sorted_features[:top_n]):
            results.append(MLFeatureImportance(
                feature_name=name,
                importance=score,
                rank=i + 1
            ))
        
        return results
