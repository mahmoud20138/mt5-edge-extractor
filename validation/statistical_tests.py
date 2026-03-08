"""
Statistical Testing Module.

Provides statistical hypothesis tests for edge validation:
- T-tests
- Non-parametric tests
- Multiple testing corrections
- Effect size calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from dataclasses import dataclass


@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    details: Optional[Dict] = None


class StatisticalTests:
    """
    Perform statistical tests for edge validation.
    
    Provides methods for:
    - Hypothesis testing
    - Multiple testing corrections
    - Effect size calculations
    - Confidence intervals
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    # ============ PARAMETRIC TESTS ============
    
    def t_test_one_sample(self, data: pd.Series, 
                          mu: float = 0) -> TestResult:
        """
        One-sample t-test.
        
        Tests whether the mean of the sample is significantly
        different from a hypothesized value.
        
        Args:
            data: Sample data
            mu: Hypothesized mean
            
        Returns:
            TestResult with test statistics
        """
        data = data.dropna()
        
        if len(data) < 3:
            return TestResult(
                test_name="One-sample t-test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        t_stat, p_val = stats.ttest_1samp(data, mu)
        
        # Effect size (Cohen's d)
        cohens_d = (data.mean() - mu) / data.std()
        
        # Confidence interval
        se = data.std() / np.sqrt(len(data))
        ci = stats.t.interval(
            1 - self.significance_level,
            len(data) - 1,
            loc=data.mean(),
            scale=se
        )
        
        return TestResult(
            test_name="One-sample t-test",
            statistic=t_stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=cohens_d,
            confidence_interval=ci,
            details={
                'sample_mean': data.mean(),
                'sample_std': data.std(),
                'sample_size': len(data),
                'hypothesized_mean': mu
            }
        )
    
    def t_test_two_sample(self, group1: pd.Series, 
                         group2: pd.Series,
                         equal_var: bool = False) -> TestResult:
        """
        Two-sample t-test.
        
        Tests whether two groups have significantly different means.
        
        Args:
            group1: First sample
            group2: Second sample
            equal_var: Assume equal variances
            
        Returns:
            TestResult with test statistics
        """
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        if len(group1) < 3 or len(group2) < 3:
            return TestResult(
                test_name="Two-sample t-test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) /
            (len(group1) + len(group2) - 2)
        )
        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
        
        return TestResult(
            test_name="Two-sample t-test",
            statistic=t_stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=cohens_d,
            details={
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'group1_size': len(group1),
                'group2_size': len(group2)
            }
        )
    
    def paired_t_test(self, before: pd.Series, 
                     after: pd.Series) -> TestResult:
        """
        Paired t-test.
        
        Tests whether paired observations have significantly
        different means.
        
        Args:
            before: Before measurements
            after: After measurements
            
        Returns:
            TestResult with test statistics
        """
        # Align indices
        common_idx = before.dropna().index.intersection(after.dropna().index)
        before = before.loc[common_idx]
        after = after.loc[common_idx]
        
        if len(before) < 3:
            return TestResult(
                test_name="Paired t-test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        t_stat, p_val = stats.ttest_rel(before, after)
        
        # Effect size
        diff = after - before
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        return TestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=cohens_d,
            details={
                'mean_difference': diff.mean(),
                'sample_size': len(before)
            }
        )
    
    # ============ NON-PARAMETRIC TESTS ============
    
    def mann_whitney_u(self, group1: pd.Series,
                       group2: pd.Series) -> TestResult:
        """
        Mann-Whitney U test (Wilcoxon rank-sum).
        
        Non-parametric alternative to two-sample t-test.
        
        Args:
            group1: First sample
            group2: Second sample
            
        Returns:
            TestResult with test statistics
        """
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        if len(group1) < 3 or len(group2) < 3:
            return TestResult(
                test_name="Mann-Whitney U",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        return TestResult(
            test_name="Mann-Whitney U",
            statistic=u_stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=effect_size,
            details={
                'group1_median': group1.median(),
                'group2_median': group2.median()
            }
        )
    
    def wilcoxon_signed_rank(self, x: pd.Series,
                             y: pd.Series = None) -> TestResult:
        """
        Wilcoxon signed-rank test.
        
        Non-parametric alternative to paired t-test.
        
        Args:
            x: First sample or differences
            y: Second sample (optional)
            
        Returns:
            TestResult with test statistics
        """
        if y is not None:
            common_idx = x.dropna().index.intersection(y.dropna().index)
            x = x.loc[common_idx] - y.loc[common_idx]
        
        x = x.dropna()
        
        if len(x) < 3:
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        stat, p_val = stats.wilcoxon(x)
        
        # Effect size (matched-pairs rank-biserial)
        effect_size = stat / (len(x) * (len(x) + 1) / 4) if len(x) > 0 else 0
        
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=effect_size,
            details={'median': x.median()}
        )
    
    def kruskal_wallis(self, *groups: pd.Series) -> TestResult:
        """
        Kruskal-Wallis H-test.
        
        Non-parametric alternative to one-way ANOVA.
        
        Args:
            *groups: Multiple samples
            
        Returns:
            TestResult with test statistics
        """
        groups = [g.dropna() for g in groups]
        
        if any(len(g) < 3 for g in groups):
            return TestResult(
                test_name="Kruskal-Wallis",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        h_stat, p_val = stats.kruskal(*groups)
        
        # Effect size (epsilon squared)
        n = sum(len(g) for g in groups)
        k = len(groups)
        effect_size = (h_stat - k + 1) / (n - k) if n > k else 0
        
        return TestResult(
            test_name="Kruskal-Wallis",
            statistic=h_stat,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=effect_size,
            details={'n_groups': len(groups)}
        )
    
    def chi_squared_goodness_of_fit(self, observed: np.ndarray,
                                    expected: np.ndarray = None) -> TestResult:
        """
        Chi-squared goodness of fit test.
        
        Tests whether observed frequencies match expected.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (default: uniform)
            
        Returns:
            TestResult with test statistics
        """
        if expected is None:
            expected = np.ones(len(observed)) * observed.sum() / len(observed)
        
        chi2, p_val = stats.chisquare(observed, expected)
        
        # Effect size (Cramer's V)
        n = observed.sum()
        effect_size = np.sqrt(chi2 / n) if n > 0 else 0
        
        return TestResult(
            test_name="Chi-squared goodness of fit",
            statistic=chi2,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=effect_size,
            details={
                'observed': observed.tolist(),
                'expected': expected.tolist()
            }
        )
    
    def chi_squared_independence(self, contingency_table: np.ndarray) -> TestResult:
        """
        Chi-squared test of independence.
        
        Tests whether two categorical variables are independent.
        
        Args:
            contingency_table: Contingency table
            
        Returns:
            TestResult with test statistics
        """
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Effect size (Cramer's V)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        effect_size = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
        
        return TestResult(
            test_name="Chi-squared independence",
            statistic=chi2,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=effect_size,
            details={'degrees_of_freedom': dof}
        )
    
    # ============ CORRELATION TESTS ============
    
    def pearson_correlation(self, x: pd.Series, 
                           y: pd.Series) -> TestResult:
        """
        Pearson correlation test.
        
        Tests whether correlation is significantly different from zero.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            TestResult with test statistics
        """
        common_idx = x.dropna().index.intersection(y.dropna().index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 3:
            return TestResult(
                test_name="Pearson correlation",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        r, p_val = stats.pearsonr(x, y)
        
        return TestResult(
            test_name="Pearson correlation",
            statistic=r,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=r,
            details={
                'correlation': r,
                'sample_size': len(x)
            }
        )
    
    def spearman_correlation(self, x: pd.Series,
                            y: pd.Series) -> TestResult:
        """
        Spearman rank correlation test.
        
        Non-parametric correlation test.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            TestResult with test statistics
        """
        common_idx = x.dropna().index.intersection(y.dropna().index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 3:
            return TestResult(
                test_name="Spearman correlation",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                details={'error': 'Insufficient data'}
            )
        
        rho, p_val = stats.spearmanr(x, y)
        
        return TestResult(
            test_name="Spearman correlation",
            statistic=rho,
            p_value=p_val,
            is_significant=p_val < self.significance_level,
            effect_size=rho,
            details={
                'correlation': rho,
                'sample_size': len(x)
            }
        )
    
    # ============ MULTIPLE TESTING CORRECTIONS ============
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """
        Bonferroni correction for multiple testing.
        
        Adjusts p-values by multiplying by the number of tests.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of adjusted p-values
        """
        n_tests = len(p_values)
        return [min(p * n_tests, 1.0) for p in p_values]
    
    def holm_bonferroni(self, p_values: List[float]) -> List[float]:
        """
        Holm-Bonferroni correction.
        
        Sequentially rejective method, more powerful than Bonferroni.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of adjusted p-values
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        indices, sorted_p = zip(*indexed)
        
        adjusted = [0] * n
        adjusted[indices[n-1]] = sorted_p[n-1]
        
        for i in range(n-2, -1, -1):
            adjusted[indices[i]] = min(
                max(adjusted[indices[i+1]], sorted_p[i] * (n - i)),
                1.0
            )
        
        return adjusted
    
    def benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """
        Benjamini-Hochberg FDR correction.
        
        Controls the false discovery rate.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of adjusted p-values
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        indices, sorted_p = zip(*indexed)
        
        adjusted = [0] * n
        
        # Last p-value
        adjusted[indices[n-1]] = sorted_p[n-1]
        
        # Work backwards
        for i in range(n-2, -1, -1):
            adjusted[indices[i]] = min(
                adjusted[indices[i+1]],
                sorted_p[i] * n / (i + 1)
            )
        
        return adjusted
    
    # ============ EFFECT SIZE INTERPRETATION ============
    
    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def interpret_correlation(r: float) -> str:
        """
        Interpret correlation coefficient.
        
        Args:
            r: Correlation value
            
        Returns:
            Interpretation string
        """
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        elif r < 0.7:
            return "large"
        else:
            return "very large"
