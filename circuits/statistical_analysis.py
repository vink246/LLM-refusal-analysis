"""
Statistical analysis utilities for circuit comparison and hypothesis testing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import torch


def compute_statistical_significance(similarity_values: List[float], 
                                    null_hypothesis: float = 0.5,
                                    alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Compute statistical significance of circuit similarities
    
    Args:
        similarity_values: List of similarity scores between circuit pairs
        null_hypothesis: Expected value under null hypothesis (default 0.5 = random)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dictionary with p-value, t-statistic, and confidence interval
    """
    if len(similarity_values) < 2:
        return {
            'p_value': 1.0,
            't_statistic': 0.0,
            'mean': np.mean(similarity_values) if similarity_values else 0.0,
            'std': 0.0,
            'confidence_interval_95': (0.0, 0.0),
            'significant': False
        }
    
    values = np.array(similarity_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    
    # One-sample t-test against null hypothesis
    t_stat, p_value = stats.ttest_1samp(values, null_hypothesis, alternative=alternative)
    
    # 95% confidence interval
    n = len(values)
    se = std / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)  # 95% confidence, two-tailed
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    
    # Determine significance (p < 0.05)
    significant = p_value < 0.05
    
    return {
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'mean': float(mean),
        'std': float(std),
        'n': n,
        'confidence_interval_95': (float(ci_lower), float(ci_upper)),
        'significant': significant,
        'null_hypothesis': null_hypothesis
    }


def compare_similarity_distributions(similarities1: List[float], 
                                    similarities2: List[float]) -> Dict[str, float]:
    """
    Compare two distributions of similarity scores using t-test
    
    Args:
        similarities1: First set of similarity scores
        similarities2: Second set of similarity scores
    
    Returns:
        Dictionary with comparison statistics
    """
    if len(similarities1) < 2 or len(similarities2) < 2:
        return {
            'p_value': 1.0,
            't_statistic': 0.0,
            'mean_diff': 0.0,
            'significant': False
        }
    
    arr1 = np.array(similarities1)
    arr2 = np.array(similarities2)
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(arr1, arr2)
    
    mean_diff = np.mean(arr1) - np.mean(arr2)
    significant = p_value < 0.05
    
    return {
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'mean_diff': float(mean_diff),
        'mean1': float(np.mean(arr1)),
        'mean2': float(np.mean(arr2)),
        'std1': float(np.std(arr1, ddof=1)),
        'std2': float(np.std(arr2, ddof=1)),
        'significant': significant
    }


def assess_modularity_with_statistics(similarity_values: List[float],
                                     monolithic_threshold: float = 0.8,
                                     partially_modular_threshold: float = 0.5) -> Dict[str, any]:
    """
    Assess modularity with statistical significance testing
    
    Args:
        similarity_values: List of similarity scores
        monolithic_threshold: Threshold for monolithic assessment
        partially_modular_threshold: Threshold for partially modular assessment
    
    Returns:
        Dictionary with assessment and statistics
    """
    if not similarity_values:
        return {
            'assessment': 'UNKNOWN',
            'average_similarity': 0.0,
            'statistics': None
        }
    
    avg_similarity = np.mean(similarity_values)
    
    # Statistical test against null hypothesis of 0.5 (random)
    stats_result = compute_statistical_significance(similarity_values, null_hypothesis=0.5)
    
    # Determine assessment
    if avg_similarity >= monolithic_threshold:
        assessment = "MONOLITHIC"
    elif avg_similarity >= partially_modular_threshold:
        assessment = "PARTIALLY MODULAR"
    else:
        assessment = "MODULAR"
    
    # Add confidence to assessment based on p-value
    if stats_result['significant']:
        if avg_similarity > 0.5:
            assessment_confidence = "HIGH"  # Statistically significantly different from random
        else:
            assessment_confidence = "HIGH"
    else:
        assessment_confidence = "LOW"  # Not significantly different from random
    
    return {
        'assessment': assessment,
        'assessment_confidence': assessment_confidence,
        'average_similarity': float(avg_similarity),
        'statistics': stats_result
    }

