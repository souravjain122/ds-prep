import math
from statistics import NormalDist
import numpy as np
from scipy import stats


def two_prop_ztest(x1, n1, x2, n2, alpha=0.05):
    """
    Computes z-statistic, p_value and confidence intervals.
    
    Parameters
    ----------
    x1 : int
        Number of successful events in treatment group
    n1 : int
        Total number of events in treatment group
    x2 : int
        Number of successful events in control group
    n2 : int
        Total number of events in control group
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha

    Returns
    -------
    diff : int
        Change in success rate in treatment group as to the control group
    z : float
        z-statistic
    p_value : float
        probability of statistic, at least this extreme, if null hypothesis were true
    ci_low : float
        lowest range of confidence interval
    ci_high: float
        highest range of confidence interval

    Notes
    -----
    During the computation of z-statistic, null hypothesis is assumed to be true so SE is calculated by combining both groups
    For the computation of confidence intervals, we are no longer assuming null hypothesis to be true so SE is calculated unpooled
    """
    # Step 1: diff
    p1 = x1/n1
    p2 = x2/n2
    diff = p1 - p2
    # Step 2: SE pooled
    p = (x1 + x2)/(n1 + n2)
    se_pooled = (p*(1-p)*((1/n1) + (1/n2)))**0.5
    # Step 3: z-statistic
    z_obs = diff/se_pooled
    # Step 4: p-value
    p_value = math.erfc(abs(z_obs)/math.sqrt(2))
    # Step 5: Z-critical
    area = 1 - alpha/2
    z_critical = NormalDist().inv_cdf(area)
    # Step 6 : SE unpooled
    se_unpooled = ((p1*(1-p1)/n1) + (p2*(1-p2)/n2))**0.5
    # Step 7: Confidence Interval
    ci_low = diff - z_critical * se_unpooled
    ci_high = diff + z_critical * se_unpooled
    # Step 8: Final dictionary to return
    final_dict = {
        "diff": diff,
        "z": z_obs,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

    return final_dict

def welch_ttest(a, b, alpha=0.05):
    """
    Computes t-statistic, degree of freedom, p_value and confidence intervals.
    
    Parameters
    ----------
    a : list
        Values in treatment group
    b : list
        Values in control group
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha

    Returns
    -------
    diff : float
        Difference of mean value from treatment group as to the mean value of control group
    t : float
        t-statistic
    df: float
        Degree of freedom
    p_value : float
        probability of statistic, at least this extreme, if null hypothesis were true
    ci_low : float
        lowest range of confidence interval
    ci_high: float
        highest range of confidence interval
    """
    # Step 1: mean, variance and number of events
    a = np.asarray(a)
    b = np.asarray(b)
    x_a = a.mean()
    x_b = b.mean()
    var_a = a.var(ddof = 1)
    var_b = b.var(ddof = 1)
    n_a = len(a)
    n_b = len(b)
    # Step 2: Difference of mean
    diff = x_a - x_b
    # Step 3: Standard Error of difference
    var_a_mean = var_a/n_a
    var_b_mean = var_b/n_b
    se_diff = (var_a_mean + var_b_mean)**0.5
    # Step 4: t-statistic
    t_obs = diff/se_diff
    # Step 5: Degree of freedom
    df = ((var_a_mean + var_b_mean)**2)/((var_a_mean**2)/(n_a - 1) + (var_b_mean**2)/(n_b - 1))
    # Step 6: p-value
    p_value = (1 - stats.t.cdf(t_obs, df)) * 2
    # Step 7: t-critical as per significance level and degree of freedom
    t_critical = stats.t.ppf(1 - alpha/2, df)
    # Step 8: Confidence intervals
    ci_low = diff - t_critical * se_diff
    ci_high = diff + t_critical * se_diff
    # Step 9: Final dictionary to return
    final_dict = {
        "diff": diff,
        "t": t_obs,
        "df": df,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

    return final_dict

def bootstrap_ci(a, b, stat=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """
    Computes confidence intervals by forming new samples using given samples only 
    
    Parameters
    ----------
    a : list
        Values in treatment group
    b : list
        Values in control group
    stat : numpy function
        Statistic to use to get confidence interval
    n_boot : int
        Number of new samples to be formed
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha
    seed : int
        Starting point for numpy’s random number generator

    Returns
    -------
    diff : float
        Difference of stat value from treatment group as to the stat value of control group
    ci_low : float
        lowest range of confidence interval
    ci_high: float
        highest range of confidence interval
    """
    # Step 1: statistic's difference
    a = np.asarray(a)
    b = np.asarray(b)
    stat_a = stat(a)
    stat_b = stat(b)
    diff = stat_a - stat_b
    # Step 2: Generate random samples using given samples
    rng = np.random.default_rng(seed)
    n_a = len(a)
    n_b = len(b)
    new_a = rng.choice(a, size = (n_boot, n_a), replace = True)
    new_b = rng.choice(b, size = (n_boot, n_b), replace = True)
    # Step 3: Generated Sample's statistics
    stat_new_a = stat(new_a, axis = 1)
    stat_new_b = stat(new_b, axis = 1)
    # Step 4: Confidence Intervals
    diff_stats = stat_new_a - stat_new_b
    ci = np.quantile(diff_stats, [alpha, 1-alpha])
    ci_low = ci[0]
    ci_high = ci[1]
    # Step 5: Final dictionary to return
    final_dict = {
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high
    }
    return final_dict

# if __name__ == "__main__":
#     # Reorder-button example from the Day 3 page: treatment first, control second
#     # print(two_prop_ztest(2120, 20000, 2000, 20000))
#     # print(two_prop_ztest(3816, 36000, 400, 4000))
#     # print(two_prop_ztest(424, 4000, 3600, 36000))
#     # print(two_prop_ztest(300, 1000, 10, 100))
#     # print(welch_ttest([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9]))
#     print(bootstrap_ci([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9]))