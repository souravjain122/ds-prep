import math
from statistics import NormalDist

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
    z_obs : float
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
    # Step 8: Final dict
    final_dict = {
        "diff": diff,
        "z": z_obs,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

    return final_dict


if __name__ == "__main__":
    # Reorder-button example from the Day 3 page: treatment first, control second
    print(two_prop_ztest(2120, 20000, 2000, 20000))
    print(two_prop_ztest(3816, 36000, 400, 4000))
    print(two_prop_ztest(424, 4000, 3600, 36000))
    print(two_prop_ztest(300, 1000, 10, 100))