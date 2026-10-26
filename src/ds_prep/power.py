import math
from statistics import NormalDist

def sample_size_prop(p1, mde_abs, alpha=0.05, power=0.8, ratio=1.0, pooled=True):
    """
    Computes sample size for proportional test. 
    
    Parameters
    ----------
    p1 : float
        Conversion rate for control group
    mde_abs : float
        Minimum detectable effect in absolute points
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha
    power : float
        Power, probability of detecting an effect given it exists
    ratio : float
        Ratio of number of users in treatment group by number of users in control group
    pooled : True  -> SE0 (cut-off) uses the pooled rate p̄ under H0,
                      SE1 (power) uses p1 and p2 separately.
             False -> one unpooled SE used for both steps (simpler approximation).

    Returns
    -------
    n_control : int
        Sample size of control group
    n_treatment : int
        Sample size of treatment group

    """
    # Step 1: z-statistics
    z_a = NormalDist().inv_cdf(1 - alpha/2)
    z_b = NormalDist().inv_cdf(power)
    # Step 2: Conversion rate for treatment group
    p2 = p1 + mde_abs
    # Step 3: Sample size for pooled SE under H0
    if pooled:
        p = (p1 + ratio*p2)/(ratio + 1)
        n_control = (z_a*((p*(1-p)*(1+ratio)/ratio)**0.5) + z_b*((p1*(1-p1) + p2*(1-p2)/ratio)**0.5))**2/(mde_abs**2)
    else:
        n_control = ((z_a + z_b)**2) * (ratio*p1*(1-p1) + p2*(1-p2))/(ratio * (mde_abs**2))
        
    return math.ceil(n_control), math.ceil(ratio * n_control)


def sample_size_mean(sd, mde, alpha=0.05, power=0.8, ratio=1.0):
    """
    Computes sample size for mean test. 
    
    Parameters
    ----------
    sd : float
        Standard deviation of control and treatment group
    mde : float
        Minimum detectable effect in absolute points
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha
    power : float
        Power, probability of detecting an effect given it exists
    ratio : float
        Ratio of number of users in treatment group by number of users in control group

    Returns
    -------
    n_control : int
            Sample size of control group
        n_treatment : int
            Sample size of treatment group

    """
    # Step 1: z-statistics
    z_a = NormalDist().inv_cdf(1 - alpha/2)
    z_b = NormalDist().inv_cdf(power)
    # Step 2: Sample size
    n_control = ((sd**2)*((z_a + z_b)**2)*(ratio+1))/(ratio*(mde**2))

    return math.ceil(n_control), math.ceil(ratio * n_control)

def mde_prop(p1, n_per_arm, alpha=0.05, power=0.8):
    """
    Computes Minimum Detectable Effect for proportional test. 
    
    Parameters
    ----------
    p1 : float
        Conversion rate for control group
    n_per_arm : float
        Sample size per arm
    alpha : float
        Significance level, with Confidence Interval covering 1 - alpha
    power : float
        Power, probability of detecting an effect given it exists

    Returns
    -------
    mde : float
        Minimum Detectable Effect

    Notes
    -----
    ratio is assumed to be one for this function meaning sample size of control group is equal to sample size of treatment group.
    mde is considered small so p2(1-p2) is approximately equal to p1(1-p1)
    assuming approximates p₂(1−p₂) ≈ p₁(1−p₁), one SE for both H₀ and H₁
    """
    # Step 1: z-statistics
    z_a = NormalDist().inv_cdf(1 - alpha/2)
    z_b = NormalDist().inv_cdf(power)
    # Step 2: MDE
    mde = (z_a + z_b)*((2*p1*(1-p1)/n_per_arm)**0.5)

    return mde