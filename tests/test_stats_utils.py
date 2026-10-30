import pytest
from ds_prep.stats_utils import two_prop_ztest, welch_ttest, bootstrap_ci
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

@pytest.mark.parametrize(
    "x1,n1,x2,n2", [
        (1200, 10000, 1260, 10000),
        (2160, 18000, 252, 2000),
        (1200, 10000, 1140, 10000),
        (1200, 10000, 1200, 10000),
    ]
)
def test_ztest_matches_statsmodels(x1, n1, x2, n2):
    # Our Result
    our_result = two_prop_ztest(x1, n1, x2, n2)
    our_stat = our_result["z"]
    our_p_value = our_result["p_value"]
    # Reference
    ref_stat, ref_p_value = proportions_ztest([x1, x2], [n1, n2])
    # assert
    assert our_stat == pytest.approx(ref_stat)
    assert our_p_value == pytest.approx(ref_p_value)

@pytest.mark.parametrize(
    "x1,n1,x2,n2", [
        (1200, 10000, 1260, 10000),
        (2160, 18000, 252, 2000),
        (1200, 10000, 1140, 10000),
        (1200, 10000, 1200, 10000),
    ]
)
def test_ztest_order_invariance(x1, n1, x2, n2):
    # Our Result
    our_result = two_prop_ztest(x1, n1, x2, n2)
    our_stat = our_result["z"]
    our_p_value = our_result["p_value"]
    # Flipped result
    flipped_result = two_prop_ztest(x2, n2, x1, n1)
    flipped_stat = flipped_result["z"]
    flipped_p_value = flipped_result["p_value"]
    # assert
    assert our_stat == pytest.approx(-flipped_stat)
    assert our_p_value == pytest.approx(flipped_p_value)

@pytest.mark.parametrize(
    "x1,n1,x2,n2", [
        (10000, 10000, 10000, 10000),
        (0, 10000, 0, 10000)
    ]
)
def test_ztest_rejects_0_1_pooled_rate(x1, n1, x2, n2):
    with pytest.raises(ValueError, match = "pooled rate"):
        two_prop_ztest(x1, n1, x2, n2)

@pytest.mark.parametrize(
    "x1,n1,x2,n2", [
        (10000, 0, 1260, 10000),
        (11000, 10000, 1260, 10000)
    ]
)
def test_ztest_rejects_invalid_counts(x1, n1, x2, n2):
    with pytest.raises(ValueError, match = "invalid inputs"):
        two_prop_ztest(x1, n1, x2, n2)

@pytest.mark.parametrize(
    "a,b", [
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9]),
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4]),
        (
            [42.0, 38.5, 51.2, 47.8, 39.9, 44.1, 120.0, 41.3, 45.6, 40.2, 43.8, 49.5, 37.6, 46.9, 42.7, 50.3, 39.1, 44.8, 48.2, 41.9],
            [35.2, 40.8, 33.9, 38.4, 36.7]
        )
    ]
)
def test_welch_matches_scipy(a, b):
    # Our Result
    our_result = welch_ttest(a, b)
    our_stat = our_result["t"]
    our_p_value = our_result["p_value"]
    our_df = our_result["df"]
    # Reference
    ref_result = stats.ttest_ind(a, b, equal_var=False)
    ref_stat = ref_result.statistic
    ref_p_value = ref_result.pvalue
    ref_df = ref_result.df  
    # assert
    assert our_stat == pytest.approx(ref_stat)
    assert our_p_value == pytest.approx(ref_p_value)
    assert our_df == pytest.approx(ref_df)

@pytest.mark.parametrize(
    "a,b", [
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9]),
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4]),
        (
            [42.0, 38.5, 51.2, 47.8, 39.9, 44.1, 120.0, 41.3, 45.6, 40.2, 43.8, 49.5, 37.6, 46.9, 42.7, 50.3, 39.1, 44.8, 48.2, 41.9],
            [35.2, 40.8, 33.9, 38.4, 36.7]
        )
    ]
)
def test_welch_order_invariance(a, b):
    # Our Result
    our_result = welch_ttest(a, b)
    our_stat = our_result["t"]
    our_p_value = our_result["p_value"]
    our_df = our_result["df"]
    # Flipped Result
    flipped_result = welch_ttest(b, a)
    flipped_stat = flipped_result["t"]
    flipped_p_value = flipped_result["p_value"]
    flipped_df = flipped_result["df"] 
    # assert
    assert our_stat == pytest.approx(-flipped_stat)
    assert our_p_value == pytest.approx(flipped_p_value)
    assert our_df == pytest.approx(flipped_df)

@pytest.mark.parametrize(
    "a,b", [([16.2, 16.2, 16.2, 16.2, 16.2, 16.2, 16.2, 16.2], [15.0, 15.0, 15.0, 15.0, 15.0, 15.0])]
)
def test_welch_0_variance(a, b):
    with pytest.raises(ValueError, match = "Variance"):
        welch_ttest(a, b)

@pytest.mark.parametrize(
    "a,b", [
        ([16.2], [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0])
    ]
)
def test_welch_size_1_group(a, b):
    with pytest.raises(ValueError, match = "invalid inputs"):
        welch_ttest(a, b)

@pytest.mark.parametrize(
    "a,b,seed", [
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9], 0),
        (
            [42.0, 38.5, 51.2, 47.8, 39.9, 44.1, 120.0, 41.3, 45.6, 40.2, 43.8, 49.5, 37.6, 46.9, 42.7, 50.3, 39.1, 44.8, 48.2, 41.9],
            [35.2, 40.8, 33.9, 38.4, 36.7],
            7
        )
    ]
)
def test_bootstrap_reproducible(a, b, seed):
    # 1st Result
    result_1 = bootstrap_ci(a, b, seed = seed)
    low_ci_1 = result_1["ci_low"]
    high_ci_1 = result_1["ci_high"]
    # 2nd Result
    result_2 = bootstrap_ci(a, b, seed = seed)
    low_ci_2 = result_2["ci_low"]
    high_ci_2 = result_2["ci_high"]
    # assert
    assert low_ci_1 == low_ci_2
    assert high_ci_1 == high_ci_2

@pytest.mark.parametrize(
    "a,b", [
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9]),
        (
            [42.0, 38.5, 51.2, 47.8, 39.9, 44.1, 120.0, 41.3, 45.6, 40.2, 43.8, 49.5, 37.6, 46.9, 42.7, 50.3, 39.1, 44.8, 48.2, 41.9],
            [35.2, 40.8, 33.9, 38.4, 36.7]
        )
    ]
)
def test_bootstrap_ci_contains_estimate(a, b):
    # Result
    result = bootstrap_ci(a, b)
    low_ci = result["ci_low"]
    high_ci = result["ci_high"]
    estimate = result["diff"]
    # assert
    assert low_ci <= estimate
    assert estimate <= high_ci

@pytest.mark.parametrize(
    "a,b,alphas", [
        ([16.2, 14.8, 17.5, 15.9, 16.8, 15.1, 17.2, 16.4], [15.0, 14.1, 15.8, 14.6, 16.2, 13.9], [0.05, 0.01]),
        (
            [42.0, 38.5, 51.2, 47.8, 39.9, 44.1, 120.0, 41.3, 45.6, 40.2, 43.8, 49.5, 37.6, 46.9, 42.7, 50.3, 39.1, 44.8, 48.2, 41.9],
            [35.2, 40.8, 33.9, 38.4, 36.7],
            [0.05, 0.01]
        )
    ]
)
def test_bootstrap_wider_at_higher_confidence(a, b, alphas):
    # 1st Result
    result_1 = bootstrap_ci(a, b, alpha = alphas[0])
    low_ci_1 = result_1["ci_low"]
    high_ci_1 = result_1["ci_high"]
    # 2nd Result
    result_2 = bootstrap_ci(a, b, alpha = alphas[1])
    low_ci_2 = result_2["ci_low"]
    high_ci_2 = result_2["ci_high"]
    # assert
    if alphas[0] > alphas[1]:
        assert high_ci_2 - low_ci_2 >= high_ci_1 - low_ci_1
    elif alphas[0] < alphas[1]:
        assert high_ci_2 - low_ci_2 <= high_ci_1 - low_ci_1
    else:
        assert high_ci_2 == high_ci_1
        assert low_ci_2 == low_ci_1