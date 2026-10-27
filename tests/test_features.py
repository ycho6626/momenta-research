"""Hand L1 identities, factor residuals and information timing (Methodology)."""


from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from momenta.features import (
    average_residual_correlation,
    completed_week_indices,
    historical_book_variance,
    log_realized_variance,
    macro_features,
    quantile_skewness,
    rolling_beta,
    weekly_compound,
)


def test_realized_variance_uses_squared_returns_without_demeaning_and_full_windows() -> None:
    values = np.array([.1, -.2, .3, np.nan])
    actual = log_realized_variance(values, 2, 21)
    assert np.isnan(actual[0])
    assert actual[1] == pytest.approx(np.log(21*.025))
    assert actual[2] == pytest.approx(np.log(21*.065))
    assert np.isnan(actual[3])
    np.testing.assert_array_equal(actual[:3], log_realized_variance(values[:3], 2, 21))


def test_beta_includes_intercept_and_uses_current_observation() -> None:
    market = np.array([-.03, .02, .01, -.01])
    book = .02 - 2*market
    actual = rolling_beta(book, market, 3)
    assert np.isnan(actual[:2]).all()
    np.testing.assert_allclose(actual[2:], -2)


def test_book_risk_uses_fixed_signed_weights_missing_zero_and_weighted_coverage() -> None:
    panel = np.array([[.1, .2, .3], [-.1, np.nan, .1]])
    weights = np.array([.75, .25, -1.])
    # Fixed book returns -.175 and -.175; second coverage is 1.75/2.
    variance, coverage = historical_book_variance(panel, weights, 21)
    assert variance == pytest.approx(np.log(21*.175**2))
    assert coverage == pytest.approx((1+1.75/2)/2)


def test_quantile_skewness_uses_registered_quantiles_and_rejects_constant_series() -> None:
    assert quantile_skewness(np.array([0., 1., 2., 3., 8.]), [0, .5, 1]) == .5
    assert np.isnan(quantile_skewness(np.ones(5), [.05, .5, .95]))


def test_residual_correlation_removes_ff3_and_rf_and_requires_balanced_panel() -> None:
    # Orthogonal Hadamard columns give an independent exact correlation of .6.
    a = np.array([1, -1, 1, -1, 1, -1, 1, -1])*.01
    b = np.array([1, 1, -1, -1, 1, 1, -1, -1])*.01
    c = np.array([1, 1, 1, 1, -1, -1, -1, -1])*.01
    d = np.array([1, -1, -1, 1, 1, -1, -1, 1])*.01
    e = np.array([1, -1, 1, -1, -1, 1, -1, 1])*.01
    rf = np.array([1, 1, -1, -1, -1, -1, 1, 1])*.02
    ff = np.column_stack([c, d, e, rf])
    stocks = np.column_stack([.02+3*c-2*d+e+rf+a,
                              -.01-c+4*d-2*e+rf+.6*a+.8*b, a])
    stocks[3, 2] = np.nan
    result, count = average_residual_correlation(stocks, ff)
    assert count == 2
    assert result == pytest.approx(.6, abs=1e-12)


def test_weekly_returns_do_not_include_partial_future_calendar_week() -> None:
    start = date(2001, 1, 1)
    calendar = [start+timedelta(days=i) for i in range(19)
                if (start+timedelta(days=i)).weekday() < 5]
    groups = completed_week_indices(calendar, date(2001, 1, 10))
    assert len(groups) == 1
    assert calendar[groups[0][-1]] == date(2001, 1, 5)
    returns = np.full((len(calendar), 2), .1)
    returns[2, 1] = np.nan
    compounded = weekly_compound(returns, groups)
    assert compounded[0, 0] == pytest.approx(1.1**5-1)
    assert np.isnan(compounded[0, 1])


def test_macro_features_forward_fill_observed_values_without_future_or_recession_input() -> None:
    dates = pl.DataFrame({"date": [date(2001, 1, 2), date(2001, 1, 3), date(2001, 1, 4)]})
    data = pl.DataFrame({"date": [date(2001, 1, 1), date(2001, 1, 3), date(2001, 1, 5)],
                         "value": [10., None, 1000.]})
    fred = {key: data for key in ("VIXCLS", "BAA10Y", "T10Y3M", "USREC")}
    actual = macro_features(dates, fred)
    np.testing.assert_allclose(actual["log_vix"].to_numpy(), np.log(10))
    assert actual["baa10y"].to_list() == [10, 10, 10]
    assert "USREC" not in actual.columns
