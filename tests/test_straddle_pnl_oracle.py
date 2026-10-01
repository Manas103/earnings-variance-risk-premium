"""Hand-computed / closed-form reference cases for options pricing and the
straddle P&L pipeline, diffed exactly (options) or to a tight numerical
tolerance (the quad-integration oracle) against the production code.
"""
import math

import pytest

from evrp import options
from evrp.pnl import price_event
from evrp.reference_oracle import bs_price_quad, straddle_price_quad, straddle_pnl_loop


def test_bs_price_zero_vol_is_intrinsic():
    # Hand-computed: at sigma -> 0, a call is worth exactly max(S-K, 0), no time value.
    assert options.bs_price(110.0, 100.0, 2 / 252, 0.0, True) == pytest.approx(10.0, abs=1e-12)
    assert options.bs_price(90.0, 100.0, 2 / 252, 0.0, False) == pytest.approx(10.0, abs=1e-12)
    assert options.bs_price(100.0, 100.0, 2 / 252, 0.0, True) == pytest.approx(0.0, abs=1e-12)


def test_bs_price_expired_is_intrinsic():
    assert options.bs_price(105.0, 100.0, 0.0, 0.30, True) == pytest.approx(5.0, abs=1e-12)
    assert options.bs_price(95.0, 100.0, 0.0, 0.30, False) == pytest.approx(5.0, abs=1e-12)


def test_atm_straddle_hand_computed_case():
    # Hand computation, S=K=100, T=2/252, sigma=0.30, r=0:
    # d1 = 0.5*sigma*sqrt(T); d2 = -d1 (since log(S/K) = 0).
    spot, strike, t, sigma = 100.0, 100.0, 2 / 252, 0.30
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = 0.5 * vol_sqrt_t
    norm_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    call = spot * norm_cdf(d1) - strike * norm_cdf(-d1)
    put = strike * norm_cdf(d1) - spot * norm_cdf(-d1)
    expected_straddle = call + put

    assert options.bs_price(spot, strike, t, sigma, True) == pytest.approx(call, abs=1e-12)
    assert options.bs_price(spot, strike, t, sigma, False) == pytest.approx(put, abs=1e-12)
    assert options.straddle_price(spot, strike, t, sigma) == pytest.approx(expected_straddle, abs=1e-12)


@pytest.mark.parametrize("spot,strike,t,sigma,is_call", [
    (100.0, 100.0, 2 / 252, 0.30, True),
    (100.0, 100.0, 2 / 252, 0.30, False),
    (105.0, 100.0, 1 / 252, 0.55, True),
    (95.0, 100.0, 3 / 252, 0.20, False),
])
def test_closed_form_matches_independent_quad_integration_oracle(spot, strike, t, sigma, is_call):
    closed = options.bs_price(spot, strike, t, sigma, is_call)
    quad = bs_price_quad(spot, strike, t, sigma, is_call)
    assert closed == pytest.approx(quad, abs=5e-4)


def test_straddle_price_matches_quad_oracle():
    closed = options.straddle_price(100.0, 100.0, 2 / 252, 0.35)
    quad = straddle_price_quad(100.0, 100.0, 2 / 252, 0.35)
    assert closed == pytest.approx(quad, abs=1e-3)


@pytest.mark.parametrize("s_entry,s_mid,s_exit,trailing_vol,implied_event_vol", [
    (100.0, 102.0, 101.0, 25.0, 40.0),
    (50.0, 48.0, 49.5, 30.0, 10.0),
    (200.0, 220.0, 215.0, 20.0, 60.0),
    (75.0, 75.0, 75.0, 35.0, 5.0),   # degenerate: no move at all
])
def test_production_pnl_matches_independent_loop_oracle(s_entry, s_mid, s_exit, trailing_vol, implied_event_vol):
    from evrp.pnl import OPTION_HALF_SPREAD_FRAC, STOCK_HALF_SPREAD_BPS
    p = price_event(s_entry, s_mid, s_exit, trailing_vol, implied_event_vol)
    # price_event's defaults are module constants; pass them explicitly to the oracle too.
    ref = straddle_pnl_loop(s_entry, s_mid, s_exit, trailing_vol, implied_event_vol,
                             OPTION_HALF_SPREAD_FRAC, STOCK_HALF_SPREAD_BPS)
    assert p.premium_received == pytest.approx(ref["premium"], abs=1e-9)
    assert p.mark_value == pytest.approx(ref["mark"], abs=1e-9)
    assert p.gross_pnl == pytest.approx(ref["gross_pnl"], abs=1e-9)
    assert p.net_pnl == pytest.approx(ref["net_pnl"], abs=1e-9)


def test_straddle_pnl_sign_short_vol_loses_on_big_move_gross():
    # A huge, unhedgeable-in-one-step move should produce a gross loss on the short
    # straddle net of hedge p&l being nowhere near enough to offset the mark paid.
    p = price_event(s_entry=100.0, s_mid=100.0, s_exit=160.0,
                     trailing_vol_long_points=20.0, implied_event_vol_points=30.0)
    assert p.gross_pnl < 0
