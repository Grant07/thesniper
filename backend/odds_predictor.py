"""
Odds-Based Predictor for The Sniper Football Predictor
Uses bookmaker odds to derive probabilities (alternative to Poisson model)
"""

from typing import List, Optional
from dataclasses import dataclass

from backend.betpawa_client import (
    BetpawaMatch,
    BetpawaOdds,
    get_true_probabilities
)


@dataclass
class OddsPrediction:
    """Prediction based on bookmaker odds"""
    home_team: str
    away_team: str
    start_time: str
    competition: str

    # Raw odds
    home_odds: float
    draw_odds: float
    away_odds: float
    over_2_5_odds: Optional[float]
    under_2_5_odds: Optional[float]

    # True probabilities (margin removed)
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5_prob: Optional[float]
    under_2_5_prob: Optional[float]


@dataclass
class SniperBet:
    """A high-confidence bet recommendation"""
    home_team: str
    away_team: str
    start_time: str
    competition: str
    bet_type: str  # "home_win", "away_win", "over_2_5", "under_2_5"
    confidence: float
    odds: float
    label: str


def match_to_prediction(match: BetpawaMatch) -> Optional[OddsPrediction]:
    """
    Convert a BetpawaMatch to an OddsPrediction.

    Args:
        match: BetpawaMatch with odds

    Returns:
        OddsPrediction or None if odds unavailable
    """
    if not match.odds:
        return None

    probs = get_true_probabilities(match.odds)

    return OddsPrediction(
        home_team=match.home_team,
        away_team=match.away_team,
        start_time=match.start_time,
        competition=match.competition,
        home_odds=match.odds.home_win,
        draw_odds=match.odds.draw,
        away_odds=match.odds.away_win,
        over_2_5_odds=match.odds.over_2_5,
        under_2_5_odds=match.odds.under_2_5,
        home_win_prob=probs["home_win"],
        draw_prob=probs["draw"],
        away_win_prob=probs["away_win"],
        over_2_5_prob=probs.get("over_2_5"),
        under_2_5_prob=probs.get("under_2_5"),
    )


def get_sniper_bets_from_odds(
    matches: List[BetpawaMatch],
    threshold_1x2: float = 0.65,
    threshold_ou: float = 0.70
) -> List[SniperBet]:
    """
    Find high-confidence bets from bookmaker odds.

    The Sniper Strategy: Only bet when implied probability exceeds thresholds.
    Uses bookmaker's own analysis (embedded in odds) rather than our model.

    Args:
        matches: List of BetpawaMatch objects
        threshold_1x2: Minimum probability for 1X2 bets (default 65%)
        threshold_ou: Minimum probability for Over/Under bets (default 70%)

    Returns:
        List of SniperBet objects meeting confidence thresholds
    """
    sniper_bets = []

    for match in matches:
        pred = match_to_prediction(match)
        if not pred:
            continue

        # Check for strong home win
        if pred.home_win_prob > threshold_1x2:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                start_time=pred.start_time,
                competition=pred.competition,
                bet_type="home_win",
                confidence=pred.home_win_prob,
                odds=pred.home_odds,
                label=f"Strong Home Win: {pred.home_team}"
            ))

        # Check for strong away win
        if pred.away_win_prob > threshold_1x2:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                start_time=pred.start_time,
                competition=pred.competition,
                bet_type="away_win",
                confidence=pred.away_win_prob,
                odds=pred.away_odds,
                label=f"Strong Away Win: {pred.away_team}"
            ))

        # Check for high confidence over 2.5
        if pred.over_2_5_prob and pred.over_2_5_prob > threshold_ou:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                start_time=pred.start_time,
                competition=pred.competition,
                bet_type="over_2_5",
                confidence=pred.over_2_5_prob,
                odds=pred.over_2_5_odds,
                label="High Confidence Over 2.5 Goals"
            ))

        # Check for high confidence under 2.5
        if pred.under_2_5_prob and pred.under_2_5_prob > threshold_ou:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                start_time=pred.start_time,
                competition=pred.competition,
                bet_type="under_2_5",
                confidence=pred.under_2_5_prob,
                odds=pred.under_2_5_odds,
                label="High Confidence Under 2.5 Goals"
            ))

    # Sort by confidence (highest first)
    sniper_bets.sort(key=lambda x: x.confidence, reverse=True)

    return sniper_bets


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime to readable string."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except:
        return iso_string
