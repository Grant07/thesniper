"""
Smart Predictor for The Sniper Football Predictor
Uses cross-market analysis to find high-value betting opportunities
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from backend.betpawa_client import (
    BetpawaMatch,
    BetpawaOdds,
    get_true_probabilities
)


class MatchProfile(Enum):
    """Match profile based on odds analysis"""
    DOMINANT_HOME = "dominant_home"      # Heavy home favorite, likely high-scoring
    TIGHT_HOME = "tight_home"            # Home favorite, low-scoring expected
    DOMINANT_AWAY = "dominant_away"      # Heavy away favorite, likely high-scoring
    TIGHT_AWAY = "tight_away"            # Away favorite, low-scoring expected
    OPEN_GAME = "open_game"              # Even match, goals expected
    DEFENSIVE_BATTLE = "defensive_battle" # Even match, few goals expected
    UNCERTAIN = "uncertain"              # No clear pattern


@dataclass
class SmartBet:
    """Enhanced bet recommendation with cross-market analysis"""
    home_team: str
    away_team: str
    start_time: str
    competition: str

    # Primary bet
    bet_type: str
    confidence: float
    odds: float
    label: str

    # Cross-market validation
    match_profile: MatchProfile
    profile_confidence: float  # How well markets agree
    signals: List[str]  # Reasoning

    # Value assessment
    edge: float  # Expected edge over bookmaker
    kelly_fraction: float  # Suggested stake fraction


def analyze_match_profile(probs: dict) -> Tuple[MatchProfile, float, List[str]]:
    """
    Analyze match profile by correlating 1X2 and O/U markets.

    Returns:
        Tuple of (profile, confidence, signals)
    """
    signals = []

    home_prob = probs["home_win"]
    draw_prob = probs["draw"]
    away_prob = probs["away_win"]
    over_prob = probs.get("over_2_5", 0.5)
    under_prob = probs.get("under_2_5", 0.5)

    # Determine favorite
    max_1x2 = max(home_prob, draw_prob, away_prob)
    favorite = "home" if home_prob == max_1x2 else ("away" if away_prob == max_1x2 else "draw")

    # Favorite strength thresholds
    is_favorite = max_1x2 >= 0.55  # Slight favorite
    is_strong_favorite = max_1x2 >= 0.65  # Strong favorite (passes our threshold)

    # Goals expectation - use lower threshold for better detection
    expects_goals = over_prob >= 0.55
    expects_many_goals = over_prob >= 0.65
    expects_few_goals = under_prob >= 0.55

    # Profile determination
    profile_confidence = 0.5

    # Strong favorite scenarios (65%+)
    if is_strong_favorite and favorite == "home":
        if expects_goals:
            profile = MatchProfile.DOMINANT_HOME
            profile_confidence = min(home_prob, over_prob)
            signals.append(f"Home team heavily favored ({home_prob:.0%})")
            signals.append(f"Goals expected ({over_prob:.0%} over 2.5)")
            signals.append("Pattern: Dominant home win likely (2-0, 3-0, 3-1)")
        elif expects_few_goals:
            profile = MatchProfile.TIGHT_HOME
            profile_confidence = min(home_prob, under_prob)
            signals.append(f"Home team heavily favored ({home_prob:.0%})")
            signals.append(f"Few goals expected ({under_prob:.0%} under 2.5)")
            signals.append("Pattern: Controlled home win likely (1-0, 2-0)")
        else:
            # Favorite but O/U is neutral - still valid but lower confidence
            profile = MatchProfile.DOMINANT_HOME
            profile_confidence = home_prob * 0.8  # Reduced confidence
            signals.append(f"Home team heavily favored ({home_prob:.0%})")
            signals.append(f"Goals market neutral (O:{over_prob:.0%} U:{under_prob:.0%})")
            signals.append("Pattern: Home win likely, goal count uncertain")

    elif is_strong_favorite and favorite == "away":
        if expects_goals:
            profile = MatchProfile.DOMINANT_AWAY
            profile_confidence = min(away_prob, over_prob)
            signals.append(f"Away team heavily favored ({away_prob:.0%})")
            signals.append(f"Goals expected ({over_prob:.0%} over 2.5)")
            signals.append("Pattern: Dominant away win likely (0-2, 0-3, 1-3)")
        elif expects_few_goals:
            profile = MatchProfile.TIGHT_AWAY
            profile_confidence = min(away_prob, under_prob)
            signals.append(f"Away team heavily favored ({away_prob:.0%})")
            signals.append(f"Few goals expected ({under_prob:.0%} under 2.5)")
            signals.append("Pattern: Controlled away win likely (0-1, 0-2)")
        else:
            profile = MatchProfile.DOMINANT_AWAY
            profile_confidence = away_prob * 0.8
            signals.append(f"Away team heavily favored ({away_prob:.0%})")
            signals.append(f"Goals market neutral (O:{over_prob:.0%} U:{under_prob:.0%})")
            signals.append("Pattern: Away win likely, goal count uncertain")

    # Even match scenarios (no strong favorite)
    elif not is_favorite:
        if expects_goals:
            profile = MatchProfile.OPEN_GAME
            profile_confidence = over_prob * (1 - abs(home_prob - away_prob))
            signals.append(f"Evenly matched ({home_prob:.0%} vs {away_prob:.0%})")
            signals.append(f"Goals expected ({over_prob:.0%} over 2.5)")
            signals.append("Pattern: Open, attacking game likely (2-2, 3-2)")
        elif expects_few_goals:
            profile = MatchProfile.DEFENSIVE_BATTLE
            profile_confidence = under_prob * (1 - abs(home_prob - away_prob))
            signals.append(f"Evenly matched ({home_prob:.0%} vs {away_prob:.0%})")
            signals.append(f"Few goals expected ({under_prob:.0%} under 2.5)")
            signals.append("Pattern: Tight, defensive game likely (0-0, 1-0, 1-1)")
        else:
            profile = MatchProfile.UNCERTAIN
            signals.append(f"Evenly matched ({home_prob:.0%} vs {away_prob:.0%})")
            signals.append("No clear goal expectation")

    # Slight favorite but not strong (55-65%)
    elif is_favorite and not is_strong_favorite:
        if expects_many_goals:
            profile = MatchProfile.OPEN_GAME
            profile_confidence = over_prob
            signals.append(f"Slight favorite ({favorite}: {max_1x2:.0%})")
            signals.append(f"High-scoring expected ({over_prob:.0%} over 2.5)")
            signals.append("Pattern: Goals likely, winner less certain")
        elif expects_few_goals:
            profile = MatchProfile.DEFENSIVE_BATTLE
            profile_confidence = under_prob
            signals.append(f"Slight favorite ({favorite}: {max_1x2:.0%})")
            signals.append(f"Low-scoring expected ({under_prob:.0%} under 2.5)")
            signals.append("Pattern: Tight game, few goals")
        else:
            profile = MatchProfile.UNCERTAIN
            signals.append(f"Slight favorite ({favorite}: {max_1x2:.0%})")
            signals.append("No clear pattern from markets")
    else:
        profile = MatchProfile.UNCERTAIN
        signals.append("Mixed signals from markets")

    return profile, profile_confidence, signals


def calculate_edge(confidence: float, odds: float) -> float:
    """
    Calculate expected edge over bookmaker.

    Edge = (probability * odds) - 1
    Positive edge = value bet
    """
    expected_value = confidence * odds
    return expected_value - 1


def calculate_kelly(confidence: float, odds: float) -> float:
    """
    Calculate Kelly Criterion fraction for optimal stake sizing.

    Kelly = (bp - q) / b
    where b = odds - 1, p = win probability, q = lose probability

    We use fractional Kelly (25%) for safety.
    """
    b = odds - 1
    p = confidence
    q = 1 - p

    if b <= 0:
        return 0

    kelly = (b * p - q) / b

    # Use quarter Kelly for safety, cap at 5%
    return max(0, min(kelly * 0.25, 0.05))


def get_smart_bets(
    matches: List[BetpawaMatch],
    threshold_1x2: float = 0.65,
    threshold_ou: float = 0.70,
    min_edge: float = -0.10,  # Allow up to -10% edge (bookmaker margin reality)
    require_profile_match: bool = True,
    home_advantage_bias: bool = True  # Skip away wins (based on nothing 😂)
) -> List[SmartBet]:
    """
    Find high-value bets using cross-market analysis.

    Strategy:
    1. Convert odds to true probabilities
    2. Analyze match profile from market correlation
    3. Filter bets that align with profile
    4. Calculate edge and recommended stake

    Args:
        matches: List of BetpawaMatch objects
        threshold_1x2: Minimum probability for 1X2 bets
        threshold_ou: Minimum probability for O/U bets
        min_edge: Minimum expected edge (default 5%)
        require_profile_match: Only suggest bets that align with profile
        home_advantage_bias: Skip away wins because... reasons 🏠

    Returns:
        List of SmartBet objects, sorted by edge
    """
    smart_bets = []

    for match in matches:
        if not match.odds:
            continue

        probs = get_true_probabilities(match.odds)
        profile, profile_conf, signals = analyze_match_profile(probs)

        # Check for strong home win
        if probs["home_win"] >= threshold_1x2:
            edge = calculate_edge(probs["home_win"], match.odds.home_win)
            kelly = calculate_kelly(probs["home_win"], match.odds.home_win)

            # Profile alignment check
            profile_aligned = profile in [
                MatchProfile.DOMINANT_HOME,
                MatchProfile.TIGHT_HOME
            ]

            if edge >= min_edge and (not require_profile_match or profile_aligned):
                bet_signals = signals.copy()
                if profile_aligned:
                    bet_signals.append("Markets agree: Home win expected")

                smart_bets.append(SmartBet(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    start_time=match.start_time,
                    competition=match.competition,
                    bet_type="home_win",
                    confidence=probs["home_win"],
                    odds=match.odds.home_win,
                    label=f"Strong Home Win: {match.home_team}",
                    match_profile=profile,
                    profile_confidence=profile_conf,
                    signals=bet_signals,
                    edge=edge,
                    kelly_fraction=kelly
                ))

        # Check for strong away win (skip if home_advantage_bias is enabled 🏠)
        if not home_advantage_bias and probs["away_win"] >= threshold_1x2:
            edge = calculate_edge(probs["away_win"], match.odds.away_win)
            kelly = calculate_kelly(probs["away_win"], match.odds.away_win)

            profile_aligned = profile in [
                MatchProfile.DOMINANT_AWAY,
                MatchProfile.TIGHT_AWAY
            ]

            if edge >= min_edge and (not require_profile_match or profile_aligned):
                bet_signals = signals.copy()
                if profile_aligned:
                    bet_signals.append("Markets agree: Away win expected")

                smart_bets.append(SmartBet(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    start_time=match.start_time,
                    competition=match.competition,
                    bet_type="away_win",
                    confidence=probs["away_win"],
                    odds=match.odds.away_win,
                    label=f"Strong Away Win: {match.away_team}",
                    match_profile=profile,
                    profile_confidence=profile_conf,
                    signals=bet_signals,
                    edge=edge,
                    kelly_fraction=kelly
                ))

        # Check for over 2.5 - but validate against profile
        if probs.get("over_2_5") and probs["over_2_5"] >= threshold_ou:
            edge = calculate_edge(probs["over_2_5"], match.odds.over_2_5)
            kelly = calculate_kelly(probs["over_2_5"], match.odds.over_2_5)

            profile_aligned = profile in [
                MatchProfile.DOMINANT_HOME,
                MatchProfile.DOMINANT_AWAY,
                MatchProfile.OPEN_GAME
            ]

            if edge >= min_edge and (not require_profile_match or profile_aligned):
                bet_signals = signals.copy()
                if profile_aligned:
                    bet_signals.append("Markets agree: Goals expected")

                smart_bets.append(SmartBet(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    start_time=match.start_time,
                    competition=match.competition,
                    bet_type="over_2_5",
                    confidence=probs["over_2_5"],
                    odds=match.odds.over_2_5,
                    label="High Confidence Over 2.5 Goals",
                    match_profile=profile,
                    profile_confidence=profile_conf,
                    signals=bet_signals,
                    edge=edge,
                    kelly_fraction=kelly
                ))

        # Check for under 2.5 - validate against profile
        if probs.get("under_2_5") and probs["under_2_5"] >= threshold_ou:
            edge = calculate_edge(probs["under_2_5"], match.odds.under_2_5)
            kelly = calculate_kelly(probs["under_2_5"], match.odds.under_2_5)

            profile_aligned = profile in [
                MatchProfile.TIGHT_HOME,
                MatchProfile.TIGHT_AWAY,
                MatchProfile.DEFENSIVE_BATTLE
            ]

            if edge >= min_edge and (not require_profile_match or profile_aligned):
                bet_signals = signals.copy()
                if profile_aligned:
                    bet_signals.append("Markets agree: Few goals expected")

                smart_bets.append(SmartBet(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    start_time=match.start_time,
                    competition=match.competition,
                    bet_type="under_2_5",
                    confidence=probs["under_2_5"],
                    odds=match.odds.under_2_5,
                    label="High Confidence Under 2.5 Goals",
                    match_profile=profile,
                    profile_confidence=profile_conf,
                    signals=bet_signals,
                    edge=edge,
                    kelly_fraction=kelly
                ))

    # Sort by edge (best value first)
    smart_bets.sort(key=lambda x: x.edge, reverse=True)

    return smart_bets


def format_smart_bet(bet: SmartBet) -> str:
    """Format a smart bet for display."""
    lines = [
        f"{'=' * 60}",
        f"{bet.home_team} vs {bet.away_team}",
        f"Competition: {bet.competition}",
        f"",
        f"BET: {bet.label}",
        f"Confidence: {bet.confidence:.1%} | Odds: {bet.odds:.2f}",
        f"Edge: {bet.edge:.1%} | Kelly Stake: {bet.kelly_fraction:.1%} of bankroll",
        f"",
        f"Match Profile: {bet.match_profile.value.replace('_', ' ').title()}",
        f"Profile Confidence: {bet.profile_confidence:.1%}",
        f"",
        "Analysis:",
    ]
    for signal in bet.signals:
        lines.append(f"  - {signal}")

    return "\n".join(lines)
