"""
Poisson Model for Football Match Prediction
The mathematical engine behind "The Sniper" strategy
"""

from scipy.stats import poisson
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TeamStats:
    """Team statistics for prediction"""
    name: str
    matches_played: int
    goals_for: int
    goals_against: int
    home_gf: Optional[int] = None
    home_ga: Optional[int] = None
    away_gf: Optional[int] = None
    away_ga: Optional[int] = None


@dataclass
class MatchPrediction:
    """Prediction results for a match"""
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5_prob: float
    under_2_5_prob: float
    home_expected_goals: float
    away_expected_goals: float
    most_likely_score: Tuple[int, int]
    score_probability: float


@dataclass
class SniperBet:
    """A high-confidence bet recommendation"""
    home_team: str
    away_team: str
    bet_type: str  # "home_win", "away_win", "over_2_5", "under_2_5"
    confidence: float
    label: str  # Human-readable label


class PoissonModel:
    """
    Poisson distribution model for football match prediction.

    The model calculates expected goals based on team attack/defense strengths
    relative to league averages, then uses Poisson distribution to generate
    probability matrices for all possible scorelines.
    """

    HOME_ADVANTAGE = 1.1  # Home teams typically score ~10% more
    MAX_GOALS = 6  # Consider scores from 0-0 to 5-5

    def __init__(self, use_home_away_split: bool = True):
        """
        Initialize the Poisson model.

        Args:
            use_home_away_split: If True, uses separate home/away stats for
                                 more accurate predictions.
        """
        self.use_home_away_split = use_home_away_split

    def calculate_attack_strength(
        self,
        team_gf: int,
        team_mp: int,
        league_avg: float
    ) -> float:
        """
        Calculate team's attack strength relative to league average.

        Attack Strength = (Team Goals / Team Matches) / League Average Goals

        Args:
            team_gf: Team goals for (scored)
            team_mp: Team matches played
            league_avg: League average goals per team per match

        Returns:
            Attack strength ratio (>1 = above average, <1 = below average)
        """
        if team_mp == 0 or league_avg == 0:
            return 1.0
        team_avg = team_gf / team_mp
        return team_avg / league_avg

    def calculate_defense_strength(
        self,
        team_ga: int,
        team_mp: int,
        league_avg: float
    ) -> float:
        """
        Calculate team's defense strength (goals conceded) relative to league average.

        Defense Strength = (Team Goals Against / Team Matches) / League Average Goals

        Note: Higher value = worse defense (concedes more)

        Args:
            team_ga: Team goals against (conceded)
            team_mp: Team matches played
            league_avg: League average goals per team per match

        Returns:
            Defense strength ratio (>1 = below average, <1 = above average)
        """
        if team_mp == 0 or league_avg == 0:
            return 1.0
        team_avg = team_ga / team_mp
        return team_avg / league_avg

    def calculate_expected_goals(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        league_avg: float
    ) -> Tuple[float, float]:
        """
        Calculate expected goals (lambda) for each team in a match.

        Home Expected Goals = Home Attack * Away Defense * League Avg * Home Advantage
        Away Expected Goals = Away Attack * Home Defense * League Avg

        Args:
            home_stats: Statistics for the home team
            away_stats: Statistics for the away team
            league_avg: League average goals per team per match

        Returns:
            Tuple of (home_expected_goals, away_expected_goals)
        """
        if self.use_home_away_split and home_stats.home_gf is not None:
            # Use home/away specific stats for more accuracy
            home_mp = home_stats.matches_played // 2 or 1
            away_mp = away_stats.matches_played // 2 or 1

            home_attack = self.calculate_attack_strength(
                home_stats.home_gf, home_mp, league_avg
            )
            home_defense = self.calculate_defense_strength(
                home_stats.home_ga, home_mp, league_avg
            )
            away_attack = self.calculate_attack_strength(
                away_stats.away_gf, away_mp, league_avg
            )
            away_defense = self.calculate_defense_strength(
                away_stats.away_ga, away_mp, league_avg
            )
        else:
            # Use overall stats
            home_attack = self.calculate_attack_strength(
                home_stats.goals_for, home_stats.matches_played, league_avg
            )
            home_defense = self.calculate_defense_strength(
                home_stats.goals_against, home_stats.matches_played, league_avg
            )
            away_attack = self.calculate_attack_strength(
                away_stats.goals_for, away_stats.matches_played, league_avg
            )
            away_defense = self.calculate_defense_strength(
                away_stats.goals_against, away_stats.matches_played, league_avg
            )

        # Calculate expected goals
        home_exp = home_attack * away_defense * league_avg * self.HOME_ADVANTAGE
        away_exp = away_attack * home_defense * league_avg

        # Ensure minimum expected goals to avoid edge cases
        home_exp = max(0.1, home_exp)
        away_exp = max(0.1, away_exp)

        return home_exp, away_exp

    def generate_probability_matrix(
        self,
        home_lambda: float,
        away_lambda: float
    ) -> List[List[float]]:
        """
        Generate a probability matrix for all possible scorelines.

        Uses Poisson distribution: P(k goals) = (lambda^k * e^-lambda) / k!

        Args:
            home_lambda: Expected goals for home team
            away_lambda: Expected goals for away team

        Returns:
            6x6 matrix where matrix[h][a] = P(Home scores h, Away scores a)
        """
        matrix = []
        for home_goals in range(self.MAX_GOALS):
            row = []
            for away_goals in range(self.MAX_GOALS):
                prob = (
                    poisson.pmf(home_goals, home_lambda) *
                    poisson.pmf(away_goals, away_lambda)
                )
                row.append(prob)
            matrix.append(row)
        return matrix

    def calculate_market_probabilities(
        self,
        matrix: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate probabilities for different betting markets.

        Args:
            matrix: Probability matrix from generate_probability_matrix()

        Returns:
            Dictionary with probabilities for each market
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        over_2_5 = 0.0
        under_2_5 = 0.0

        for home_goals in range(self.MAX_GOALS):
            for away_goals in range(self.MAX_GOALS):
                prob = matrix[home_goals][away_goals]
                total_goals = home_goals + away_goals

                # 1X2 Market
                if home_goals > away_goals:
                    home_win += prob
                elif home_goals < away_goals:
                    away_win += prob
                else:
                    draw += prob

                # Over/Under 2.5 Market
                if total_goals > 2.5:
                    over_2_5 += prob
                else:
                    under_2_5 += prob

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_2_5": over_2_5,
            "under_2_5": under_2_5
        }

    def get_most_likely_score(
        self,
        matrix: List[List[float]]
    ) -> Tuple[Tuple[int, int], float]:
        """
        Find the most likely scoreline from the probability matrix.

        Args:
            matrix: Probability matrix

        Returns:
            Tuple of ((home_goals, away_goals), probability)
        """
        max_prob = 0.0
        most_likely = (0, 0)

        for home_goals in range(self.MAX_GOALS):
            for away_goals in range(self.MAX_GOALS):
                if matrix[home_goals][away_goals] > max_prob:
                    max_prob = matrix[home_goals][away_goals]
                    most_likely = (home_goals, away_goals)

        return most_likely, max_prob

    def predict_match(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        league_avg: float
    ) -> MatchPrediction:
        """
        Generate a complete prediction for a match.

        Args:
            home_stats: Statistics for the home team
            away_stats: Statistics for the away team
            league_avg: League average goals per team per match

        Returns:
            MatchPrediction with all probabilities and expected values
        """
        # Calculate expected goals
        home_exp, away_exp = self.calculate_expected_goals(
            home_stats, away_stats, league_avg
        )

        # Generate probability matrix
        matrix = self.generate_probability_matrix(home_exp, away_exp)

        # Calculate market probabilities
        probs = self.calculate_market_probabilities(matrix)

        # Get most likely score
        most_likely, score_prob = self.get_most_likely_score(matrix)

        return MatchPrediction(
            home_team=home_stats.name,
            away_team=away_stats.name,
            home_win_prob=probs["home_win"],
            draw_prob=probs["draw"],
            away_win_prob=probs["away_win"],
            over_2_5_prob=probs["over_2_5"],
            under_2_5_prob=probs["under_2_5"],
            home_expected_goals=home_exp,
            away_expected_goals=away_exp,
            most_likely_score=most_likely,
            score_probability=score_prob
        )


def get_sniper_bets(
    predictions: List[MatchPrediction],
    threshold_1x2: float = 0.65,
    threshold_ou: float = 0.70
) -> List[SniperBet]:
    """
    Filter predictions to only return high-confidence "sniper" bets.

    The Sniper Strategy: Only bet when mathematical confidence exceeds thresholds.
    This reduces variance and improves long-term profitability.

    Args:
        predictions: List of match predictions
        threshold_1x2: Minimum probability for 1X2 bets (default 65%)
        threshold_ou: Minimum probability for Over/Under bets (default 70%)

    Returns:
        List of SniperBet objects that meet the confidence thresholds
    """
    sniper_bets = []

    for pred in predictions:
        # Check for strong home win
        if pred.home_win_prob > threshold_1x2:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                bet_type="home_win",
                confidence=pred.home_win_prob,
                label=f"Strong Home Win: {pred.home_team}"
            ))

        # Check for strong away win
        if pred.away_win_prob > threshold_1x2:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                bet_type="away_win",
                confidence=pred.away_win_prob,
                label=f"Strong Away Win: {pred.away_team}"
            ))

        # Check for high confidence over 2.5
        if pred.over_2_5_prob > threshold_ou:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                bet_type="over_2_5",
                confidence=pred.over_2_5_prob,
                label="High Confidence Over 2.5 Goals"
            ))

        # Check for high confidence under 2.5
        if pred.under_2_5_prob > threshold_ou:
            sniper_bets.append(SniperBet(
                home_team=pred.home_team,
                away_team=pred.away_team,
                bet_type="under_2_5",
                confidence=pred.under_2_5_prob,
                label="High Confidence Under 2.5 Goals"
            ))

    # Sort by confidence (highest first)
    sniper_bets.sort(key=lambda x: x.confidence, reverse=True)

    return sniper_bets
