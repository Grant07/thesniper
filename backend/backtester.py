"""
Backtesting Module for The Sniper Football Predictor
Validates model accuracy against historical results
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd

from backend.predictor import PoissonModel, MatchPrediction, SniperBet, get_sniper_bets
from backend.data_loader import DataLoader


@dataclass
class BacktestResult:
    """Results from a single backtest prediction"""
    home_team: str
    away_team: str
    predicted_bet: str
    predicted_confidence: float
    actual_home_goals: int
    actual_away_goals: int
    was_correct: bool


@dataclass
class BacktestSummary:
    """Aggregated backtest results"""
    total_bets: int
    correct_bets: int
    accuracy: float
    by_market: Dict[str, Dict[str, float]] = field(default_factory=dict)
    results: List[BacktestResult] = field(default_factory=list)


class Backtester:
    """
    Backtesting engine to validate model accuracy against historical data.

    This helps verify the 60%+ accuracy claim of the sniper strategy.
    """

    def __init__(
        self,
        model: PoissonModel = None,
        threshold_1x2: float = 0.65,
        threshold_ou: float = 0.70
    ):
        """
        Initialize the backtester.

        Args:
            model: PoissonModel instance (creates new one if None)
            threshold_1x2: Threshold for 1X2 bets
            threshold_ou: Threshold for Over/Under bets
        """
        self.model = model or PoissonModel()
        self.threshold_1x2 = threshold_1x2
        self.threshold_ou = threshold_ou

    def check_bet_outcome(
        self,
        bet: SniperBet,
        home_goals: int,
        away_goals: int
    ) -> bool:
        """
        Check if a bet was successful based on actual result.

        Args:
            bet: The SniperBet prediction
            home_goals: Actual home team goals
            away_goals: Actual away team goals

        Returns:
            True if bet was correct, False otherwise
        """
        total_goals = home_goals + away_goals

        if bet.bet_type == "home_win":
            return home_goals > away_goals
        elif bet.bet_type == "away_win":
            return away_goals > home_goals
        elif bet.bet_type == "over_2_5":
            return total_goals > 2.5
        elif bet.bet_type == "under_2_5":
            return total_goals < 2.5

        return False

    def run_backtest(
        self,
        league_name: str,
        data_loader: DataLoader = None
    ) -> BacktestSummary:
        """
        Run backtest on historical data for a specific league.

        This simulates what bets the model would have suggested for past
        matches and checks them against actual results.

        Args:
            league_name: Name of the league to backtest
            data_loader: DataLoader instance (creates new one if None)

        Returns:
            BacktestSummary with accuracy statistics
        """
        loader = data_loader or DataLoader(use_mock=True)

        # Get current standings (used as proxy for historical form)
        standings = loader.get_standings(league_name)
        league_avg = loader.get_league_averages(standings)

        # Get historical results
        historical = loader.get_historical_results(league_name)

        results = []
        market_stats = {
            "home_win": {"total": 0, "correct": 0},
            "away_win": {"total": 0, "correct": 0},
            "over_2_5": {"total": 0, "correct": 0},
            "under_2_5": {"total": 0, "correct": 0}
        }

        for match in historical:
            home_team = match["home"]
            away_team = match["away"]
            actual_home = match["home_goals"]
            actual_away = match["away_goals"]

            # Get team stats
            home_stats = loader.get_team_stats(home_team, standings)
            away_stats = loader.get_team_stats(away_team, standings)

            if home_stats is None or away_stats is None:
                continue  # Skip if team not found in standings

            # Generate prediction
            prediction = self.model.predict_match(
                home_stats, away_stats, league_avg
            )

            # Get sniper bets for this match
            sniper_bets = get_sniper_bets(
                [prediction],
                self.threshold_1x2,
                self.threshold_ou
            )

            # Check each bet
            for bet in sniper_bets:
                was_correct = self.check_bet_outcome(
                    bet, actual_home, actual_away
                )

                results.append(BacktestResult(
                    home_team=home_team,
                    away_team=away_team,
                    predicted_bet=bet.bet_type,
                    predicted_confidence=bet.confidence,
                    actual_home_goals=actual_home,
                    actual_away_goals=actual_away,
                    was_correct=was_correct
                ))

                # Update market stats
                market_stats[bet.bet_type]["total"] += 1
                if was_correct:
                    market_stats[bet.bet_type]["correct"] += 1

        # Calculate summary statistics
        total_bets = len(results)
        correct_bets = sum(1 for r in results if r.was_correct)
        accuracy = correct_bets / total_bets if total_bets > 0 else 0.0

        # Calculate per-market accuracy
        by_market = {}
        for market, stats in market_stats.items():
            if stats["total"] > 0:
                by_market[market] = {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"]
                }

        return BacktestSummary(
            total_bets=total_bets,
            correct_bets=correct_bets,
            accuracy=accuracy,
            by_market=by_market,
            results=results
        )

    def run_all_leagues_backtest(
        self,
        data_loader: DataLoader = None
    ) -> Dict[str, BacktestSummary]:
        """
        Run backtest across all available leagues.

        Args:
            data_loader: DataLoader instance

        Returns:
            Dictionary mapping league name to BacktestSummary
        """
        loader = data_loader or DataLoader(use_mock=True)
        leagues = loader.get_available_leagues()

        all_results = {}
        for league in leagues:
            try:
                summary = self.run_backtest(league.name, loader)
                all_results[league.name] = summary
            except Exception as e:
                print(f"Error backtesting {league.name}: {e}")
                continue

        return all_results

    def get_combined_accuracy(
        self,
        all_results: Dict[str, BacktestSummary]
    ) -> Tuple[float, int, int]:
        """
        Calculate combined accuracy across all leagues.

        Args:
            all_results: Dictionary of BacktestSummary by league

        Returns:
            Tuple of (accuracy, total_bets, correct_bets)
        """
        total_bets = sum(r.total_bets for r in all_results.values())
        correct_bets = sum(r.correct_bets for r in all_results.values())
        accuracy = correct_bets / total_bets if total_bets > 0 else 0.0

        return accuracy, total_bets, correct_bets

    def generate_backtest_report(
        self,
        all_results: Dict[str, BacktestSummary]
    ) -> pd.DataFrame:
        """
        Generate a detailed backtest report as a DataFrame.

        Args:
            all_results: Dictionary of BacktestSummary by league

        Returns:
            DataFrame with backtest statistics
        """
        rows = []

        for league, summary in all_results.items():
            rows.append({
                "League": league,
                "Total Bets": summary.total_bets,
                "Correct": summary.correct_bets,
                "Accuracy": f"{summary.accuracy:.1%}",
                "Home Win Acc": self._market_accuracy(summary, "home_win"),
                "Away Win Acc": self._market_accuracy(summary, "away_win"),
                "Over 2.5 Acc": self._market_accuracy(summary, "over_2_5"),
                "Under 2.5 Acc": self._market_accuracy(summary, "under_2_5"),
            })

        # Add combined row
        accuracy, total, correct = self.get_combined_accuracy(all_results)
        rows.append({
            "League": "COMBINED",
            "Total Bets": total,
            "Correct": correct,
            "Accuracy": f"{accuracy:.1%}",
            "Home Win Acc": "-",
            "Away Win Acc": "-",
            "Over 2.5 Acc": "-",
            "Under 2.5 Acc": "-",
        })

        return pd.DataFrame(rows)

    def _market_accuracy(
        self,
        summary: BacktestSummary,
        market: str
    ) -> str:
        """Helper to format market accuracy."""
        if market in summary.by_market:
            acc = summary.by_market[market]["accuracy"]
            total = summary.by_market[market]["total"]
            return f"{acc:.1%} ({total})"
        return "N/A"
