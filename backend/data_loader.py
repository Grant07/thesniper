"""
Data Loader for The Sniper Football Predictor
Handles data fetching from API-Football or mock data
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from data.mock_data import LEAGUES, STANDINGS, FIXTURES, HISTORICAL_RESULTS
from backend.predictor import TeamStats


@dataclass
class LeagueInfo:
    """League metadata"""
    name: str
    id: int
    country: str


class DataLoader:
    """
    Data loader that supports both mock data and live API.

    Mock mode is used when no API key is provided or for testing.
    Live mode connects to API-Football (direct or via RapidAPI).
    """

    # Direct API-Football endpoint (api-football.com keys)
    DIRECT_API_URL = "https://v3.football.api-sports.io"

    # RapidAPI endpoint (rapidapi.com keys)
    RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"
    RAPIDAPI_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(self, api_key: Optional[str] = None, use_mock: bool = True, use_rapidapi: bool = False):
        """
        Initialize the data loader.

        Args:
            api_key: API key for API-Football (optional)
            use_mock: If True, use mock data regardless of API key
            use_rapidapi: If True, use RapidAPI endpoint; otherwise use direct API
        """
        self.api_key = api_key
        self.use_mock = use_mock or api_key is None
        self.use_rapidapi = use_rapidapi

        if not self.use_mock and not HAS_REQUESTS:
            print("Warning: requests library not available, falling back to mock mode")
            self.use_mock = True

    def _get_headers(self) -> Dict[str, str]:
        """Get appropriate headers based on API type."""
        if self.use_rapidapi:
            return {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": self.RAPIDAPI_HOST
            }
        else:
            return {
                "x-apisports-key": self.api_key
            }

    def _get_base_url(self) -> str:
        """Get appropriate base URL based on API type."""
        return self.RAPIDAPI_URL if self.use_rapidapi else self.DIRECT_API_URL

    def get_available_leagues(self) -> List[LeagueInfo]:
        """
        Get list of available leagues.

        Returns:
            List of LeagueInfo objects
        """
        return [
            LeagueInfo(name=name, id=info["id"], country=info["country"])
            for name, info in LEAGUES.items()
        ]

    def get_standings(self, league_name: str) -> pd.DataFrame:
        """
        Get current standings for a league.

        Args:
            league_name: Name of the league (e.g., "Premier League")

        Returns:
            DataFrame with team standings
        """
        if self.use_mock:
            return self._get_mock_standings(league_name)
        else:
            return self._get_api_standings(league_name)

    def _get_mock_standings(self, league_name: str) -> pd.DataFrame:
        """Get standings from mock data."""
        if league_name not in STANDINGS:
            raise ValueError(f"League '{league_name}' not found in mock data")

        data = STANDINGS[league_name]
        df = pd.DataFrame(data)

        # Rename columns to be consistent
        df.columns = [col.lower() for col in df.columns]

        return df

    def _get_api_standings(self, league_name: str) -> pd.DataFrame:
        """Get standings from API-Football."""
        if league_name not in LEAGUES:
            raise ValueError(f"League '{league_name}' not supported")

        league_id = LEAGUES[league_name]["id"]

        # Get current season
        response = requests.get(
            f"{self._get_base_url()}/standings",
            headers=self._get_headers(),
            params={"league": league_id, "season": 2024}
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")

        data = response.json()

        # Parse API response into DataFrame
        standings = []
        for team in data["response"][0]["league"]["standings"][0]:
            standings.append({
                "team": team["team"]["name"],
                "mp": team["all"]["played"],
                "gf": team["all"]["goals"]["for"],
                "ga": team["all"]["goals"]["against"],
                "home_gf": team["home"]["goals"]["for"],
                "home_ga": team["home"]["goals"]["against"],
                "away_gf": team["away"]["goals"]["for"],
                "away_ga": team["away"]["goals"]["against"],
            })

        return pd.DataFrame(standings)

    def get_fixtures(self, league_name: str) -> List[Dict[str, str]]:
        """
        Get upcoming fixtures for a league.

        Args:
            league_name: Name of the league

        Returns:
            List of fixtures with home/away teams
        """
        if self.use_mock:
            return self._get_mock_fixtures(league_name)
        else:
            return self._get_api_fixtures(league_name)

    def _get_mock_fixtures(self, league_name: str) -> List[Dict[str, str]]:
        """Get fixtures from mock data."""
        if league_name not in FIXTURES:
            raise ValueError(f"League '{league_name}' not found in mock data")

        return FIXTURES[league_name]

    def _get_api_fixtures(self, league_name: str) -> List[Dict[str, str]]:
        """Get fixtures from API-Football."""
        if league_name not in LEAGUES:
            raise ValueError(f"League '{league_name}' not supported")

        league_id = LEAGUES[league_name]["id"]

        response = requests.get(
            f"{self._get_base_url()}/fixtures",
            headers=self._get_headers(),
            params={
                "league": league_id,
                "season": 2024,
                "next": 10  # Get next 10 fixtures
            }
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")

        data = response.json()

        fixtures = []
        for match in data["response"]:
            fixtures.append({
                "home": match["teams"]["home"]["name"],
                "away": match["teams"]["away"]["name"]
            })

        return fixtures

    def get_historical_results(self, league_name: str) -> List[Dict]:
        """
        Get historical match results for backtesting.

        Args:
            league_name: Name of the league

        Returns:
            List of historical results
        """
        if league_name not in HISTORICAL_RESULTS:
            raise ValueError(f"League '{league_name}' not found in historical data")

        return HISTORICAL_RESULTS[league_name]

    def get_league_averages(self, standings: pd.DataFrame) -> float:
        """
        Calculate league average goals per team per match.

        Formula: Total Goals Scored / Total Matches Played

        Args:
            standings: DataFrame with team standings

        Returns:
            League average goals per team per match
        """
        total_goals = standings["gf"].sum()
        total_matches = standings["mp"].sum()

        if total_matches == 0:
            return 1.5  # Default fallback

        # Each match has 2 teams, so divide by 2 to get per-team average
        return total_goals / total_matches

    def get_team_stats(
        self,
        team_name: str,
        standings: pd.DataFrame
    ) -> Optional[TeamStats]:
        """
        Get statistics for a specific team.

        Args:
            team_name: Name of the team
            standings: DataFrame with standings

        Returns:
            TeamStats object or None if team not found
        """
        team_row = standings[standings["team"] == team_name]

        if team_row.empty:
            return None

        row = team_row.iloc[0]

        return TeamStats(
            name=team_name,
            matches_played=int(row["mp"]),
            goals_for=int(row["gf"]),
            goals_against=int(row["ga"]),
            home_gf=int(row["home_gf"]) if "home_gf" in row else None,
            home_ga=int(row["home_ga"]) if "home_ga" in row else None,
            away_gf=int(row["away_gf"]) if "away_gf" in row else None,
            away_ga=int(row["away_ga"]) if "away_ga" in row else None,
        )

    def prepare_match_data(
        self,
        league_name: str
    ) -> Tuple[List[Dict[str, str]], pd.DataFrame, float]:
        """
        Prepare all data needed for match predictions.

        Args:
            league_name: Name of the league

        Returns:
            Tuple of (fixtures, standings DataFrame, league average)
        """
        standings = self.get_standings(league_name)
        fixtures = self.get_fixtures(league_name)
        league_avg = self.get_league_averages(standings)

        return fixtures, standings, league_avg
