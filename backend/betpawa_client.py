"""
BetPawa API Client for The Sniper Football Predictor
Fetches live fixtures and odds from betpawa.co.tz
"""

import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from urllib.parse import quote


@dataclass
class BetpawaOdds:
    """Odds for a match"""
    home_win: float
    draw: float
    away_win: float
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None


@dataclass
class BetpawaMatch:
    """Match data from Betpawa"""
    id: str
    home_team: str
    away_team: str
    start_time: str
    competition: str
    odds: Optional[BetpawaOdds] = None


# Competition IDs for major leagues
BETPAWA_LEAGUES = {
    "Premier League": {"id": "11965", "region": "England"},
    "La Liga": {"id": "12039", "region": "Spain"},
    "Serie A": {"id": "12097", "region": "Italy"},
    "Bundesliga": {"id": "12110", "region": "Germany"},
    "Ligue 1": {"id": "12127", "region": "France"},
}

# Market type IDs
MARKET_1X2 = "3743"  # 1X2 Full Time
MARKET_OU = "5000"  # Over/Under Full Time (has multiple handicap lines)
# Handicap mappings: 2=0.5, 6=1.5, 10=2.5, 14=3.5, 18=4.5, 22=5.5
HANDICAP_2_5 = 10  # Handicap value for 2.5 goals line


class BetpawaClient:
    """
    Client for fetching data from Betpawa API.

    This is a free alternative to paid football APIs.
    Data includes fixtures and betting odds.
    """

    BASE_URL = "https://www.betpawa.co.tz/api/sportsbook/v3"

    def __init__(self, brand: str = "betpawa-tanzania", language: str = "en"):
        """
        Initialize the Betpawa client.

        Args:
            brand: Betpawa brand identifier
            language: Language for responses
        """
        self.brand = brand
        self.language = language
        self.session = requests.Session()
        self.session.headers.update({
            "x-pawa-brand": brand,
            "x-pawa-language": language,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def _build_query(
        self,
        competition_id: str,
        event_type: str = "UPCOMING",
        market_types: List[str] = None,
        skip: int = 0,
        take: int = 50
    ) -> str:
        """Build the query parameter for the API."""
        if market_types is None:
            market_types = [MARKET_1X2]

        query = {
            "queries": [{
                "query": {
                    "eventType": event_type,
                    "categories": ["2"],  # Football
                    "zones": {
                        "competitions": [competition_id]
                    },
                    "hasOdds": True
                },
                "view": {
                    "marketTypes": market_types
                },
                "skip": skip,
                "take": take
            }]
        }
        return quote(json.dumps(query, separators=(',', ':')))

    def get_fixtures(
        self,
        league_name: str,
        include_ou: bool = True,
        limit: int = 50
    ) -> List[BetpawaMatch]:
        """
        Get upcoming fixtures for a league.

        Args:
            league_name: Name of the league (e.g., "Premier League")
            include_ou: Include Over/Under 2.5 odds
            limit: Maximum number of fixtures to fetch

        Returns:
            List of BetpawaMatch objects
        """
        if league_name not in BETPAWA_LEAGUES:
            raise ValueError(f"League '{league_name}' not supported. "
                           f"Available: {list(BETPAWA_LEAGUES.keys())}")

        competition_id = BETPAWA_LEAGUES[league_name]["id"]

        # Fetch 1X2 odds
        market_types = [MARKET_1X2]
        if include_ou:
            market_types.append(MARKET_OU)

        query = self._build_query(
            competition_id=competition_id,
            market_types=market_types,
            take=limit
        )

        url = f"{self.BASE_URL}/events/lists/by-queries?q={query}"

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch fixtures: {e}")

        matches = []
        events = data.get("responses", [{}])[0].get("responses", [])

        for event in events:
            match = self._parse_event(event, league_name)
            if match:
                matches.append(match)

        return matches

    def _parse_event(self, event: Dict, league_name: str) -> Optional[BetpawaMatch]:
        """Parse an event from the API response."""
        try:
            participants = event.get("participants", [])
            if len(participants) < 2:
                return None

            home_team = participants[0]["name"]
            away_team = participants[1]["name"]

            # Parse odds from markets
            odds = self._parse_odds(event.get("markets", []))

            return BetpawaMatch(
                id=event.get("id", ""),
                home_team=home_team,
                away_team=away_team,
                start_time=event.get("startTime", ""),
                competition=league_name,
                odds=odds
            )
        except (KeyError, IndexError):
            return None

    def _parse_odds(self, markets: List[Dict]) -> Optional[BetpawaOdds]:
        """Parse odds from market data."""
        home_win = None
        draw = None
        away_win = None
        over_2_5 = None
        under_2_5 = None

        for market in markets:
            market_type_id = market.get("marketType", {}).get("id")
            rows = market.get("row", [])

            if not rows:
                continue

            if market_type_id == MARKET_1X2:
                prices = rows[0].get("prices", [])
                for price in prices:
                    name = price.get("displayName", "")
                    value = price.get("price")
                    if name == "1":
                        home_win = value
                    elif name == "X":
                        draw = value
                    elif name == "2":
                        away_win = value

            elif market_type_id == MARKET_OU:
                # Find the 2.5 goals line (handicap = 10)
                for row in rows:
                    handicap = row.get("handicap")
                    if handicap == HANDICAP_2_5:
                        prices = row.get("prices", [])
                        for price in prices:
                            name = price.get("displayName", "").lower()
                            value = price.get("price")
                            if "over" in name:
                                over_2_5 = value
                            elif "under" in name:
                                under_2_5 = value
                        break

        if home_win and draw and away_win:
            return BetpawaOdds(
                home_win=home_win,
                draw=draw,
                away_win=away_win,
                over_2_5=over_2_5,
                under_2_5=under_2_5
            )
        return None

    def get_all_leagues_fixtures(self, limit_per_league: int = 20) -> Dict[str, List[BetpawaMatch]]:
        """
        Get fixtures for all supported leagues.

        Args:
            limit_per_league: Maximum fixtures per league

        Returns:
            Dictionary mapping league name to list of matches
        """
        all_fixtures = {}

        for league_name in BETPAWA_LEAGUES:
            try:
                fixtures = self.get_fixtures(league_name, limit=limit_per_league)
                all_fixtures[league_name] = fixtures
            except Exception as e:
                print(f"Warning: Failed to fetch {league_name}: {e}")
                all_fixtures[league_name] = []

        return all_fixtures


def odds_to_probability(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Formula: probability = 1 / odds

    Args:
        odds: Decimal odds (e.g., 2.5)

    Returns:
        Implied probability (e.g., 0.4 for 40%)
    """
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def normalize_probabilities(probs: List[float]) -> List[float]:
    """
    Normalize probabilities to sum to 1.0 (remove bookmaker margin).

    Args:
        probs: List of implied probabilities

    Returns:
        Normalized probabilities
    """
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


def get_true_probabilities(odds: BetpawaOdds) -> Dict[str, float]:
    """
    Convert betting odds to true probabilities (margin removed).

    Args:
        odds: BetpawaOdds object

    Returns:
        Dictionary with normalized probabilities
    """
    # 1X2 probabilities
    p_home = odds_to_probability(odds.home_win)
    p_draw = odds_to_probability(odds.draw)
    p_away = odds_to_probability(odds.away_win)

    normalized_1x2 = normalize_probabilities([p_home, p_draw, p_away])

    result = {
        "home_win": normalized_1x2[0],
        "draw": normalized_1x2[1],
        "away_win": normalized_1x2[2],
    }

    # Over/Under probabilities
    if odds.over_2_5 and odds.under_2_5:
        p_over = odds_to_probability(odds.over_2_5)
        p_under = odds_to_probability(odds.under_2_5)
        normalized_ou = normalize_probabilities([p_over, p_under])
        result["over_2_5"] = normalized_ou[0]
        result["under_2_5"] = normalized_ou[1]

    return result
