"""
The Sniper - Football Match Predictor
A disciplined betting strategy using cross-market odds analysis

Only suggests bets with 65%+ (1X2) or 70%+ (Over/Under) confidence
with cross-market validation for higher conviction.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
from datetime import datetime

from backend.betpawa_client import BetpawaClient, BetpawaMatch, BETPAWA_LEAGUES
from backend.odds_predictor import (
    SniperBet,
    OddsPrediction,
    match_to_prediction,
    get_sniper_bets_from_odds,
    format_datetime
)
from backend.smart_predictor import (
    SmartBet,
    MatchProfile,
    get_smart_bets,
    analyze_match_profile
)
from backend.betpawa_client import get_true_probabilities

# For backtesting with mock data
from backend.predictor import PoissonModel, get_sniper_bets
from backend.data_loader import DataLoader
from backend.backtester import Backtester


# Page configuration
st.set_page_config(
    page_title="The Sniper - Football Predictor",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling (works in both light and dark mode)
st.markdown("""
<style>
    .match-title {
        font-size: 1.4em;
        font-weight: 700;
        color: #1a1a2e !important;
        margin-bottom: 8px;
    }
    .bet-label {
        font-size: 1.1em;
        color: #0066cc !important;
        font-weight: 600;
        margin: 12px 0;
    }
    .confidence-display {
        font-size: 2.5em;
        font-weight: 800;
        text-align: center;
        padding: 10px;
    }
    .confidence-high { color: #00aa55 !important; }
    .confidence-medium { color: #cc8800 !important; }
    .confidence-low { color: #cc4444 !important; }
    .odds-display {
        font-size: 1.8em;
        font-weight: 700;
        color: #0066cc !important;
        text-align: center;
    }
    .profile-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 4px 0;
    }
    .profile-dominant { background: #00cc66 !important; color: #000 !important; }
    .profile-tight { background: #ffcc00 !important; color: #000 !important; }
    .profile-open { background: #ff6666 !important; color: #fff !important; }
    .profile-defensive { background: #9966ff !important; color: #fff !important; }
    .market-agree {
        color: #00aa55 !important;
        font-weight: 600;
    }
    .summary-stat {
        text-align: center;
        padding: 20px;
        background: rgba(0,102,204,0.1) !important;
        border-radius: 12px;
        border: 1px solid rgba(0,102,204,0.2);
    }
    .summary-number {
        font-size: 2.5em;
        font-weight: 800;
        color: #0066cc !important;
    }
    .summary-label {
        color: #666 !important;
        font-size: 0.9em;
    }
    .match-meta {
        color: #666 !important;
        font-size: 0.85em;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .match-title { color: #ffffff !important; }
        .bet-label { color: #00d4ff !important; }
        .odds-display { color: #00d4ff !important; }
        .summary-number { color: #00d4ff !important; }
        .summary-label { color: #aaa !important; }
        .match-meta { color: #aaa !important; }
        .confidence-high { color: #00ff88 !important; }
        .confidence-medium { color: #ffcc00 !important; }
        .summary-stat {
            background: rgba(0,212,255,0.1) !important;
            border-color: rgba(0,212,255,0.2) !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Profile styling
PROFILE_STYLES = {
    MatchProfile.DOMINANT_HOME: {"icon": "🔥", "label": "Dominant Home", "class": "profile-dominant"},
    MatchProfile.TIGHT_HOME: {"icon": "🛡️", "label": "Tight Home Win", "class": "profile-tight"},
    MatchProfile.DOMINANT_AWAY: {"icon": "⚡", "label": "Dominant Away", "class": "profile-dominant"},
    MatchProfile.TIGHT_AWAY: {"icon": "🔒", "label": "Tight Away Win", "class": "profile-tight"},
    MatchProfile.OPEN_GAME: {"icon": "⚽", "label": "Open Game", "class": "profile-open"},
    MatchProfile.DEFENSIVE_BATTLE: {"icon": "🧱", "label": "Defensive Battle", "class": "profile-defensive"},
    MatchProfile.UNCERTAIN: {"icon": "❓", "label": "Uncertain", "class": ""},
}

BET_TYPE_ICONS = {
    "home_win": "🏠",
    "away_win": "✈️",
    "over_2_5": "⬆️",
    "under_2_5": "⬇️",
}


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.75:
        return "confidence-high"
    elif confidence >= 0.65:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_card_class(confidence: float) -> str:
    """Get card CSS class based on confidence."""
    if confidence >= 0.75:
        return "bet-card bet-card-high"
    else:
        return "bet-card bet-card-medium"


def display_smart_bet_card(bet: SmartBet, index: int):
    """Display a smart bet as a styled card."""
    profile_style = PROFILE_STYLES.get(bet.match_profile, PROFILE_STYLES[MatchProfile.UNCERTAIN])
    bet_icon = BET_TYPE_ICONS.get(bet.bet_type, "🎯")
    conf_class = get_confidence_class(bet.confidence)

    col1, col2, col3 = st.columns([3, 1.5, 1])

    with col1:
        # Match info with proper styling
        st.markdown(f"**#{index}** • {bet.competition} • 📅 {format_datetime(bet.start_time)}")
        st.markdown(f"### 🏟️ {bet.home_team} vs {bet.away_team}")

        # Bet recommendation
        st.markdown(f"**{bet_icon} {bet.label}**")

        # Profile badge
        st.markdown(f"""
        <span class="profile-badge {profile_style['class']}">
            {profile_style['icon']} {profile_style['label']}
        </span>
        """, unsafe_allow_html=True)

    with col2:
        # Confidence meter using native Streamlit
        conf_color = "🟢" if bet.confidence >= 0.75 else "🟡"
        st.metric(
            label="Confidence",
            value=f"{conf_color} {bet.confidence:.0%}"
        )
        st.progress(bet.confidence)

    with col3:
        # Odds using native Streamlit
        st.metric(
            label="Odds",
            value=f"💰 {bet.odds:.2f}"
        )

    # Analysis signals in expander
    with st.expander("📊 View Analysis", expanded=False):
        st.markdown("**Why this bet?**")
        for signal in bet.signals:
            if "agree" in signal.lower():
                st.markdown(f"✅ <span class='market-agree'>{signal}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"• {signal}")

        # Show probability breakdown
        st.markdown("---")
        st.markdown("**Market Probabilities:**")

        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            if bet.bet_type in ["home_win", "away_win"]:
                st.metric("Win Probability", f"{bet.confidence:.1%}")
        with prob_col2:
            st.metric("Profile Confidence", f"{bet.profile_confidence:.1%}")

    st.markdown("---")


def display_summary_dashboard(bets: List[SmartBet], total_matches: int):
    """Display summary statistics at the top."""

    # Count by type
    home_wins = sum(1 for b in bets if b.bet_type == "home_win")
    overs = sum(1 for b in bets if b.bet_type == "over_2_5")
    unders = sum(1 for b in bets if b.bet_type == "under_2_5")

    # Average confidence
    avg_conf = sum(b.confidence for b in bets) / len(bets) if bets else 0

    # High confidence count
    high_conf = sum(1 for b in bets if b.confidence >= 0.75)

    # Selection rate
    selection_rate = len(bets) / total_matches * 100 if total_matches else 0

    # Use native Streamlit metrics for better theme support
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Analyzed", total_matches)
    with col2:
        st.metric("🎯 Sniper Bets", len(bets))
    with col3:
        st.metric("🟢 High Conf", high_conf)
    with col4:
        st.metric("📈 Avg Conf", f"{avg_conf:.0%}")
    with col5:
        st.metric("🎚️ Select Rate", f"{selection_rate:.0f}%")

    # Bet type breakdown
    st.markdown("")
    type_col1, type_col2, type_col3 = st.columns(3)
    with type_col1:
        st.metric("🏠 Home Wins", home_wins)
    with type_col2:
        st.metric("⬆️ Over 2.5", overs)
    with type_col3:
        st.metric("⬇️ Under 2.5", unders)


def display_strategy_explainer():
    """Display the strategy explanation."""
    with st.expander("🎯 How The Sniper Strategy Works", expanded=False):
        st.markdown("""
        ### The Sniper uses **Cross-Market Validation** to find high-conviction bets

        **Step 1: Convert Odds → True Probabilities**
        - Remove bookmaker margin (~5%) to get actual probabilities

        **Step 2: Apply Confidence Thresholds**
        - 1X2 (Home/Away Win): Must be **≥65%**
        - Over/Under 2.5: Must be **≥70%**

        **Step 3: Cross-Market Validation** ⭐
        - Check if 1X2 and O/U markets **agree** on match pattern
        - Only recommend bets where markets align

        ---

        ### Match Profiles

        | Profile | 1X2 Signal | O/U Signal | Meaning |
        |---------|------------|------------|---------|
        | 🔥 **Dominant Home** | Strong home favorite | Over 2.5 likely | Home team dominates and scores freely |
        | 🛡️ **Tight Home Win** | Strong home favorite | Under 2.5 likely | Home team wins but controls the game |
        | ⚽ **Open Game** | Even match | Over 2.5 likely | Both teams will attack, goals expected |
        | 🧱 **Defensive Battle** | Even match | Under 2.5 likely | Tight, cagey affair |

        > 🏠 **Home Advantage Mode**: We only bet on home wins (away wins skipped).
        > *Based on vibes, not science.*

        ---

        ### Why This Works

        When **both markets agree**, it means:
        - Bookmakers have consistent view of the match
        - Higher conviction in the predicted outcome
        - Reduces "surprise" results

        **Example:** Bayern Munich
        - 1X2: 83% home win → Strong favorite
        - O/U: 75% over 2.5 → Goals expected
        - Profile: **Dominant Home** ✅ Markets agree!
        """)


def display_all_matches_table(matches: List[BetpawaMatch], league_name: str):
    """Display all matches in a clean table format."""
    with st.expander(f"📋 View All {league_name} Matches ({len(matches)})", expanded=False):
        data = []
        for match in matches:
            pred = match_to_prediction(match)
            if pred:
                probs = get_true_probabilities(match.odds)
                profile, conf, _ = analyze_match_profile(probs)
                profile_style = PROFILE_STYLES.get(profile, PROFILE_STYLES[MatchProfile.UNCERTAIN])

                row = {
                    "Match": f"{pred.home_team} vs {pred.away_team}",
                    "Date": format_datetime(pred.start_time),
                    "Home": f"{pred.home_win_prob:.0%}",
                    "Draw": f"{pred.draw_prob:.0%}",
                    "Away": f"{pred.away_win_prob:.0%}",
                    "Over 2.5": f"{pred.over_2_5_prob:.0%}" if pred.over_2_5_prob else "-",
                    "Under 2.5": f"{pred.under_2_5_prob:.0%}" if pred.under_2_5_prob else "-",
                    "Profile": f"{profile_style['icon']} {profile_style['label']}",
                }
                data.append(row)

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main application entry point."""

    # Header using native Streamlit
    st.title("🎯 The Sniper")
    st.caption("High-Confidence Football Predictions with Cross-Market Validation")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Data source
    data_source = st.sidebar.radio(
        "Data Source",
        ["🔴 Live Data", "📁 Mock Data"],
        index=0,
        help="Live odds data or mock data for testing"
    )
    use_live = "Live" in data_source

    # League selection
    league_options = ["All Leagues"] + list(BETPAWA_LEAGUES.keys())
    selected_league = st.sidebar.selectbox(
        "Select League",
        league_options,
        index=0
    )

    # Thresholds
    st.sidebar.subheader("Confidence Thresholds")
    threshold_1x2 = st.sidebar.slider(
        "1X2 Minimum",
        min_value=0.50,
        max_value=0.80,
        value=0.65,
        step=0.05,
        format="%.0f%%",
        help="Minimum confidence for Home/Away win bets"
    )

    threshold_ou = st.sidebar.slider(
        "Over/Under Minimum",
        min_value=0.50,
        max_value=0.85,
        value=0.70,
        step=0.05,
        format="%.0f%%",
        help="Minimum confidence for Over/Under 2.5 bets"
    )

    # Cross-market validation toggle
    use_smart = st.sidebar.checkbox(
        "Cross-Market Validation",
        value=True,
        help="Only show bets where 1X2 and O/U markets agree on match pattern"
    )

    # Strategy explainer
    display_strategy_explainer()

    st.divider()

    if use_live:
        st.success("📡 **Live Data** - Real-time odds")

        try:
            client = BetpawaClient()
            all_matches = []
            all_bets = []
            matches_by_league = {}

            # Determine which leagues to fetch
            leagues_to_fetch = [selected_league] if selected_league != "All Leagues" else list(BETPAWA_LEAGUES.keys())

            with st.spinner(f"Fetching fixtures..."):
                for league in leagues_to_fetch:
                    try:
                        matches = client.get_fixtures(league, include_ou=True, limit=30)
                        matches_by_league[league] = matches
                        all_matches.extend(matches)

                        if use_smart:
                            bets = get_smart_bets(matches, threshold_1x2, threshold_ou, require_profile_match=True)
                        else:
                            # Convert simple bets to smart format for consistent display
                            simple_bets = get_sniper_bets_from_odds(matches, threshold_1x2, threshold_ou)
                            bets = []
                            for sb in simple_bets:
                                # Find match to get profile
                                for m in matches:
                                    if m.home_team == sb.home_team and m.away_team == sb.away_team:
                                        probs = get_true_probabilities(m.odds)
                                        profile, conf, signals = analyze_match_profile(probs)
                                        bets.append(SmartBet(
                                            home_team=sb.home_team,
                                            away_team=sb.away_team,
                                            start_time=sb.start_time,
                                            competition=sb.competition,
                                            bet_type=sb.bet_type,
                                            confidence=sb.confidence,
                                            odds=sb.odds,
                                            label=sb.label,
                                            match_profile=profile,
                                            profile_confidence=conf,
                                            signals=signals,
                                            edge=0,
                                            kelly_fraction=0
                                        ))
                                        break

                        all_bets.extend(bets)
                    except Exception as e:
                        st.warning(f"Could not fetch {league}: {e}")

            if not all_matches:
                st.warning("No matches found. Try again later.")
                return

            # Sort bets by match time (next match first)
            all_bets.sort(key=lambda x: x.start_time)

            # Summary dashboard
            display_summary_dashboard(all_bets, len(all_matches))

            st.divider()

            # Display bets
            if all_bets:
                st.markdown("## 🎯 Sniper Bets")
                st.caption("Sorted by kick-off time (next match first)")

                for i, bet in enumerate(all_bets, 1):
                    display_smart_bet_card(bet, i)
            else:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("## 💰 No Safe Bets Right Now")
                    st.info(
                        f"The Sniper strategy requires discipline. When no matches meet our "
                        f"confidence thresholds ({threshold_1x2:.0%} for 1X2, {threshold_ou:.0%} for O/U), "
                        f"we wait for better opportunities.\n\n"
                        f"**Save your money. The best bet is sometimes no bet.**"
                    )

            # All matches tables
            st.divider()
            st.markdown("### 📊 All Analyzed Matches")

            for league, matches in matches_by_league.items():
                if matches:
                    display_all_matches_table(matches, league)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Try switching to Mock Data in the sidebar")

    else:
        # Mock data mode
        st.info("📁 **Mock Data Mode** - Using simulated data for testing")

        try:
            data_loader = DataLoader(use_mock=True)
            model = PoissonModel(use_home_away_split=True)

            leagues_to_use = [selected_league] if selected_league != "All Leagues" else list(BETPAWA_LEAGUES.keys())

            all_predictions = []
            all_bets = []

            for league in leagues_to_use:
                try:
                    fixtures, standings, league_avg = data_loader.prepare_match_data(league)

                    for fixture in fixtures:
                        home_stats = data_loader.get_team_stats(fixture["home"], standings)
                        away_stats = data_loader.get_team_stats(fixture["away"], standings)

                        if home_stats and away_stats:
                            pred = model.predict_match(home_stats, away_stats, league_avg)
                            pred.competition = league
                            all_predictions.append(pred)

                    bets = get_sniper_bets(all_predictions, threshold_1x2, threshold_ou)
                    all_bets.extend(bets)
                except Exception as e:
                    st.warning(f"Error with {league}: {e}")

            # Display summary
            st.markdown(f"""
            <div class="summary-stat" style="display: inline-block; margin: 10px;">
                <div class="summary-number">{len(all_predictions)}</div>
                <div class="summary-label">Matches Analyzed</div>
            </div>
            <div class="summary-stat" style="display: inline-block; margin: 10px;">
                <div class="summary-number">{len(all_bets)}</div>
                <div class="summary-label">Sniper Bets</div>
            </div>
            """, unsafe_allow_html=True)

            if all_bets:
                st.success(f"Found **{len(all_bets)}** high-confidence bets!")

                for bet in all_bets:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {bet.home_team} vs {bet.away_team}")
                        st.markdown(f"**{bet.label}**")
                    with col2:
                        st.metric("Confidence", f"{bet.confidence:.0%}")
                    st.progress(bet.confidence)
                    st.divider()
            else:
                st.warning("No high-confidence bets found in mock data.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Footer
    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Gambling involves risk. Past performance does not guarantee future results. "
        "Always bet responsibly."
    )
    st.caption("🎯 The Sniper v2.0 | Cross-Market Validation Strategy | 🏠 Home Advantage Mode")


if __name__ == "__main__":
    main()
