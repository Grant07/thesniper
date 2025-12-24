---

# INSTRUCTIONS.md: The "Sniper" Football Predictor

## 1. Project Overview & Philosophy

**Goal:** Build a Python-based web application using **Streamlit** that predicts Football (Soccer) match outcomes.
**Core Strategy:** "The Sniper." We do not predict every game. We strictly filter for matches where the mathematical confidence exceeds **65-70%**.
**Target Markets:**

1. **1x2** (Home Win / Draw / Away Win)
2. **Over/Under 2.5 Goals**
**Success Metric:** Accuracy > 60% on suggested bets.

---

## 2. Tech Stack Requirements

* **Language:** Python 3.9+
* **Frontend/App Framework:** Streamlit (`streamlit`)
* **Data Manipulation:** Pandas (`pandas`)
* **Math/Stats:** Scipy (`scipy.stats` for Poisson distribution)
* **API Connection:** Requests (`requests`)
* **Data Source:** **API-Football (via RapidAPI)**. *Note to Agent: Use mock data structure if API key is not provided, but structure code to easily swap in the live API.*

---

## 3. Step-by-Step Implementation Instructions

### Step 1: Project Structure Setup

Create a project folder with the following file structure:

* `main.py` (The Streamlit dashboard)
* `backend/`
* `data_loader.py` (Handles API fetching and cleaning)
* `predictor.py` (Contains the Math/Poisson logic)


* `requirements.txt` (List dependencies: streamlit, pandas, scipy, requests)

### Step 2: Data Acquisition Module (`data_loader.py`)

**Action:** Create a class `DataLoader` that fetches two specific datasets.

1. **League Standings / Form Data:**
* Fetch the current standings for a specific league (e.g., Premier League).
* **Crucial Data Points Needed:**
* Matches Played (MP)
* Goals For (GF)
* Goals Against (GA)
* *Research Note:* If possible, separate Home GF/GA and Away GF/GA for higher accuracy. If not, total GF/GA is acceptable for MVP.




2. **Upcoming Fixtures:**
* Fetch the next round of matches (Home Team vs. Away Team).



**Action:** Implement a `get_league_averages()` function.

* Calculate the **League Average Goals per Game**. Formula: `(Total League Goals Scored) / (Total League Matches Played)`.

### Step 3: The Mathematical Engine (`predictor.py`)

**Action:** Create a class `PoissonModel`. This is the core logic.

1. **Calculate Team Strengths:**
* **Attack Strength:** `(Team Goals Scored / Team Matches)` divided by `League Average Goals`.
* **Defense Strength:** `(Team Goals Conceded / Team Matches)` divided by `League Average Goals`.


2. **Calculate Expected Goals (Lambda) for a Match:**
* `Home_Exp_Goals` = `Home_Attack` * `Away_Defense` * `League_Avg_Goals`.
* `Away_Exp_Goals` = `Away_Attack` * `Home_Defense` * `League_Avg_Goals`.


3. **Simulate Probabilities (The Matrix):**
* Use `scipy.stats.poisson`.
* Iterate through a score matrix from 0-0 to 5-5.
* Calculate the probability of *each* specific scoreline.


4. **Derive Market Probabilities:**
* **Home Win %**: Sum of probabilities where Home Score > Away Score.
* **Away Win %**: Sum of probabilities where Away Score > Home Score.
* **Under 2.5 %**: Sum of probabilities where (Home + Away Goals) < 2.5.
* **Over 2.5 %**: 1 - Under 2.5 %.



### Step 4: The "Sniper" Logic (Filtering)

**Action:** Do NOT return every prediction. Implement a filter function `get_sniper_bets(fixtures_list)`.

* **Logic:** Iterate through all upcoming fixtures.
* **Condition 1 (1x2):** If `Home Win %` > 0.65 (65%), flag as "Strong Home Win".
* **Condition 2 (1x2):** If `Away Win %` > 0.65 (65%), flag as "Strong Away Win".
* **Condition 3 (O/U):** If `Over 2.5 %` > 0.70 (70%), flag as "High Confidence Over".
* **Condition 4 (O/U):** If `Under 2.5 %` > 0.70 (70%), flag as "High Confidence Under".
* **Output:** Return *only* the matches that meet these criteria.

### Step 5: The Frontend (`main.py`)

**Action:** Build a clean, "Glance-able" dashboard.

1. **Sidebar:**
* Dropdown to select League (e.g., Premier League, La Liga).
* Input field for API Key (masked).


2. **Main Area:**
* **Title:** "70% Sniper Bot 🎯"
* **Section:** "This Weekend's Sure Bets"
* **Display Logic:**
* Load the sniper bets from Step 4.
* Display them in cards or a table.
* **Visuals:** Use a Green progress bar to show the probability (e.g., "Arsenal Win: 72%").
* If no games meet the criteria, display: *"No safe bets this weekend. Save your money."* (This reinforces the discipline of the strategy).





---

## 4. Execution Plan for the Agent

1. Write `requirements.txt` first.
2. Write `backend/predictor.py` and verify the math logic with dummy data (e.g., Arsenal vs. Burnley stats).
3. Write `backend/data_loader.py` with a "Mock Mode" (hardcoded stats) so the UI can be tested without an API key initially.
4. Write `main.py` to display the filtered results.

**Constraint Checklist:**

* [ ] Did you use Poisson Distribution?
* [ ] Did you implement the 65%/70% filters?
* [ ] Is the code modular (separation of logic and UI)?
* [ ] Does the UI strictly hide "low confidence" bets?
