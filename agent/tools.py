# tools -- houses all of the functions the agent can call to

#get_race_results - year, race

#get_qualifying_results - year, race

#get_fastest_lap - year, race

#get_driver_standings - year


"""
F1 data tools.

These functions retrieve real Formula 1 data from OpenF1.
They are called by agent.py when the model selects a tool.
"""

import re
import unicodedata
from difflib import SequenceMatcher

import requests


# OpenF1's API address
BASE_URL = "https://api.openf1.org/v1"

FIRST_YEAR = 2023
LAST_YEAR = 2026
TIMEOUT_SECONDS = 20


# --------------------------------------------------
# 1. BASIC HELPERS
# --------------------------------------------------

class OpenF1Error(Exception):
    """An error while retrieving or processing F1 data."""
    pass


def _check_year(year):
    """Make sure the requested season is supported."""

    if isinstance(year, bool) or not isinstance(year, int):
        raise OpenF1Error("The year must be an integer.")

    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise OpenF1Error(
            f"Choose a year between {FIRST_YEAR} and {LAST_YEAR}."
        )


def _get_data(endpoint, params=None):
    """Send a request to OpenF1 and return its JSON data."""

    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        raise OpenF1Error(f"Could not reach OpenF1: {error}") from error

    except ValueError as error:
        raise OpenF1Error("OpenF1 returned invalid JSON.") from error

    if not isinstance(data, list):
        raise OpenF1Error("OpenF1 returned an unexpected response.")

    return data


def _run_tool(function, **arguments):
    """Return every tool's result using the same format."""

    try:
        data = function(**arguments)

        return {
            "ok": True,
            "data": data,
            "error": None
        }

    except Exception as error:
        return {
            "ok": False,
            "data": None,
            "error": str(error)
        }


def _normalize_name(name):
    """Make race names easier to compare."""

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(
        character for character in name
        if not unicodedata.combining(character)
    )

    name = name.lower()
    name = re.sub(r"\bformula\s*1\b", " ", name)
    name = re.sub(r"\bgrand prix\b", " ", name)
    name = re.sub(r"\bgp\b", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)

    return " ".join(name.split())


def _find_meeting(year, race):
    """Find the Grand Prix that matches the requested race name."""

    if not isinstance(race, str) or not race.strip():
        raise OpenF1Error("Please provide a race name.")

    meetings = _get_data(
        "meetings",
        {"year": year}
    )

    if not meetings:
        raise OpenF1Error(f"No meetings found for {year}.")

    wanted = _normalize_name(race)
    matches = []

    for meeting in meetings:
        possible_names = [
            meeting.get("meeting_name", ""),
            meeting.get("meeting_official_name", ""),
            meeting.get("circuit_short_name", ""),
            meeting.get("location", ""),
            meeting.get("country_name", "")
        ]

        best_score = 0

        for name in possible_names:
            candidate = _normalize_name(name)

            if not candidate:
                continue

            if candidate == wanted:
                score = 1.0
            elif wanted in candidate.split():
                score = 0.95
            elif wanted in candidate:
                score = 0.85
            else:
                score = SequenceMatcher(
                    None, wanted, candidate
                ).ratio()

            best_score = max(best_score, score)

        matches.append((best_score, meeting))

    matches.sort(key=lambda item: item[0], reverse=True)

    if not matches or matches[0][0] < 0.60:
        available = [
            meeting.get("meeting_name", "Unknown")
            for meeting in meetings
        ]

        raise OpenF1Error(
            f"Couldn't find the race '{race}' in {year}. "
            f"Try a more specific name. Some available races: "
            f"{', '.join(available[:8])}"
        )

    best_score, best_meeting = matches[0]

    # If the best two matches are almost equally good, don't guess.
    if (
        len(matches) > 1
        and best_score < 1.0
        and best_score - matches[1][0] < 0.03
        and best_meeting.get("meeting_key")
        != matches[1][1].get("meeting_key")
    ):
        raise OpenF1Error(
            f"The race name '{race}' is ambiguous. "
            "Please provide a more specific name."
        )

    return best_meeting


def _find_session(year, race, session_name):
    """Find the race or qualifying session for a Grand Prix."""

    meeting = _find_meeting(year, race)

    sessions = _get_data(
        "sessions",
        {
            "meeting_key": meeting["meeting_key"],
            "session_name": session_name
        }
    )

    if not sessions:
        raise OpenF1Error(
            f"No {session_name} session found for "
            f"{meeting.get('meeting_name', race)} in {year}."
        )

    # Usually there is one matching session. If there are multiple,
    # use the one with the latest start time.
    sessions.sort(
        key=lambda session: session.get("date_start", "")
    )

    return meeting, sessions[-1]


def _get_drivers(session_key):
    """Get driver names for a particular session."""

    drivers = _get_data(
        "drivers",
        {"session_key": session_key}
    )

    # Convert the driver list into a dictionary for easy lookup.
    return {
        str(driver["driver_number"]): driver
        for driver in drivers
        if driver.get("driver_number") is not None
    }


def _driver_details(driver_number, drivers):
    """Return useful information about a driver."""

    driver = drivers.get(str(driver_number), {})

    return {
        "driver_number": driver_number,
        "driver": driver.get("full_name", "Unknown driver"),
        "abbreviation": driver.get("name_acronym"),
        "team": driver.get("team_name")
    }


def _get_session_results(year, race, session_name):
    """Retrieve and organize the results for a session."""

    meeting, session = _find_session(
        year, race, session_name
    )

    raw_results = _get_data(
        "session_result",
        {"session_key": session["session_key"]}
    )

    if not raw_results:
        raise OpenF1Error(
            f"No results are available for "
            f"{meeting.get('meeting_name', race)}."
        )

    drivers = _get_drivers(session["session_key"])

    # Put the results in finishing-position order.
    raw_results.sort(
        key=lambda result: (
            result.get("position") is None,
            result.get("position") or 999
        )
    )

    results = []

    for row in raw_results:
        driver_info = _driver_details(
            row.get("driver_number"), drivers
        )

        results.append({
            **driver_info,
            "position": row.get("position"),
            "duration": row.get("duration"),
            "gap_to_leader": row.get("gap_to_leader"),
            "number_of_laps": row.get("number_of_laps"),
            "dnf": row.get("dnf"),
            "dns": row.get("dns"),
            "disqualified": row.get("dsq")
        })

    return {
        "year": year,
        "race": meeting.get("meeting_name", race),
        "session": session_name,
        "results": results
    }


# --------------------------------------------------
# 2. TOOL: RACE RESULTS
# --------------------------------------------------

def _race_results_data(year, race):
    """Get the finishing order for a Grand Prix."""

    return _get_session_results(year, race, "Race")


def get_race_results(year: int, race: str):
    """Public tool: race winner, podium, and finishing order."""

    return _run_tool(
        _race_results_data,
        year=year,
        race=race
    )


# --------------------------------------------------
# 3. TOOL: QUALIFYING RESULTS
# --------------------------------------------------

def _qualifying_results_data(year, race):
    """Get the qualifying classification for a Grand Prix."""

    return _get_session_results(year, race, "Qualifying")


def get_qualifying_results(year: int, race: str):
    """Public tool: pole sitter and qualifying positions."""

    return _run_tool(
        _qualifying_results_data,
        year=year,
        race=race
    )


# --------------------------------------------------
# 4. TOOL: FASTEST LAP
# --------------------------------------------------

def _fastest_lap_data(year, race):
    """Find the quickest recorded lap in the Grand Prix race."""

    meeting, session = _find_session(year, race, "Race")

    laps = _get_data(
        "laps",
        {"session_key": session["session_key"]}
    )

    # Ignore missing lap times and pit-out laps.
    valid_laps = [
        lap for lap in laps
        if lap.get("lap_duration") is not None
        and lap.get("is_pit_out_lap") is not True
    ]

    if not valid_laps:
        raise OpenF1Error(
            f"No usable lap times found for "
            f"{meeting.get('meeting_name', race)} in {year}."
        )

    fastest = min(
        valid_laps,
        key=lambda lap: float(lap["lap_duration"])
    )

    drivers = _get_drivers(session["session_key"])

    return {
        "year": year,
        "race": meeting.get("meeting_name", race),
        "driver": _driver_details(
            fastest.get("driver_number"), drivers
        ),
        "lap_number": fastest.get("lap_number"),
        "lap_time_seconds": fastest.get("lap_duration"),
        "session": "Race"
    }


def get_fastest_lap(year: int, race: str):
    """Public tool: fastest recorded race lap and driver."""

    return _run_tool(
        _fastest_lap_data,
        year=year,
        race=race
    )


# --------------------------------------------------
# 5. TOOL: DRIVER STANDINGS
# --------------------------------------------------

def _driver_standings_data(year):
    """Get the championship standings after the latest race of a season."""

    # Check that the requested year is valid.
    _check_year(year)

    race_sessions = _get_data(
        "sessions",
        {
            "year": year,
            "session_name": "Race"
        }
    )

    if not race_sessions:
        raise OpenF1Error(
            f"No completed race sessions found for {year}."
        )

    # The latest race session gives us the latest standings
    # available for that season.
    race_sessions.sort(
        key=lambda session: session.get("date_start", "")
    )

    latest_race = race_sessions[-1]
    session_key = latest_race["session_key"]

    standings = _get_data(
        "championship_drivers",
        {"session_key": session_key}
    )

    if not standings:
        raise OpenF1Error(
            f"No championship standings available for {year}."
        )

    drivers = _get_drivers(session_key)

    # Sort from first place down.
    standings.sort(
        key=lambda row: (
            row.get("position_current") is None,
            row.get("position_current") or 999
        )
    )

    results = []

    for row in standings:
        driver_info = _driver_details(
            row.get("driver_number"), drivers
        )

        results.append({
            **driver_info,
            "position": row.get("position_current"),
            "points": row.get("points_current"),
            "points_before_latest_race": row.get("points_start")
        })

    return {
        "year": year,
        "standings_as_of_race": latest_race.get("meeting_key"),
        "latest_race_session": latest_race.get("session_name"),
        "standings": results
    }


def get_driver_standings(year: int):
    """Public tool: season-to-date driver championship standings."""

    return _run_tool(
        _driver_standings_data,
        year=year
    )
