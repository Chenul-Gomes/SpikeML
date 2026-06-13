"""
scraper.py — SpikeML Data Collection
=====================================
Scrapes pro match data from VLR.gg including:
  - Match metadata (teams, scores, dates, tournaments)
  - Per-map player stats (ACS, kills, deaths, KAST, ADR, etc.)
"""

import time
import re

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.vlr.gg"
SLEEP_TIME = 1
NUM_PAGES = 5

def scrape_match(match_url):
    """
    Scrape player stats from a VLR match page.

    Args:
        match_url (str): URL of the match page to scrape

    Returns:
        player_stats (list): List of player stats dictionaries
    """
    response = requests.get(match_url)
    time.sleep(SLEEP_TIME)
    soup = BeautifulSoup(response.text, "html.parser")

    # grab all stats tables — first two are "All Maps" summaries, skip them
    tables = soup.select(".wf-table-inset.mod-overview")

    # extract map names from the nav tabs, stripping leading numbers
    map_nav = soup.select(".vm-stats-gamesnav-item")
    map_names = [
        re.sub(r"^\d+", "", m.get_text(strip=True)).strip()
        for m in map_nav
        if "all" not in m.get_text(strip=True).lower()
    ]

    player_stats = []

    for i, table in enumerate(tables[2:]):
        # tables come in pairs per map — even index = team1, odd = team2
        team = "team1" if i % 2 == 0 else "team2"
        rows = table.select("tr")
        map_name = map_names[i // 2] if i // 2 < len(map_names) else None

        for row in rows:
            name = row.select_one(".mod-player .text-of")

            if not name:
                continue
            
            # stat columns are ordered: rating, acs, kills, deaths,
            # assists, kd_diff, kast, adr, hs, fk, fd
            stats = row.select("td.mod-stat")
            rating = stats[0].select_one(".side.mod-side.mod-both")
            acs = stats[1].select_one(".side.mod-side.mod-both")
            kills = stats[2].select_one(".side.mod-side.mod-both")
            deaths = stats[3].select_one(".side.mod-both")
            assists = stats[4].select_one(".side.mod-both")
            kd_diff = stats[5].select_one(".side.mod-both")
            kast = stats[6].select_one(".side.mod-both")
            adr = stats[7].select_one(".side.mod-both")
            hs = stats[8].select_one(".side.mod-both")
            fk = stats[9].select_one(".side.mod-both")
            fd = stats[10].select_one(".side.mod-both")

            player_stats.append({
                "match_url": match_url,
                "team": team,
                "name": name.get_text(strip=True),
                "rating": rating.get_text(strip=True) if rating else None,
                "acs": acs.get_text(strip=True) if acs else None,
                "kills": kills.get_text(strip=True) if kills else None,
                "deaths": deaths.get_text(strip=True) if deaths else None,
                "assists": assists.get_text(strip=True) if assists else None,
                "kd_diff": kd_diff.get_text(strip=True) if kd_diff else None,
                "kast": kast.get_text(strip=True) if kast else None,
                "adr": adr.get_text(strip=True) if adr else None,
                "hs": hs.get_text(strip=True) if hs else None,
                "fk": fk.get_text(strip=True) if fk else None,
                "fd": fd.get_text(strip=True) if fd else None,
                "map": map_name
            })
    return player_stats

def scrape_match_index(num_pages=NUM_PAGES):
    """
    Scrape match metadata from the VLR match index pages.

    Args:
        num_pages (int): Number of index pages to scrape

    Returns:
        all_matches (list): List of match metadata dictionaries
    """
    all_matches = []

    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/matches/results?page={page}"
        response = requests.get(url)
        time.sleep(SLEEP_TIME)
        soup = BeautifulSoup(response.text, "html.parser")

        # select both date headers and match items in page order
        current_date = None
        elements = soup.select(".wf-label.mod-large, a.wf-module-item")

        for element in elements:
            if "wf-label" in element.get("class", []):
                # strip "Today"/"Yesterday" span tags before extracting date
                for tag in element.select("span"):
                    tag.decompose()
                current_date = element.get_text(strip=True)
            else:
                teams = element.select(".match-item-vs-team-name .text-of")
                scores = element.select(".match-item-vs-team-score")
                tournament = element.select(".match-item-event .text-of")
                match_url = element["href"]

                all_matches.append({
                    "date": current_date,
                    "team1": teams[0].get_text(strip=True) if len(teams) > 0 else None,
                    "team2": teams[1].get_text(strip=True) if len(teams) > 1 else None,
                    "score1": scores[0].get_text(strip=True) if len(scores) > 0 else None,
                    "score2": scores[1].get_text(strip=True) if len(scores) > 1 else None,
                    "tournament": tournament[0].get_text(strip=True) if len(tournament) > 0 else None,
                    "url": match_url
                })      
    return all_matches

def scrape_all_player_stats(match_urls):
    """
    Scrape player stats for all matches in the given list of match URLs.

    Args:
        match_urls (list): List of match URLs to scrape

    Returns:
        all_player_stats (list): List of player stats dictionaries for all matches
    """
    all_player_stats = []

    for i, url in enumerate(match_urls):
        print(f"Scraping match {i + 1}/{len(match_urls)}: {url}")
        player_stats = scrape_match(url)
        all_player_stats.extend(player_stats)
    
    return all_player_stats