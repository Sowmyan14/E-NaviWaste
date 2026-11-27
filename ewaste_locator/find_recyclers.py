"""Find the NYC electronics drop-off sites within 5 miles of a ZIP code.

Run:     python find_recyclers.py            (asks for a ZIP code)
         python find_recyclers.py 11220
Import:  from ewaste_locator.find_recyclers import find_recyclers

Distances are straight-line miles from the center of the ZIP code, not driving miles.
"""

import csv
import re
import sys
from functools import lru_cache
from urllib.parse import urlparse

import pgeocode
from geopy.distance import geodesic

# ewaste_locator is the name of both this folder and the download script, so the import depends
# on whether this file is imported from the project root or run directly.
if __package__:
    from .ewaste_locator import CSV_FILE
else:
    from ewaste_locator import CSV_FILE

RADIUS_MILES = 5


class LocatorError(Exception):
    """A problem to show to the user, such as an invalid ZIP code."""


# --- ZIP code -> coordinates ---------------------------------------------------------------


def normalize_zip(text):
    """Return the 5-digit ZIP code from "11220" or "11220-1234"."""
    match = re.fullmatch(r"([0-9]{5})(-[0-9]{4})?", text.strip())
    if not match:
        raise LocatorError("Please enter a 5-digit US ZIP code, such as 11220.")
    return match.group(1)


@lru_cache(maxsize=1)
def zip_database():
    """Load the US ZIP code table. Downloaded once, then read from a local cache."""
    return pgeocode.Nominatim("us")


def locate_zip(zip_code):
    """Return (latitude, longitude, "City, ST") for the center of a ZIP code."""
    zip_code = normalize_zip(zip_code)
    try:
        row = zip_database().query_postal_code(zip_code)
    except OSError as exc:  # includes network errors during the first download
        raise LocatorError(f"The ZIP code data could not be downloaded: {exc}") from exc
    if row.latitude != row.latitude:  # NaN: the ZIP code is not in the table
        raise LocatorError(f"ZIP code {zip_code} was not found.")
    return float(row.latitude), float(row.longitude), f"{row.place_name}, {row.state_code}"


# --- facilities ----------------------------------------------------------------------------


def clean(value):
    """Collapse whitespace, and turn placeholders into an empty string."""
    text = " ".join((value or "").split())
    return "" if text.lower() in {"n/a", "na", "none"} else text


def clean_url(value):
    """Return a full URL, or "" when the value is not a web address."""
    text = clean(value)
    if "." not in text or " " in text:
        return ""
    return text if re.match(r"https?://", text, re.IGNORECASE) else f"http://{text}"


def domain(url):
    return (urlparse(url).hostname or "").removeprefix("www.")


def read_facility(row):
    return {
        "name": clean(row["name"]),
        "address": clean(row["address"]).removesuffix(", United States"),
        "phone": clean(row["phone"]),
        "email": clean(row["email"]),
        "url": clean_url(row["website"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }


def merge_duplicates(facilities):
    """Drop repeat listings of one site, keeping the one with the most contact details.

    Both the location and the website must match: different companies can share a building.
    """
    merged = {}
    for index, facility in enumerate(facilities):
        site = domain(facility["url"])
        key = (
            (round(facility["latitude"], 4), round(facility["longitude"], 4), site)
            if site
            else index  # no website to compare, so never merged
        )
        kept = merged.get(key)
        if kept is None or detail_count(facility) > detail_count(kept):
            merged[key] = facility
    return list(merged.values())


def detail_count(facility):
    return sum(bool(facility[field]) for field in ("phone", "email", "url"))


def load_facilities():
    """Read the NYC drop-off CSV. It is never downloaded from here: see ewaste_locator.py."""
    if not CSV_FILE.exists():
        raise LocatorError(
            "The NYC drop-off list is missing. Run python ewaste_locator/ewaste_locator.py first."
        )
    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        return merge_duplicates([read_facility(row) for row in csv.DictReader(file)])


# --- search --------------------------------------------------------------------------------


def find_recyclers(zip_code, radius_miles=RADIUS_MILES):
    """Return the facilities within ``radius_miles`` of a ZIP code, nearest first."""
    latitude, longitude, _ = locate_zip(zip_code)
    nearby = []
    for facility in load_facilities():
        miles = geodesic(
            (latitude, longitude), (facility["latitude"], facility["longitude"])
        ).miles
        if miles <= radius_miles:
            nearby.append((miles, facility))
    nearby.sort(key=lambda pair: pair[0])
    return [{**facility, "distance_mi": round(miles, 1)} for miles, facility in nearby]


# --- command line --------------------------------------------------------------------------


def print_results(zip_code):
    _, _, place = locate_zip(zip_code)
    results = find_recyclers(zip_code)
    where = f"within {RADIUS_MILES} miles of {normalize_zip(zip_code)} ({place})"

    if not results:
        print(f"\nNo NYC electronics drop-off sites found {where}.")
        return

    noun = "recycler" if len(results) == 1 else "recyclers"
    print(f"\n{len(results)} e-waste {noun} {where}:\n")
    for number, facility in enumerate(results, 1):
        print(f"{number}. {facility['name']} - {facility['distance_mi']} mi")
        for field in ("address", "phone", "email", "url"):
            if facility[field]:
                print(f"   {facility[field]}")
        print()


def main():
    try:
        zip_code = sys.argv[1] if len(sys.argv) > 1 else input("Enter your ZIP code: ")
        print_results(zip_code)
    except LocatorError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
