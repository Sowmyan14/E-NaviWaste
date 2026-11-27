import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import requests

# NYC Open Data: "Electronics Drop Off Locations in NYC", published by the Department of
# Sanitation (DSNY). Every site on it takes electronics from NYC residents.
API_URL = "https://data.cityofnewyork.us/api/v3/views/wshr-5vic/query.json"

DATA_FOLDER = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_FOLDER / "nyc_dropoffs.csv"


def api_request():
    """Return every row of the NYC drop-off dataset."""

    response = requests.get(API_URL, timeout=45)

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, list):  # errors come back as an object
        raise RuntimeError(result.get("message") or result)

    return result


# --- duplicates ---------------------------------------------------------------------------
# A dataset can list a site twice, often under two names. Two rows are the same site only when
# the location, the street number and the company all match: different companies can share a
# building, and one company can have several buildings on a street.


def text(value):
    """Return the value as a string without extra spaces, or "" for a placeholder."""

    cleaned = " ".join(str(value or "").split())

    return "" if cleaned.lower() in {"n/a", "na", "none"} else cleaned


def website_domain(facility):
    """Return the website's domain, or "" if there is no usable website."""

    url = text(facility["website"])

    if "." not in url or " " in url:
        return ""

    if "://" not in url:
        url = f"http://{url}"

    try:
        return (urlparse(url).hostname or "").removeprefix("www.")
    except ValueError:  # malformed, such as "http://[abc.com"
        return ""


def site_key(facility):
    """Return the location and street number, or None if the address has no number."""

    number = re.search(r"\d+", text(facility["address"]))

    if number is None:
        return None

    return (
        round(float(facility["latitude"]), 4),
        round(float(facility["longitude"]), 4),
        number.group()
    )


def company_words(facility):
    """Return the words of the company name, without endings such as Inc or LLC."""

    words = re.findall(r"[a-z0-9]+", text(facility["name"]).lower())
    endings = {"inc", "llc", "llp", "ltd", "co", "corp", "corporation", "the", "dba"}

    return [word for word in words if word not in endings]


def same_company(first, second):
    """Compare websites when both have one, and names otherwise."""

    first_domain = website_domain(first)
    second_domain = website_domain(second)

    if first_domain and second_domain:
        return first_domain == second_domain

    first_words = company_words(first)
    second_words = company_words(second)

    if not first_words or not second_words:
        return False

    # Padded so that a name only matches whole words of the other name.
    first_name = f" {' '.join(first_words)} "
    second_name = f" {' '.join(second_words)} "

    return (
        first_words[0] == second_words[0]
        or first_name in second_name
        or second_name in first_name
        or SequenceMatcher(None, first_name, second_name).ratio() >= 0.9
    )


def contact_details(facility):
    """Count how many ways there are to contact a facility."""

    return sum(
        bool(text(facility[field]))
        for field in ("phone", "email", "website")
    )


def remove_duplicates(facilities):
    """Keep one row per site: the one with the most contact details."""

    kept = []
    rows_at_site = {}  # site key -> positions in kept

    for facility in facilities:

        key = site_key(facility)

        if key is None:
            kept.append(facility)
            continue

        positions = rows_at_site.setdefault(key, [])

        for position in positions:

            if same_company(kept[position], facility):

                if contact_details(facility) > contact_details(kept[position]):
                    kept[position] = facility

                break

        else:
            positions.append(len(kept))
            kept.append(facility)

    return kept


CONTACT_FIELDS = ("phone", "email", "website", "contact_source")


def previous_contacts():
    """Return {(name, address): contact fields} from the CSV already on disk."""

    if not CSV_FILE.exists():
        return {}

    with open(CSV_FILE, newline="", encoding="utf-8") as file:
        return {
            (row["name"], row["address"]): {
                field: row.get(field, "") for field in CONTACT_FIELDS
            }
            for row in csv.DictReader(file)
        }


def download_facilities():
    """Download the NYC electronics drop-off sites."""

    print("Connecting to NYC Open Data...")

    rows = api_request()

    print("Records retrieved:", len(rows))

    facilities = []

    for row in rows:

        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        street = text(row.get("address"))
        borough = text(row.get("borough"))
        zipcode = text(row.get("zipcode"))

        facilities.append({
            "facility_id": row.get("objectid"),
            "name": text(row.get("dropoff_sitename")),
            "address": ", ".join(
                part for part in (street, borough, f"NY {zipcode}".strip()) if part
            ),
            "city": borough,
            "state": "NY",
            "zipcode": zipcode,
            "phone": "",  # the dataset has none: verify_dropoff.py --contacts fills them
            "email": "",
            "website": "",
            "contact_source": "",
            "facility_type": "Electronics Drop-Off",
            "feedstock": "Electronics",
            "latitude": latitude,
            "longitude": longitude,
            "source": "NYC DSNY",
            "dropoff_verified": True  # DSNY lists them as open to residents
        })

    total = len(facilities)
    facilities = remove_duplicates(facilities)

    print("Duplicates removed:", total - len(facilities))

    # Keep contact details found earlier, so that a new download never erases them.
    earlier = previous_contacts()

    for facility in facilities:
        facility.update(earlier.get((facility["name"], facility["address"]), {}))

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    if facilities:

        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=facilities[0].keys()
            )

            writer.writeheader()
            writer.writerows(facilities)

    print("Facilities saved:", len(facilities))
    print("Saved to:", CSV_FILE)


if __name__ == "__main__":
    download_facilities()
