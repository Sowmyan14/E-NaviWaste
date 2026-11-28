"""Find the NYC electronics drop-off sites within 5 miles of a ZIP code.

Run:     python find_recyclers.py            (asks for a ZIP code)
         python find_recyclers.py 11220
         python find_recyclers.py 11220 --item laptop
         python find_recyclers.py 11220 --item "hard drive"
         python find_recyclers.py 11220 --item laptop --all   (every nearby site, each marked)
Import:  from ewaste_locator.find_recyclers import find_recyclers

Distances are straight-line miles from the center of the ZIP code, not driving miles.

What each site accepts comes from data/acceptance.csv: "yes", "no" or blank (not known yet).
"""

import argparse
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
ACCEPTANCE_FILE = CSV_FILE.parent / "acceptance.csv"

# The devices a user can ask about. They are also the column names in acceptance.csv.
DEVICES = (
    "Phone", "Laptop", "Tablet", "Television", "Router", "Hard Drive", "Smartwatch", "Modem",
    "Smart Speaker", "Remote Control", "Game Console", "Monitor", "Webcam", "Camera",
)

# Keywords allowed in the conditions column, and how to say them to the user.
# {value} is the number after the colon, as in limit:7 or max_size:50.
CONDITION_TEXT = {
    "working_only": "Only takes items that still work.",
    "donation": "Takes items as donations for resale, not for recycling.",
    "trade_in": "May give store credit for eligible items.",
    "pays_cash": "A scrap buyer: may pay cash by weight.",
    "brand_only": "Only takes its own brand of devices.",
    "own_brand_free": "Free for the store's own brand.",
    "fee": "Charges a fee.",
    "limit": "Up to {value} items per day.",
    "max_size": "Screens up to {value} inches.",
    "max_age": "Only if less than {value} years old.",
    "remove_battery": "Remove batteries first.",
    "residents_only": "Residents only. You may be asked for proof of address.",
    "id_required": "Bring a photo ID.",
    "business_only": "Serves businesses only.",
    "appointment": "Book an appointment first.",
    "limited_hours": "Open limited hours. Check before going.",
    "ask_staff": "Ask staff at the service desk.",
    "offers_data_wipe": "Offers data wiping.",
}


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
        "facility_id": clean(row["facility_id"]),
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


# --- what each facility accepts -----------------------------------------------------------


def normalize_device(text):
    """Return the DEVICES spelling of a device name typed in any case, such as "hard drive"."""
    for device in DEVICES:
        if device.lower() == " ".join(text.split()).lower():
            return device
    raise LocatorError(f"Unknown device {text!r}. Choose one of: {', '.join(DEVICES)}.")


def parse_conditions(text, where):
    """Turn "limit:7;Television:fee" into [(None, "limit", "7"), ("Television", "fee", "")].

    A typo raises an error naming the row, instead of being silently ignored.
    """
    conditions = []
    for token in filter(None, (part.strip() for part in text.split(";"))):
        parts = token.split(":")
        device = parts.pop(0) if parts[0] in DEVICES else None
        keyword, value = parts[0], ":".join(parts[1:])
        if keyword not in CONDITION_TEXT or ("{value}" in CONDITION_TEXT[keyword]) != bool(value):
            raise LocatorError(f"acceptance.csv, {where}: cannot read condition {token!r}.")
        conditions.append((device, keyword, value))
    return conditions


def load_acceptance():
    """Read acceptance.csv into {("name" | "facility_id", value): policy}.

    Read on every search (it is small), so edits show up without restarting the app.
    """
    if not ACCEPTANCE_FILE.exists():
        return {}  # no acceptance data yet: every site is "call ahead"
    policies = {}
    with open(ACCEPTANCE_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [device for device in DEVICES if device not in reader.fieldnames]
        if missing:
            raise LocatorError(f"acceptance.csv is missing the columns: {', '.join(missing)}.")
        for row in reader:
            where = f"row {row['match_value']!r}"
            accepts = {}
            for device in DEVICES:
                value = clean(row[device]).lower()
                if value not in ("yes", "no", ""):
                    raise LocatorError(f"acceptance.csv, {where}: {device} must be yes, no or blank.")
                accepts[device] = value
            policies[(row["match_type"], clean(row["match_value"]))] = {
                "accepts": accepts,
                "conditions": parse_conditions(row["conditions"], where),
                "source_url": clean_url(row["source_url"]),
                "verified_on": clean(row["verified_on"]),
            }
    return policies


def policy_for(facility, policies):
    """A row for this one facility wins over the row for its organization."""
    return (
        policies.get(("facility_id", facility["facility_id"]))
        or policies.get(("name", facility["name"]))
        or {"accepts": {}, "conditions": [], "source_url": "", "verified_on": ""}
    )


def describe_conditions(conditions, device=None):
    """Sentences for the user. With a device, other devices' conditions are left out.

    A device's own conditions keep their prefix ("Television: Up to 2 items per day."), so
    they do not read as contradicting the site-wide ones ("Up to 10 items per day.").
    """
    notes = []
    for scope, keyword, value in conditions:
        if scope and device and scope != device:
            continue
        text = CONDITION_TEXT[keyword].format(value=value)
        notes.append(f"{scope}: {text}" if scope else text)
    return notes


# --- search --------------------------------------------------------------------------------


def find_recyclers(zip_code, device=None, include_refused=False, radius_miles=RADIUS_MILES):
    """Return the facilities within ``radius_miles`` of a ZIP code, nearest first.

    Each facility has "accepted" (the devices it is confirmed to take) and "notes" (its
    conditions as sentences). With a ``device``, each site also has "device_status": "yes"
    (confirmed to take it), "unknown" (not known yet) or "no" (does not take it). Sites with
    "no" are left out unless ``include_refused`` is true, as for a "show all nearby" view.
    Sites that serve only businesses are always left out.
    """
    device = normalize_device(device) if device else None
    latitude, longitude, _ = locate_zip(zip_code)
    policies = load_acceptance()
    nearby = []
    for facility in load_facilities():
        policy = policy_for(facility, policies)
        if any(keyword == "business_only" for _, keyword, _ in policy["conditions"]):
            continue
        status = (policy["accepts"].get(device) or "unknown") if device else None
        if status == "no" and not include_refused:
            continue
        miles = geodesic(
            (latitude, longitude), (facility["latitude"], facility["longitude"])
        ).miles
        if miles <= radius_miles:
            nearby.append((miles, {
                **facility,
                "distance_mi": round(miles, 1),
                "accepted": [d for d in DEVICES if policy["accepts"].get(d) == "yes"],
                "notes": describe_conditions(policy["conditions"], device),
                "policy_url": policy["source_url"],
                "policy_checked": policy["verified_on"],
                **({"device_status": status} if device else {}),
            }))
    nearby.sort(key=lambda pair: pair[0])
    return [facility for _, facility in nearby]


# --- command line --------------------------------------------------------------------------


STATUS_LABEL = {
    "yes": "Takes a {device}",
    "unknown": "Might take a {device}. Call ahead",
    "no": "Does not take a {device}",
}


def print_facility(number, facility, device, show_status=False):
    print(f"{number}. {facility['name']} - {facility['distance_mi']} mi")
    if show_status:
        print(f"   [{STATUS_LABEL[facility['device_status']].format(device=device.lower())}]")
    for field in ("address", "phone", "email", "url"):
        if facility[field]:
            print(f"   {facility[field]}")
    if facility["accepted"]:
        print(f"   Confirmed to take: {', '.join(facility['accepted'])}")
    elif not device:
        print("   Accepted items not confirmed yet. Call ahead.")
    for note in facility["notes"]:
        print(f"   - {note}")
    print()


def print_results(zip_code, device=None, show_all=False):
    _, _, place = locate_zip(zip_code)
    results = find_recyclers(zip_code, device, include_refused=show_all)
    where = f"within {RADIUS_MILES} miles of {normalize_zip(zip_code)} ({place})"

    if not results:
        what = f"that take a {device.lower()}" if device and not show_all else ""
        print(f"\nNo NYC electronics drop-off sites {what} found {where}.")
        return

    if not device or show_all:
        noun = "recycler" if len(results) == 1 else "recyclers"
        print(f"\n{len(results)} e-waste {noun} {where}:\n")
        for number, facility in enumerate(results, 1):
            print_facility(number, facility, device, show_status=bool(device))
        return

    confirmed = [f for f in results if f["device_status"] == "yes"]
    unknown = [f for f in results if f["device_status"] == "unknown"]
    print(f"\n{len(confirmed)} sites {where} are confirmed to take a {device.lower()}:\n")
    for number, facility in enumerate(confirmed, 1):
        print_facility(number, facility, device)
    if unknown:
        print(f"{len(unknown)} more sites might take it. Call ahead to confirm:\n")
        for number, facility in enumerate(unknown, len(confirmed) + 1):
            print_facility(number, facility, device)


def main():
    parser = argparse.ArgumentParser(description="Find NYC electronics drop-off sites.")
    parser.add_argument("zip_code", nargs="?", help="5-digit ZIP code (asked for if left out)")
    parser.add_argument("--item", help=f"only sites that take this device: {', '.join(DEVICES)}")
    parser.add_argument("--all", action="store_true",
                        help="with --item, list every nearby site, each marked with whether it takes the device")
    args = parser.parse_args()
    try:
        device = normalize_device(args.item) if args.item else None
        zip_code = args.zip_code or input("Enter your ZIP code: ")
        print_results(zip_code, device, show_all=args.all)
    except LocatorError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
