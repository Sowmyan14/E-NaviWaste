import requests
import csv
from pathlib import Path

# EPA electronics recycler API
API_URL = (
    "https://services.arcgis.com/cJ9YHowT8TU7DUyn/"
    "ArcGIS/rest/services/High-volume_Recycling_Infrastructure/"
    "FeatureServer/0/query"
)

DATA_FOLDER = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_FOLDER / "epa_facilities.csv"

FIELDS = [
    "Name",
    "Address",
    "County",
    "TELEPHONE",
    "Email",
    "Website",
    "Infra_Type",
    "Feedstock",
    "Street",
    "City",
    "State",
    "Zip_Code"
]


def api_request(params):
    """Send a request to the EPA API."""

    response = requests.get(
        API_URL,
        params=params,
        timeout=45
    )

    response.raise_for_status()

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result



def download_facilities():
    """Download EPA electronics recycler records."""

    print("Connecting to EPA API...")

    data = api_request({
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "resultRecordCount": 1000,
        "f": "json"
    })

    features = data.get("features", [])

    print("Records retrieved:", len(features))

    facilities = []

    for feature in features:

        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}

        latitude = geometry.get("y")
        longitude = geometry.get("x")

        if latitude is None or longitude is None:
            continue

        facilities.append({
            "facility_id": attributes.get("OBJECTID"),
            "name": attributes.get("Name"),
            "address": (
                attributes.get("Address")
                or attributes.get("Street")
            ),
            "city": attributes.get("City"),
            "state": attributes.get("State"),
            "zipcode": attributes.get("Zip_Code"),
            "phone": attributes.get("TELEPHONE"),
            "email": attributes.get("Email"),
            "website": attributes.get("Website"),
            "facility_type": attributes.get("Infra_Type"),
            "feedstock": attributes.get("Feedstock"),
            "latitude": latitude,
            "longitude": longitude,
            "source": "US EPA",
            "dropoff_verified": False
        })

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
