#!/usr/bin/env python
"""GWOSC (Gravitational Wave Open Science Center) search and download tool."""
import argparse
import json
import os
import sys
import requests

BASE_URL = "https://gwosc.org"

def list_catalogs():
    """List available event catalogs."""
    url = f"{BASE_URL}/eventapi/json/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching catalogs: {e}")

    data = response.json()

    print("Available Catalogs:\n")
    for cat_name, cat_info in data.items():
        if isinstance(cat_info, dict):
            desc = cat_info.get("description", "")
            events = cat_info.get("events", [])
            print(f"{cat_name}")
            if desc:
                print(f"  {desc}")
            print(f"  Events: {len(events)}")
            print()

def list_events(catalog, max_results):
    """List events from a catalog."""
    if catalog:
        url = f"{BASE_URL}/eventapi/jsonfull/{catalog}/"
    else:
        url = f"{BASE_URL}/eventapi/jsonfull/GWTC/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching events: {e}")

    data = response.json()
    events = data.get("events", {})

    if not events:
        print("No events found.")
        return

    print(f"Found {len(events)} events\n")

    count = 0
    for event_name, event_data in events.items():
        if count >= max_results:
            break

        # jsonfull puts GPS at top level, not in parameters
        gps = event_data.get("GPS", "N/A")
        common_name = event_data.get("commonName", event_name)

        print(f"{common_name}")
        print(f"  GPS: {gps}")
        print()
        count += 1

def search_events(min_mass1=None, max_mass1=None, min_mass2=None, max_mass2=None,
                  min_distance=None, max_distance=None, min_snr=None, max_results=10):
    """Search events by criteria."""
    url = f"{BASE_URL}/eventapi/jsonfull/query/show"

    params = {}
    if min_mass1 is not None:
        params["min-mass-1-source"] = min_mass1
    if max_mass1 is not None:
        params["max-mass-1-source"] = max_mass1
    if min_mass2 is not None:
        params["min-mass-2-source"] = min_mass2
    if max_mass2 is not None:
        params["max-mass-2-source"] = max_mass2
    if min_distance is not None:
        params["min-luminosity-distance"] = min_distance
    if max_distance is not None:
        params["max-luminosity-distance"] = max_distance
    if min_snr is not None:
        params["min-network-matched-filter-snr"] = min_snr

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error searching events: {e}")

    data = response.json()
    events = data.get("events", {})

    if not events:
        print("No events match the criteria.")
        return

    print(f"Found {len(events)} matching events\n")

    count = 0
    for event_name, event_data in events.items():
        if count >= max_results:
            print(f"... and {len(events) - max_results} more")
            break

        gps = event_data.get("GPS", "N/A")
        common_name = event_data.get("commonName", event_name)

        print(f"{common_name}")
        print(f"  GPS: {gps}")
        print()
        count += 1

def get_event(event_name, version=None):
    """Get detailed information about a specific event."""
    if version:
        url = f"{BASE_URL}/eventapi/json/event/{event_name}/v{version}/"
    else:
        url = f"{BASE_URL}/eventapi/json/event/{event_name}/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching event: {e}")

    data = response.json()
    events = data.get("events", {})

    if not events:
        sys.exit(f"Event {event_name} not found")

    # Get the event data (there should be only one)
    event_key = list(events.keys())[0]
    event_data = events[event_key]

    print(f"Event: {event_key}")
    print(f"Version: {event_data.get('version', 'N/A')}")
    print(f"Catalog: {event_data.get('catalog.shortName', 'N/A')}")
    print(f"GPS: {event_data.get('GPS', 'N/A')}")
    print()

    # Parameters are nested under named PE result keys
    # Find the preferred one or use the first available
    params_dict = event_data.get("parameters", {})
    params = None
    params_name = None

    for name, p in params_dict.items():
        if isinstance(p, dict):
            if p.get("is_preferred"):
                params = p
                params_name = name
                break
            if params is None:
                params = p
                params_name = name

    if params:
        print(f"Parameters ({params_name}):")
        print(f"  Mass 1 (source): {params.get('mass_1_source', 'N/A')} M_sun")
        print(f"  Mass 2 (source): {params.get('mass_2_source', 'N/A')} M_sun")
        print(f"  Chirp Mass: {params.get('chirp_mass_source', 'N/A')} M_sun")
        print(f"  Effective Spin: {params.get('chi_eff', 'N/A')}")
        print(f"  Luminosity Distance: {params.get('luminosity_distance', 'N/A')} Mpc")
        print(f"  Redshift: {params.get('redshift', 'N/A')}")
        print(f"  Network SNR: {params.get('network_matched_filter_snr', 'N/A')}")
        print(f"  Final Mass: {params.get('final_mass_source', 'N/A')} M_sun")
        print(f"  Final Spin: {params.get('final_spin', 'N/A')}")
    else:
        print("No parameter estimates available.")

    # List available strain files
    strain_files = event_data.get("strain", [])
    if strain_files:
        print(f"\nStrain Data ({len(strain_files)} files):")
        detectors = set()
        for sf in strain_files:
            det = sf.get("detector", "")
            detectors.add(det)
        print(f"  Detectors: {', '.join(sorted(detectors))}")

        # Show format options
        formats = set()
        for sf in strain_files:
            fmt = sf.get("format", "")
            formats.add(fmt)
        print(f"  Formats: {', '.join(sorted(formats))}")

def list_strain(event_name):
    """List available strain data files for an event."""
    url = f"{BASE_URL}/eventapi/json/event/{event_name}/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching event: {e}")

    data = response.json()
    events = data.get("events", {})

    if not events:
        sys.exit(f"Event {event_name} not found")

    event_key = list(events.keys())[0]
    event_data = events[event_key]
    strain_files = event_data.get("strain", [])

    if not strain_files:
        print("No strain data available.")
        return

    print(f"Strain files for {event_name}:\n")

    for sf in strain_files:
        detector = sf.get("detector", "N/A")
        duration = sf.get("duration", "N/A")
        sample_rate = sf.get("sampling_rate", "N/A")
        fmt = sf.get("format", "N/A")
        url = sf.get("url", "")

        print(f"{detector} - {duration}s @ {sample_rate}Hz ({fmt})")
        if url:
            print(f"  {url}")
        print()

def download_strain(event_name, detector, output_dir, duration=32, fmt="hdf5"):
    """Download strain data for an event."""
    url = f"{BASE_URL}/eventapi/json/event/{event_name}/"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching event: {e}")

    data = response.json()
    events = data.get("events", {})

    if not events:
        sys.exit(f"Event {event_name} not found")

    event_key = list(events.keys())[0]
    event_data = events[event_key]
    strain_files = event_data.get("strain", [])

    # Find matching file
    target = None
    for sf in strain_files:
        if (sf.get("detector", "").upper() == detector.upper() and
            sf.get("duration") == duration and
            fmt.lower() in sf.get("format", "").lower()):
            target = sf
            break

    if not target:
        # Try without exact format match
        for sf in strain_files:
            if (sf.get("detector", "").upper() == detector.upper() and
                sf.get("duration") == duration):
                target = sf
                break

    if not target:
        print(f"No matching strain file found for {detector} {duration}s {fmt}")
        print("Available files:")
        list_strain(event_name)
        return

    file_url = target.get("url")
    if not file_url:
        sys.exit("No download URL available")

    # Determine filename
    filename = file_url.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading {filename}...")
    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.RequestException as e:
        sys.exit(f"Error downloading: {e}")

    print(f"Saved to: {output_path}")

parser = argparse.ArgumentParser(description="Search and download from GWOSC.")
parser.add_argument("event", nargs="?", help="Event name (e.g., GW150914)")
parser.add_argument("--catalogs", action="store_true", help="List available catalogs")
parser.add_argument("--list", "-l", metavar="CATALOG", nargs="?", const="GWTC", help="List events (default: GWTC)")
parser.add_argument("--max-results", "-n", type=int, default=10, help="Max results (default: 10)")

# Search parameters
parser.add_argument("--min-mass1", type=float, help="Minimum primary mass (M_sun)")
parser.add_argument("--max-mass1", type=float, help="Maximum primary mass (M_sun)")
parser.add_argument("--min-mass2", type=float, help="Minimum secondary mass (M_sun)")
parser.add_argument("--max-mass2", type=float, help="Maximum secondary mass (M_sun)")
parser.add_argument("--min-distance", type=float, help="Minimum distance (Mpc)")
parser.add_argument("--max-distance", type=float, help="Maximum distance (Mpc)")
parser.add_argument("--min-snr", type=float, help="Minimum network SNR")

# Event options
parser.add_argument("--version", "-v", type=int, help="Event version")
parser.add_argument("--strain", "-s", action="store_true", help="List strain data files")
parser.add_argument("--download", "-d", metavar="DETECTOR", help="Download strain (H1, L1, or V1)")
parser.add_argument("--duration", type=int, default=32, choices=[32, 4096], help="Strain duration (default: 32)")
parser.add_argument("--format", "-f", default="hdf5", choices=["hdf5", "gwf", "txt"], help="Strain format")
parser.add_argument("--output-dir", "-o", default=".", help="Output directory")

args = parser.parse_args()

# Determine action
has_search = any([args.min_mass1, args.max_mass1, args.min_mass2, args.max_mass2,
                  args.min_distance, args.max_distance, args.min_snr])

if args.catalogs:
    list_catalogs()
elif args.list is not None:
    list_events(args.list if args.list != "GWTC" else None, args.max_results)
elif has_search:
    search_events(
        min_mass1=args.min_mass1, max_mass1=args.max_mass1,
        min_mass2=args.min_mass2, max_mass2=args.max_mass2,
        min_distance=args.min_distance, max_distance=args.max_distance,
        min_snr=args.min_snr, max_results=args.max_results
    )
elif args.event:
    if args.strain:
        list_strain(args.event)
    elif args.download:
        download_strain(args.event, args.download, args.output_dir, args.duration, args.format)
    else:
        get_event(args.event, args.version)
else:
    parser.print_help()
    sys.exit(1)
