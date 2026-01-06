#!/usr/bin/env python
"""Zenodo search and download tool."""
import argparse
import json
import os
import sys
import requests

BASE_URL = "https://zenodo.org/api"

def search_zenodo(query, max_results, sort, record_type):
    """Search Zenodo and print results."""
    url = f"{BASE_URL}/records"
    params = {
        "q": query,
        "size": max_results,
        "sort": sort,
    }
    if record_type:
        params["type"] = record_type

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error searching Zenodo: {e}")

    data = response.json()
    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", 0)

    if not hits:
        print("No results found.")
        return

    print(f"Found {total} results\n")

    for hit in hits:
        record_id = hit.get("id", "")
        meta = hit.get("metadata", {})

        title = meta.get("title", "No title")
        creators = meta.get("creators", [])
        creator_names = [c.get("name", "") for c in creators[:3]]
        creator_str = ", ".join(creator_names)
        if len(creators) > 3:
            creator_str += "..."

        pub_date = meta.get("publication_date", "")
        resource_type = meta.get("resource_type", {}).get("title", "")
        doi = meta.get("doi", "")

        print(f"{record_id}: {title}")
        print(f"  {creator_str} ({pub_date})")
        print(f"  Type: {resource_type}")
        if doi:
            print(f"  DOI: {doi}")
        print()

def get_record(record_id, format_type):
    """Get a specific record by ID."""
    url = f"{BASE_URL}/records/{record_id}"

    headers = {}
    if format_type == "bibtex":
        headers["Accept"] = "application/x-bibtex"
    elif format_type == "datacite":
        headers["Accept"] = "application/x-datacite+xml"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    if format_type in ["bibtex", "datacite"]:
        print(response.text)
        return

    data = response.json()
    meta = data.get("metadata", {})

    print(f"Title: {meta.get('title', 'No title')}")

    creators = meta.get("creators", [])
    print(f"Creators: {', '.join(c.get('name', '') for c in creators)}")

    print(f"Publication Date: {meta.get('publication_date', '')}")
    print(f"Type: {meta.get('resource_type', {}).get('title', '')}")

    doi = meta.get("doi", "")
    if doi:
        print(f"DOI: {doi}")

    license_info = meta.get("license", {})
    if license_info:
        print(f"License: {license_info.get('id', '')}")

    # Files
    files = data.get("files", [])
    if files:
        print(f"\nFiles ({len(files)}):")
        for f in files:
            size_mb = f.get("size", 0) / (1024 * 1024)
            print(f"  - {f.get('key', '')} ({size_mb:.2f} MB)")

    # Description
    desc = meta.get("description", "")
    if desc:
        # Strip HTML tags simply
        import re
        desc_clean = re.sub(r'<[^>]+>', '', desc)
        if len(desc_clean) > 500:
            desc_clean = desc_clean[:500] + "..."
        print(f"\nDescription:\n{desc_clean}")

def list_files(record_id):
    """List files in a record."""
    url = f"{BASE_URL}/records/{record_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    data = response.json()
    files = data.get("files", [])

    if not files:
        print("No files found.")
        return

    print(f"Files in record {record_id}:\n")
    for f in files:
        size_mb = f.get("size", 0) / (1024 * 1024)
        checksum = f.get("checksum", "").replace("md5:", "")[:12]
        link = f.get("links", {}).get("self", "")
        print(f"{f.get('key', '')}")
        print(f"  Size: {size_mb:.2f} MB | MD5: {checksum}...")
        print(f"  URL: {link}")
        print()

def download_file(record_id, filename, output_dir):
    """Download a file from a record."""
    url = f"{BASE_URL}/records/{record_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    data = response.json()
    files = data.get("files", [])

    # Find the file
    target_file = None
    for f in files:
        if f.get("key") == filename or filename is None:
            target_file = f
            break

    if not target_file:
        if filename:
            sys.exit(f"File '{filename}' not found in record {record_id}")
        else:
            sys.exit(f"No files found in record {record_id}")

    file_url = target_file.get("links", {}).get("self", "")
    file_key = target_file.get("key", "file")

    if not file_url:
        sys.exit("Could not get download URL")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_key)

    print(f"Downloading {file_key}...")
    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.RequestException as e:
        sys.exit(f"Error downloading: {e}")

    print(f"Saved to: {output_path}")

parser = argparse.ArgumentParser(description="Search and download from Zenodo.")
parser.add_argument("record_id", nargs="?", help="Zenodo record ID")
parser.add_argument("--search", "-s", metavar="QUERY", help="Search query")
parser.add_argument("--max-results", "-n", type=int, default=10, help="Max results (default: 10)")
parser.add_argument("--sort", default="bestmatch", choices=["bestmatch", "mostrecent", "-mostrecent"], help="Sort order")
parser.add_argument("--type", dest="record_type", help="Filter by type (dataset, software, publication, etc.)")
parser.add_argument("--format", "-f", choices=["json", "bibtex", "datacite"], help="Output format")
parser.add_argument("--files", action="store_true", help="List files in record")
parser.add_argument("--download", "-d", metavar="FILENAME", nargs="?", const="__first__", help="Download file (omit name for first file)")
parser.add_argument("--output-dir", "-o", default=".", help="Output directory for downloads")

args = parser.parse_args()

if args.search:
    search_zenodo(args.search, args.max_results, args.sort, args.record_type)
elif args.record_id:
    if args.files:
        list_files(args.record_id)
    elif args.download:
        filename = None if args.download == "__first__" else args.download
        download_file(args.record_id, filename, args.output_dir)
    else:
        get_record(args.record_id, args.format)
else:
    parser.print_help()
    sys.exit(1)
