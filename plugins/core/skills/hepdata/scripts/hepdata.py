#!/usr/bin/env python
"""HEPData search and download tool."""
import argparse
import json
import os
import sys
import requests

BASE_URL = "https://www.hepdata.net"

def search_hepdata(query, max_results):
    """Search HEPData and print results."""
    # First get IDs
    url = f"{BASE_URL}/search/ids"
    params = {"q": query}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error searching HEPData: {e}")

    ids = response.json()
    if not ids:
        print("No results found.")
        return

    # Limit results
    ids = ids[:max_results]
    print(f"Found {len(ids)} results\n")

    # Get details for each record
    for hepdata_id in ids:
        try:
            record_url = f"{BASE_URL}/record/{hepdata_id}?format=json"
            record_response = requests.get(record_url)
            record_response.raise_for_status()
            data = record_response.json()

            # Data is nested under 'record' key
            record = data.get("record", data)

            title = record.get("title", "No title")
            collaborations = record.get("collaborations", [])
            collaboration = collaborations[0] if collaborations else ""
            inspire_id = record.get("inspire_id", "")
            year = record.get("year", "")
            tables = len(data.get("data_tables", []))

            print(f"{hepdata_id}: {title}")
            if collaboration:
                print(f"  Collaboration: {collaboration} ({year})")
            if inspire_id:
                print(f"  INSPIRE: {inspire_id}")
            print(f"  Tables: {tables}")
            print()
        except requests.RequestException:
            print(f"{hepdata_id}: (failed to fetch details)")
            print()

def get_record(identifier, id_type):
    """Get a record by HEPData ID, INSPIRE ID, or arXiv ID."""
    # Determine endpoint based on ID type
    if id_type == "inspire":
        url = f"{BASE_URL}/record/ins{identifier}?format=json"
    elif id_type == "arxiv":
        # Try to find via search
        search_url = f"{BASE_URL}/search/ids"
        params = {"q": f"arxiv:{identifier}"}
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            ids = response.json()
            if not ids:
                sys.exit(f"No HEPData record found for arXiv:{identifier}")
            url = f"{BASE_URL}/record/{ids[0]}?format=json"
        except requests.RequestException as e:
            sys.exit(f"Error searching: {e}")
    else:
        url = f"{BASE_URL}/record/{identifier}?format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    data = response.json()
    record = data.get("record", data)

    print(f"Title: {record.get('title', 'No title')}")

    collaborations = record.get("collaborations", [])
    if collaborations:
        print(f"Collaboration: {collaborations[0]}")

    print(f"Year: {record.get('year', 'N/A')}")

    inspire_id = record.get("inspire_id", "")
    if inspire_id:
        print(f"INSPIRE ID: {inspire_id}")

    doi = record.get("doi", "")
    if doi:
        print(f"DOI: {doi}")

    # List tables (from top-level data, not nested record)
    tables = data.get("data_tables", [])
    if tables:
        print(f"\nData Tables ({len(tables)}):")
        for i, table in enumerate(tables):
            if isinstance(table, dict):
                name = table.get("name", f"Table {i+1}")
                desc = table.get("description", "")
                if desc and len(desc) > 60:
                    desc = desc[:60] + "..."
            else:
                name = f"Table {i+1}"
                desc = ""
            print(f"  {i+1}. {name}")
            if desc:
                print(f"      {desc}")

def list_tables(identifier, id_type):
    """List all tables in a record."""
    if id_type == "inspire":
        url = f"{BASE_URL}/record/ins{identifier}?format=json"
    else:
        url = f"{BASE_URL}/record/{identifier}?format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    data = response.json()
    tables = data.get("data_tables", [])

    if not tables:
        print("No tables found.")
        return

    print(f"Tables in record:\n")
    for i, table in enumerate(tables):
        if isinstance(table, dict):
            name = table.get("name", f"Table {i+1}")
            desc = table.get("description", "No description")
        else:
            name = str(table)
            desc = ""
        print(f"{i+1}. {name}")
        if desc:
            print(f"   {desc}")
        print()

def download_table(identifier, table_name, file_format, output_dir, id_type):
    """Download a specific table from a record."""
    # Build download URL
    if id_type == "inspire":
        base = f"https://www.hepdata.net/download/submission/ins{identifier}"
    else:
        base = f"https://www.hepdata.net/download/submission/{identifier}"

    if table_name:
        # Download specific table
        url = f"{base}/{table_name}/{file_format}"
    else:
        # Download all tables
        url = f"{base}/{file_format}"

    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error downloading: {e}")

    os.makedirs(output_dir, exist_ok=True)

    # Determine filename
    if table_name:
        filename = f"{identifier}_{table_name}.{file_format}"
    else:
        filename = f"{identifier}_all.{file_format}"
        if file_format in ["yaml", "json"]:
            filename += ".tar.gz"

    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded: {output_path}")

parser = argparse.ArgumentParser(description="Search and download from HEPData.")
parser.add_argument("identifier", nargs="?", help="HEPData ID, INSPIRE ID, or arXiv ID")
parser.add_argument("--search", "-s", metavar="QUERY", help="Search query")
parser.add_argument("--max-results", "-n", type=int, default=10, help="Max results (default: 10)")
parser.add_argument("--inspire", "-i", action="store_true", help="Identifier is an INSPIRE ID")
parser.add_argument("--arxiv", "-a", action="store_true", help="Identifier is an arXiv ID")
parser.add_argument("--tables", "-t", action="store_true", help="List tables in record")
parser.add_argument("--download", "-d", metavar="TABLE", nargs="?", const="__all__", help="Download table (omit for all)")
parser.add_argument("--format", "-f", default="csv", choices=["csv", "yaml", "json", "root", "yoda"], help="Download format")
parser.add_argument("--output-dir", "-o", default=".", help="Output directory")

args = parser.parse_args()

# Determine ID type
if args.inspire:
    id_type = "inspire"
elif args.arxiv:
    id_type = "arxiv"
else:
    id_type = "hepdata"

if args.search:
    search_hepdata(args.search, args.max_results)
elif args.identifier:
    if args.tables:
        list_tables(args.identifier, id_type)
    elif args.download:
        table_name = None if args.download == "__all__" else args.download
        download_table(args.identifier, table_name, args.format, args.output_dir, id_type)
    else:
        get_record(args.identifier, id_type)
else:
    parser.print_help()
    sys.exit(1)
