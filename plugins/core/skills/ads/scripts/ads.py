#!/usr/bin/env python
"""NASA ADS search and retrieval tool."""
import argparse
import json
import os
import sys
import requests

BASE_URL = "https://api.adsabs.harvard.edu/v1"

def get_token():
    """Get ADS API token from environment or config file."""
    token = os.environ.get("ADS_DEV_KEY") or os.environ.get("ADS_API_TOKEN")
    if token:
        return token

    # Check config file
    config_path = os.path.expanduser("~/.ads/dev_key")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return f.read().strip()

    return None

def search_ads(query, max_results, sort, fields):
    """Search ADS and print results."""
    token = get_token()
    if not token:
        sys.exit("Error: No ADS API token found. Set ADS_DEV_KEY environment variable or create ~/.ads/dev_key")

    url = f"{BASE_URL}/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "rows": max_results,
        "sort": sort,
        "fl": fields
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error searching ADS: {e}")

    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    num_found = data.get("response", {}).get("numFound", 0)

    if not docs:
        print("No results found.")
        return

    print(f"Found {num_found} results\n")

    for doc in docs:
        bibcode = doc.get("bibcode", "")
        title = doc.get("title", ["No title"])[0] if doc.get("title") else "No title"
        authors = doc.get("author", [])
        year = doc.get("year", "")
        citation_count = doc.get("citation_count", 0)

        # First author et al.
        if len(authors) > 1:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = authors[0]
        else:
            author_str = "Unknown"

        print(f"{bibcode}: {title}")
        print(f"  {author_str} ({year}) [{citation_count} citations]")

        # Show arXiv ID if available
        arxiv = doc.get("identifier", [])
        arxiv_ids = [i for i in arxiv if i.startswith("arXiv:")]
        if arxiv_ids:
            print(f"  {arxiv_ids[0]}")
        print()

def get_record(bibcode, format_type):
    """Get a specific record by bibcode."""
    token = get_token()
    if not token:
        sys.exit("Error: No ADS API token found. Set ADS_DEV_KEY environment variable or create ~/.ads/dev_key")

    headers = {"Authorization": f"Bearer {token}"}

    if format_type == "bibtex":
        url = f"{BASE_URL}/export/bibtex"
        payload = {"bibcode": [bibcode]}
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            print(response.json().get("export", ""))
        except requests.RequestException as e:
            sys.exit(f"Error fetching BibTeX: {e}")
    else:
        url = f"{BASE_URL}/search/query"
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "title,author,year,bibcode,abstract,citation_count,doi,identifier,pub,volume,page"
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            sys.exit(f"Error fetching record: {e}")

        docs = response.json().get("response", {}).get("docs", [])
        if not docs:
            sys.exit(f"No record found for bibcode: {bibcode}")

        doc = docs[0]
        title = doc.get("title", ["No title"])[0] if doc.get("title") else "No title"
        print(f"Title: {title}")
        print(f"Authors: {', '.join(doc.get('author', []))}")
        print(f"Year: {doc.get('year', '')}")
        print(f"Bibcode: {doc.get('bibcode', '')}")

        # Publication info
        pub = doc.get("pub", "")
        vol = doc.get("volume", "")
        page = doc.get("page", [""])[0] if doc.get("page") else ""
        if pub:
            print(f"Published: {pub} {vol}, {page}")

        # DOI
        doi = doc.get("doi", [])
        if doi:
            print(f"DOI: {doi[0]}")

        # arXiv
        arxiv = [i for i in doc.get("identifier", []) if i.startswith("arXiv:")]
        if arxiv:
            print(f"arXiv: {arxiv[0]}")

        print(f"Citations: {doc.get('citation_count', 0)}")

        # Abstract
        abstract = doc.get("abstract", "")
        if abstract:
            print(f"\nAbstract:\n{abstract}")

def get_citations(bibcode, max_results):
    """Get papers citing a given bibcode."""
    token = get_token()
    if not token:
        sys.exit("Error: No ADS API token found. Set ADS_DEV_KEY environment variable or create ~/.ads/dev_key")

    url = f"{BASE_URL}/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": f"citations(bibcode:{bibcode})",
        "rows": max_results,
        "sort": "citation_count desc",
        "fl": "bibcode,title,author,year,citation_count"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching citations: {e}")

    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    num_found = data.get("response", {}).get("numFound", 0)

    print(f"Total citations: {num_found}\n")

    for doc in docs:
        bibcode = doc.get("bibcode", "")
        title = doc.get("title", ["No title"])[0] if doc.get("title") else "No title"
        authors = doc.get("author", [])
        first_author = authors[0] if authors else ""
        citation_count = doc.get("citation_count", 0)
        year = doc.get("year", "")

        print(f"{bibcode}: {title}")
        print(f"  {first_author} et al. ({year}) [{citation_count} citations]")
        print()

parser = argparse.ArgumentParser(description="Search and retrieve papers from NASA ADS.")
parser.add_argument("bibcode", nargs="?", help="ADS bibcode (e.g., 2019ApJ...882L..12P)")
parser.add_argument("--search", "-s", metavar="QUERY", help='Search query (e.g., "author:Einstein title:relativity")')
parser.add_argument("--max-results", "-n", type=int, default=10, help="Max results (default: 10)")
parser.add_argument("--sort", default="citation_count desc", help="Sort order (default: citation_count desc)")
parser.add_argument("--format", "-f", choices=["json", "bibtex"], help="Output format")
parser.add_argument("--citations", "-c", action="store_true", help="Show papers citing this record")

args = parser.parse_args()

# Default fields for search
fields = "bibcode,title,author,year,citation_count,identifier"

if args.search:
    search_ads(args.search, args.max_results, args.sort, fields)
elif args.bibcode:
    if args.citations:
        get_citations(args.bibcode, args.max_results)
    else:
        get_record(args.bibcode, args.format)
else:
    parser.print_help()
    sys.exit(1)
