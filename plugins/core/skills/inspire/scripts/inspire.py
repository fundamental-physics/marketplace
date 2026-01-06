#!/usr/bin/env python
"""INSPIRE-HEP search and retrieval tool."""
import argparse
import json
import sys
import requests

BASE_URL = "https://inspirehep.net/api"

def search_inspire(query, max_results, sort):
    """Search INSPIRE and print results."""
    url = f"{BASE_URL}/literature"
    params = {"q": query, "size": max_results, "sort": sort}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error searching INSPIRE: {e}")

    data = response.json()
    hits = data.get("hits", {}).get("hits", [])

    if not hits:
        print("No results found.")
        return

    for hit in hits:
        meta = hit.get("metadata", {})
        recid = hit.get("id", "")

        # Get title
        titles = meta.get("titles", [])
        title = titles[0].get("title", "No title") if titles else "No title"

        # Get authors (first 3)
        authors = meta.get("authors", [])
        author_names = [a.get("full_name", "") for a in authors[:3]]
        author_str = ", ".join(author_names)
        if len(authors) > 3:
            author_str += "..."

        # Get arxiv ID if available
        arxiv_eprints = meta.get("arxiv_eprints", [])
        arxiv_id = arxiv_eprints[0].get("value", "") if arxiv_eprints else ""

        # Get citation count
        citation_count = meta.get("citation_count", 0)

        print(f"{recid}: {title}")
        print(f"  Authors: {author_str}")
        if arxiv_id:
            print(f"  arXiv: {arxiv_id}")
        print(f"  Citations: {citation_count}")
        print()

def get_record(identifier, format_type):
    """Get a specific record by ID, arXiv ID, or DOI."""
    # Determine the endpoint based on identifier format
    if identifier.startswith("10."):
        url = f"{BASE_URL}/doi/{identifier}"
    elif "/" in identifier or "." in identifier:
        # Likely an arXiv ID
        url = f"{BASE_URL}/arxiv/{identifier}"
    else:
        # Assume INSPIRE recid
        url = f"{BASE_URL}/literature/{identifier}"

    params = {}
    if format_type:
        params["format"] = format_type

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching record: {e}")

    if format_type in ["bibtex", "latex-eu", "latex-us"]:
        print(response.text)
    else:
        data = response.json()
        meta = data.get("metadata", {})

        # Title
        titles = meta.get("titles", [])
        title = titles[0].get("title", "No title") if titles else "No title"
        print(f"Title: {title}")

        # Authors
        authors = meta.get("authors", [])
        print(f"Authors: {', '.join(a.get('full_name', '') for a in authors)}")

        # arXiv
        arxiv_eprints = meta.get("arxiv_eprints", [])
        if arxiv_eprints:
            print(f"arXiv: {arxiv_eprints[0].get('value', '')}")

        # DOI
        dois = meta.get("dois", [])
        if dois:
            print(f"DOI: {dois[0].get('value', '')}")

        # Publication info
        pub_info = meta.get("publication_info", [])
        if pub_info:
            pub = pub_info[0]
            journal = pub.get("journal_title", "")
            volume = pub.get("journal_volume", "")
            page = pub.get("page_start", "")
            year = pub.get("year", "")
            if journal:
                print(f"Published: {journal} {volume}, {page} ({year})")

        # Citations
        print(f"Citations: {meta.get('citation_count', 0)}")

        # Abstract
        abstracts = meta.get("abstracts", [])
        if abstracts:
            print(f"\nAbstract:\n{abstracts[0].get('value', '')}")

def get_citations(identifier, max_results):
    """Get papers that cite a given record."""
    # First get the recid (control_number)
    if identifier.startswith("10."):
        url = f"{BASE_URL}/doi/{identifier}"
    elif "/" in identifier or "." in identifier:
        url = f"{BASE_URL}/arxiv/{identifier}"
    else:
        url = f"{BASE_URL}/literature/{identifier}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        recid = data.get("metadata", {}).get("control_number", identifier)
    except requests.RequestException:
        recid = identifier

    # Search for citing papers
    search_url = f"{BASE_URL}/literature"
    params = {"q": f"refersto:recid:{recid}", "size": max_results, "sort": "mostcited"}

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Error fetching citations: {e}")

    data = response.json()
    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", 0)

    print(f"Total citations: {total}\n")

    for hit in hits:
        meta = hit.get("metadata", {})
        recid = hit.get("id", "")
        titles = meta.get("titles", [])
        title = titles[0].get("title", "No title") if titles else "No title"
        authors = meta.get("authors", [])
        first_author = authors[0].get("full_name", "") if authors else ""
        citation_count = meta.get("citation_count", 0)

        print(f"{recid}: {title}")
        print(f"  {first_author} et al. [{citation_count} citations]")
        print()

parser = argparse.ArgumentParser(description="Search and retrieve papers from INSPIRE-HEP.")
parser.add_argument("identifier", nargs="?", help="INSPIRE recid, arXiv ID, or DOI")
parser.add_argument("--search", "-s", metavar="QUERY", help='Search query (e.g., "a E.Witten.1", "t dark matter")')
parser.add_argument("--max-results", "-n", type=int, default=10, help="Max results (default: 10)")
parser.add_argument("--sort", choices=["mostrecent", "mostcited"], default="mostrecent", help="Sort order")
parser.add_argument("--format", "-f", choices=["json", "bibtex", "latex-eu", "latex-us"], help="Output format")
parser.add_argument("--citations", "-c", action="store_true", help="Show papers citing this record")

args = parser.parse_args()

if args.search:
    search_inspire(args.search, args.max_results, args.sort)
elif args.identifier:
    if args.citations:
        get_citations(args.identifier, args.max_results)
    else:
        get_record(args.identifier, args.format)
else:
    parser.print_help()
    sys.exit(1)
