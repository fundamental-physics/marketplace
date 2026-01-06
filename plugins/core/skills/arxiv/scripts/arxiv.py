#!/usr/bin/env python
import argparse, requests, tarfile, sys, os
from io import BytesIO
import xml.etree.ElementTree as ET

def search_arxiv(query, max_results):
    """Search arXiv and print results."""
    url = f'http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f'Error searching arXiv: {e}')

    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    entries = root.findall('atom:entry', ns)
    if not entries:
        print('No results found.')
        return

    for entry in entries:
        # Extract arxiv ID from the id URL
        id_url = entry.find('atom:id', ns).text
        arxiv_id = id_url.split('/abs/')[-1]
        # Remove version suffix if present
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.rsplit('v', 1)[0]

        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]

        print(f'{arxiv_id}: {title}')
        print(f'  Authors: {", ".join(authors[:3])}{"..." if len(authors) > 3 else ""}')
        print()

def download_source(arxiv_number, args):
    """Download and process arXiv source."""
    url = f'https://export.arxiv.org/e-print/{arxiv_number}'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f'Error fetching data: {e}')

    tar_stream = BytesIO(response.content)
    with tarfile.open(fileobj=tar_stream, mode='r') as tar:
        if args.save:
            os.makedirs(arxiv_number, exist_ok=True)
            tar.extractall(path=arxiv_number, filter='data')
            print(f'Source saved to directory: {arxiv_number}')
        elif args.list:
            for member in tar.getmembers():
                print(member.name)
        else:
            file_types = {
                '.tex': {'print': args.tex or not any([args.bib, args.bbl]), 'lang': 'latex'},
                '.bib': {'print': args.bib or not any([args.tex, args.bbl]), 'lang': 'bibtex'},
                '.bbl': {'print': args.bbl or not any([args.tex, args.bib]), 'lang': 'bibtex'}
            }

            for member in tar.getmembers():
                for ext, details in file_types.items():
                    if member.name.endswith(ext) and details['print']:
                        file = tar.extractfile(member)
                        if file:
                            content = file.read().decode('utf-8')
                            if args.tex or args.bib or args.bbl:
                                print(f"% {member.name}\n{content}")
                            else:
                                print(f"```{details['lang']}\n% {member.name}\n{content}\n```")
                        break

parser = argparse.ArgumentParser(description='Search and download arXiv papers.')
parser.add_argument('arxiv_number', nargs='?', help='arXiv number to process')
parser.add_argument('--search', '-s', metavar='QUERY', help='Search arXiv (e.g., "au:Handley", "ti:cosmology")')
parser.add_argument('--max-results', '-n', type=int, default=10, help='Max search results (default: 10)')

group = parser.add_mutually_exclusive_group()
group.add_argument('--save', action='store_true', help='Save the entire arXiv source to a directory')
group.add_argument('--list', action='store_true', help='List the files in the tar archive')
group.add_argument('--tex', action='store_true', help='Print only the .tex files')
group.add_argument('--bib', action='store_true', help='Print only the .bib files')
group.add_argument('--bbl', action='store_true', help='Print only the .bbl files')

args = parser.parse_args()

if args.search:
    search_arxiv(args.search, args.max_results)
elif args.arxiv_number:
    download_source(args.arxiv_number, args)
else:
    parser.print_help()
    sys.exit(1)
