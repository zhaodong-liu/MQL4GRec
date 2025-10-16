#!/usr/bin/env python3
"""
Automatic Amazon Review Dataset (2018) Downloader
Downloads metadata, ratings, and 5-core review files from UCSD repository
"""

import argparse
import os
import urllib.request
import gzip
import shutil
from tqdm import tqdm
from utils import amazon18_dataset2fullname


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=os.path.basename(output_path)) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_gzip(filepath):
    """Verify that a gzip file is valid"""
    try:
        with gzip.open(filepath, 'rb') as f:
            f.read(1)
        return True
    except Exception as e:
        print(f"Invalid gzip file {filepath}: {e}")
        return False


def download_amazon_dataset(args):
    """Download Amazon dataset files for specified category"""

    dataset = args.dataset
    output_root = args.output_path

    # Get full category name
    if dataset not in amazon18_dataset2fullname:
        print(f"Error: Unknown dataset '{dataset}'")
        print(f"Available datasets: {list(amazon18_dataset2fullname.keys())}")
        return False

    category_full = amazon18_dataset2fullname[dataset]
    print(f"\n{'='*60}")
    print(f"Downloading Amazon {dataset} ({category_full}) dataset")
    print(f"{'='*60}\n")

    # Create output directories
    metadata_dir = os.path.join(output_root, 'Metadata')
    ratings_dir = os.path.join(output_root, 'Ratings')
    review_dir = os.path.join(output_root, 'Review')

    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(ratings_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    # Base URL
    base_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2"

    # File URLs and paths
    files_to_download = [
        {
            'name': 'Metadata',
            'url': f"{base_url}/metaFiles2/meta_{category_full}.json.gz",
            'path': os.path.join(metadata_dir, f"meta_{category_full}.json.gz")
        },
        {
            'name': 'Ratings (CSV)',
            'url': f"{base_url}/categoryFilesSmall/{category_full}.csv",
            'path': os.path.join(ratings_dir, f"{category_full}.csv")
        },
        {
            'name': '5-core Reviews',
            'url': f"{base_url}/categoryFilesSmall/{category_full}_5.json.gz",
            'path': os.path.join(review_dir, f"{category_full}_5.json.gz")
        }
    ]

    # Download each file
    download_success = True
    for file_info in files_to_download:
        print(f"\n[{file_info['name']}]")
        print(f"URL: {file_info['url']}")
        print(f"Saving to: {file_info['path']}")

        # Skip if file already exists and is valid
        if os.path.exists(file_info['path']):
            if file_info['path'].endswith('.gz'):
                if verify_gzip(file_info['path']):
                    print(f"✓ File already exists and is valid, skipping download")
                    continue
                else:
                    print(f"⚠ Existing file is corrupted, re-downloading...")
                    os.remove(file_info['path'])
            else:
                if os.path.getsize(file_info['path']) > 0:
                    print(f"✓ File already exists, skipping download")
                    continue

        # Download file
        success = download_url(file_info['url'], file_info['path'])

        if success:
            # Verify gzip files
            if file_info['path'].endswith('.gz'):
                if verify_gzip(file_info['path']):
                    print(f"✓ Download successful and verified")
                else:
                    print(f"✗ Download corrupted, please retry")
                    download_success = False
            else:
                print(f"✓ Download successful")
        else:
            print(f"✗ Download failed")
            download_success = False

    print(f"\n{'='*60}")
    if download_success:
        print(f"✓ All files downloaded successfully for {dataset}!")
        print(f"\nDownloaded files:")
        print(f"  - Metadata: {metadata_dir}/meta_{category_full}.json.gz")
        print(f"  - Ratings:  {ratings_dir}/{category_full}.csv")
        print(f"  - Reviews:  {review_dir}/{category_full}_5.json.gz")
    else:
        print(f"⚠ Some files failed to download. Please check errors above.")
    print(f"{'='*60}\n")

    return download_success


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Amazon Review Dataset (2018) from UCSD repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Musical Instruments dataset
  python 0_download_amazon_data.py --dataset Instruments

  # Download to custom directory
  python 0_download_amazon_data.py --dataset Arts --output_path /path/to/data

  # Download multiple datasets
  python 0_download_amazon_data.py --dataset Instruments --dataset Arts --dataset Games

Available datasets:
  Beauty, Fashion, Appliances, Arts, Automotive, Books, CDs, Cell,
  Clothing, Music, Electronics, Gift, Food, Home, Scientific, Kindle,
  Luxury, Magazine, Movies, Instruments, Office, Garden, Pet, Pantry,
  Software, Sports, Tools, Toys, Games
        """
    )
    parser.add_argument(
        '--dataset',
        type=str,
        action='append',
        required=True,
        help='Dataset name(s) to download (can specify multiple times)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/tmp/amazon18',
        help='Output directory for downloaded files (default: /tmp/amazon18)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # List datasets if requested
    if args.list_datasets:
        print("\nAvailable Amazon 2018 datasets:")
        print("=" * 50)
        for short_name, full_name in sorted(amazon18_dataset2fullname.items()):
            print(f"  {short_name:15} -> {full_name}")
        print("=" * 50)
        print(f"\nTotal: {len(amazon18_dataset2fullname)} datasets\n")
        exit(0)

    # Download datasets
    success_count = 0
    total_count = len(args.dataset)

    for dataset in args.dataset:
        if download_amazon_dataset(args):
            success_count += 1

    # Summary
    print("\n" + "="*60)
    print(f"Download Summary: {success_count}/{total_count} datasets completed")
    print("="*60 + "\n")

    if success_count < total_count:
        exit(1)
