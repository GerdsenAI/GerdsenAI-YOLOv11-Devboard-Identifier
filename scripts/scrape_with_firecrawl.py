#!/usr/bin/env python3
"""
scrape_with_firecrawl.py - Use Firecrawl API to scrape JS-rendered product pages

This script uses Firecrawl to extract product images from retailer sites
that require JavaScript rendering (Adafruit, SparkFun, Amazon, etc.)

Usage:
    export FIRECRAWL_API_KEY=your-api-key
    python scrape_with_firecrawl.py
"""

import os
import sys
import json
import requests
import hashlib
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

# Firecrawl API
FIRECRAWL_API_KEY = os.environ.get('FIRECRAWL_API_KEY', 'REMOVED_API_KEY')
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "../dataset/raw/teensy_41"

# URLs to scrape for Teensy 4.1 images
SCRAPE_URLS = [
    "https://www.adafruit.com/product/4622",
    "https://www.sparkfun.com/products/16771",
    "https://www.pjrc.com/store/teensy41.html",
    # Add more as needed
]

# Minimum image size
MIN_WIDTH = 200
MIN_HEIGHT = 150


def firecrawl_scrape(url: str) -> dict:
    """Scrape a URL using Firecrawl API."""
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "url": url,
        "formats": ["markdown", "html"],
        "onlyMainContent": False,
        "includeTags": ["img"],
        "waitFor": 3000,  # Wait for JS to render
    }

    print(f"Scraping: {url}")
    response = requests.post(
        f"{FIRECRAWL_API_URL}/scrape",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return None


def extract_image_urls(firecrawl_response: dict) -> list:
    """Extract image URLs from Firecrawl response."""
    images = []

    if not firecrawl_response or 'data' not in firecrawl_response:
        return images

    data = firecrawl_response['data']

    # Check for images in metadata
    if 'metadata' in data:
        meta = data['metadata']
        if 'ogImage' in meta:
            images.append(meta['ogImage'])
        if 'image' in meta:
            images.append(meta['image'])

    # Parse HTML for image tags
    html = data.get('html', '')

    # Simple regex-free extraction of img src
    import re
    img_pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    for match in re.finditer(img_pattern, html, re.IGNORECASE):
        src = match.group(1)
        if src and not src.startswith('data:'):
            images.append(src)

    # Also check srcset
    srcset_pattern = r'srcset=["\']([^"\']+)["\']'
    for match in re.finditer(srcset_pattern, html, re.IGNORECASE):
        srcset = match.group(1)
        for part in srcset.split(','):
            src = part.strip().split()[0]
            if src and not src.startswith('data:'):
                images.append(src)

    # Filter for likely product images
    filtered = []
    skip_keywords = ['logo', 'icon', 'avatar', 'social', 'facebook', 'twitter',
                     'cart', 'button', 'arrow', 'pixel', 'spacer', '.svg', '.gif']

    for img in images:
        img_lower = img.lower()
        if not any(kw in img_lower for kw in skip_keywords):
            # Prefer larger image versions
            if '970x728' in img or '1200x900' in img or '/large/' in img or '/big/' in img:
                filtered.insert(0, img)  # Prioritize
            else:
                filtered.append(img)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for img in filtered:
        if img not in seen:
            seen.add(img)
            unique.append(img)

    return unique


def download_image(url: str, output_dir: Path, downloaded_hashes: set) -> bool:
    """Download and save an image if valid."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'image' not in content_type:
            return False

        image_data = response.content

        # Validate image
        try:
            img = Image.open(BytesIO(image_data))
            width, height = img.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                print(f"  Skipped (too small {width}x{height}): {url[:60]}...")
                return False
            if len(image_data) < 5000:  # < 5KB
                return False
            img.verify()
        except Exception:
            return False

        # Check for duplicates
        img_hash = hashlib.md5(image_data).hexdigest()
        if img_hash in downloaded_hashes:
            print(f"  Skipped (duplicate): {url[:60]}...")
            return False

        downloaded_hashes.add(img_hash)

        # Save
        ext = '.jpg'
        if '.png' in url.lower():
            ext = '.png'
        filename = f"teensy_41_fc_{img_hash[:8]}{ext}"
        filepath = output_dir / filename

        with open(filepath, 'wb') as f:
            f.write(image_data)

        print(f"  Downloaded: {filename}")
        return True

    except Exception as e:
        print(f"  Failed: {url[:50]}... - {e}")
        return False


def load_existing_hashes(output_dir: Path) -> set:
    """Load hashes of existing images."""
    hashes = set()
    if output_dir.exists():
        for img_file in output_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    with open(img_file, 'rb') as f:
                        hashes.add(hashlib.md5(f.read()).hexdigest())
                except:
                    pass
    return hashes


def main():
    print("=" * 60)
    print("Firecrawl Image Scraper for Teensy 4.1")
    print("=" * 60)

    if not FIRECRAWL_API_KEY:
        print("Error: FIRECRAWL_API_KEY not set")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_hashes = load_existing_hashes(OUTPUT_DIR)
    print(f"Existing images: {len(downloaded_hashes)}")

    total_downloaded = 0

    for url in SCRAPE_URLS:
        print(f"\n{'='*40}")
        result = firecrawl_scrape(url)

        if result:
            images = extract_image_urls(result)
            print(f"  Found {len(images)} candidate images")

            for img_url in images[:15]:  # Limit per page
                if download_image(img_url, OUTPUT_DIR, downloaded_hashes):
                    total_downloaded += 1
                time.sleep(0.5)

        time.sleep(2)  # Rate limiting

    print(f"\n{'='*60}")
    print(f"COMPLETE: Downloaded {total_downloaded} new images")
    print(f"Total images in folder: {len(list(OUTPUT_DIR.glob('*.[jp][pn][g]')))}")
    print("=" * 60)


if __name__ == '__main__':
    main()
