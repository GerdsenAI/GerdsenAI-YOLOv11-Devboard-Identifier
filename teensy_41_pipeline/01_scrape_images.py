#!/usr/bin/env python3
"""
01_scrape_images.py - Teensy 4.1 Image Scraper

Focused scraper for Teensy 4.1 development board images.
Sources: PJRC (official), Adafruit, SparkFun, DigiKey, Mouser

Usage:
    python 01_scrape_images.py
    python 01_scrape_images.py --max-images 100
"""

import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Teensy 4.1 specific URLs
TEENSY_SOURCES = {
    'pjrc_store': 'https://www.pjrc.com/store/teensy41.html',
    'pjrc_main': 'https://www.pjrc.com/teensy/',
    'adafruit': 'https://www.adafruit.com/product/4622',
    'sparkfun': 'https://www.sparkfun.com/products/16771',
}

# Direct image URLs known to work
DIRECT_IMAGES = [
    'https://www.pjrc.com/store/teensy41_1.jpg',
    'https://www.pjrc.com/store/teensy41_2.jpg',
    'https://www.pjrc.com/store/teensy41_3.jpg',
    'https://www.pjrc.com/store/teensy41_4.jpg',
    'https://www.pjrc.com/store/teensy41_angle.jpg',
    'https://www.pjrc.com/store/teensy41_front.jpg',
    'https://www.pjrc.com/store/teensy41_back.jpg',
]

# Keywords to SKIP (pinouts, diagrams, etc.)
SKIP_KEYWORDS = [
    'pinout', 'diagram', 'schematic', 'logo', 'icon', 'banner',
    'button', 'cart', 'checkout', 'social', 'facebook', 'twitter',
    'pinterest', 'instagram', 'youtube', 'badge', 'rating', 'star',
    'arrow', 'nav', 'menu', 'header', 'footer', 'advertisement',
    'ad_', 'ads_', 'sponsor', 'promo', 'thumbnail_small'
]

# Keywords that indicate GOOD product images
GOOD_KEYWORDS = [
    'teensy', 'product', 'main', 'hero', 'gallery', 'large',
    'zoom', 'detail', 'front', 'back', 'angle', 'board'
]

OUTPUT_DIR = Path(__file__).parent / 'data' / 'raw'
MIN_WIDTH = 200
MIN_HEIGHT = 150
MIN_SIZE_BYTES = 5000


def get_image_hash(content: bytes) -> str:
    """Generate MD5 hash for deduplication."""
    return hashlib.md5(content).hexdigest()[:12]


def is_valid_image(content: bytes) -> bool:
    """Check if content is a valid image meeting size requirements."""
    try:
        img = Image.open(BytesIO(content))
        w, h = img.size
        return w >= MIN_WIDTH and h >= MIN_HEIGHT and len(content) >= MIN_SIZE_BYTES
    except:
        return False


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped based on keywords."""
    url_lower = url.lower()
    return any(kw in url_lower for kw in SKIP_KEYWORDS)


def is_likely_product_image(url: str) -> bool:
    """Check if URL is likely a product image."""
    url_lower = url.lower()
    return any(kw in url_lower for kw in GOOD_KEYWORDS)


def download_image(url: str, session: requests.Session) -> tuple:
    """Download image and return (content, hash) or (None, None)."""
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and is_valid_image(resp.content):
            return resp.content, get_image_hash(resp.content)
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
    return None, None


def scrape_page_images(url: str, session: requests.Session) -> list:
    """Scrape all image URLs from a page."""
    images = []
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return images

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Find all img tags
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                full_url = urljoin(url, src)
                if not should_skip_url(full_url):
                    images.append(full_url)

        # Find high-res versions in srcset
        for img in soup.find_all('img', srcset=True):
            srcset = img['srcset']
            for part in srcset.split(','):
                src = part.strip().split()[0]
                full_url = urljoin(url, src)
                if not should_skip_url(full_url):
                    images.append(full_url)

        # Find links to images
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(href.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                full_url = urljoin(url, href)
                if not should_skip_url(full_url):
                    images.append(full_url)

    except Exception as e:
        logger.debug(f"Failed to scrape {url}: {e}")

    return list(set(images))  # Remove duplicates


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Scrape Teensy 4.1 images')
    parser.add_argument('--max-images', type=int, default=50, help='Maximum images to download')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })

    downloaded_hashes = set()
    count = 0

    logger.info("=== Teensy 4.1 Image Scraper ===")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target: {args.max_images} images")

    # Phase 1: Direct URLs
    logger.info("\n[Phase 1] Downloading direct URLs...")
    for url in DIRECT_IMAGES:
        if count >= args.max_images:
            break
        content, img_hash = download_image(url, session)
        if content and img_hash not in downloaded_hashes:
            filename = output_dir / f"teensy_41_{img_hash}.jpg"
            with open(filename, 'wb') as f:
                f.write(content)
            downloaded_hashes.add(img_hash)
            count += 1
            logger.info(f"  Downloaded: {filename.name}")
        time.sleep(0.5)

    # Phase 2: Scrape source pages
    logger.info("\n[Phase 2] Scraping source pages...")
    for name, url in TEENSY_SOURCES.items():
        if count >= args.max_images:
            break
        logger.info(f"  Scraping {name}...")
        images = scrape_page_images(url, session)

        # Prioritize likely product images
        images.sort(key=lambda x: (is_likely_product_image(x), 'teensy' in x.lower()), reverse=True)

        for img_url in images[:20]:  # Limit per source
            if count >= args.max_images:
                break
            content, img_hash = download_image(img_url, session)
            if content and img_hash not in downloaded_hashes:
                filename = output_dir / f"teensy_41_{img_hash}.jpg"
                with open(filename, 'wb') as f:
                    f.write(content)
                downloaded_hashes.add(img_hash)
                count += 1
                logger.info(f"    Downloaded: {filename.name}")
            time.sleep(0.3)

    print("\n" + "=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Downloaded: {count} images")
    print(f"Output: {output_dir}")
    print("=" * 50)
    print("\nNext: Review images and remove any pinouts/diagrams")
    print("Then run: python 02_manual_capture.py")


if __name__ == '__main__':
    main()
