#!/usr/bin/env python3
"""
01_scrape_images.py - Automated image collection for dev board dataset

This script:
1. Reads board configuration from boards.yaml
2. Scrapes images from manufacturer sites, retailers, and image searches
3. Downloads and organizes images by class
4. Logs sources for attribution

Usage:
    python 01_scrape_images.py --config ../config/boards.yaml --output ../dataset/raw
"""

import os
import sys
import yaml
import requests
import hashlib
import logging
import argparse
from pathlib import Path
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Minimum image dimensions
MIN_WIDTH = 200
MIN_HEIGHT = 200


class ImageScraper:
    def __init__(self, config_path: str, output_dir: str):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track downloaded images to avoid duplicates
        self.downloaded_hashes = set()

        # Attribution log
        self.attribution_log = []

    def _load_config(self, config_path: str) -> dict:
        """Load board configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image deduplication."""
        return hashlib.md5(image_data).hexdigest()

    def _is_valid_image(self, image_data: bytes) -> bool:
        """Check if image meets minimum requirements."""
        try:
            img = Image.open(BytesIO(image_data))
            width, height = img.size
            return width >= MIN_WIDTH and height >= MIN_HEIGHT
        except Exception:
            return False

    def _download_image(self, url: str, board_class: str, source_type: str) -> bool:
        """Download a single image and save it."""
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            image_data = response.content

            # Check if valid image
            if not self._is_valid_image(image_data):
                logger.debug(f"Image too small or invalid: {url}")
                return False

            # Check for duplicates
            img_hash = self._get_image_hash(image_data)
            if img_hash in self.downloaded_hashes:
                logger.debug(f"Duplicate image: {url}")
                return False

            self.downloaded_hashes.add(img_hash)

            # Create class directory
            class_dir = self.output_dir / board_class
            class_dir.mkdir(exist_ok=True)

            # Generate filename
            ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = '.jpg'
            filename = f"{board_class}_{img_hash[:8]}{ext}"
            filepath = class_dir / filename

            # Save image
            with open(filepath, 'wb') as f:
                f.write(image_data)

            # Log attribution
            self.attribution_log.append({
                'class': board_class,
                'filename': filename,
                'source_url': url,
                'source_type': source_type,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })

            logger.info(f"Downloaded: {filename}")
            return True

        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return False

    def _scrape_page_images(self, url: str, board_class: str) -> list:
        """Scrape all images from a webpage."""
        images = []
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all img tags
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Convert relative URLs to absolute
                    img_url = urljoin(url, src)
                    images.append(img_url)

            # Also check for background images in style attributes
            for elem in soup.find_all(style=True):
                style = elem['style']
                if 'url(' in style:
                    start = style.find('url(') + 4
                    end = style.find(')', start)
                    if end > start:
                        bg_url = style[start:end].strip('"\'')
                        images.append(urljoin(url, bg_url))

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")

        return images

    def _search_duckduckgo_images(self, query: str, max_results: int = 20) -> list:
        """Search DuckDuckGo for images (no API key required)."""
        # Note: This is a simplified approach. For production, use proper image APIs
        images = []
        try:
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&iax=images&ia=images"
            logger.info(f"Searching: {query}")
            # DuckDuckGo requires JavaScript, so this is limited
            # In production, use bing-image-downloader or google-images-download libraries
        except Exception as e:
            logger.debug(f"Search failed: {e}")

        return images

    def scrape_board(self, board_class: str, board_config: dict) -> int:
        """Scrape all images for a single board class."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Scraping: {board_config.get('display_name', board_class)}")
        logger.info(f"{'='*50}")

        downloaded = 0

        # 1. Download from direct image sources
        for url in board_config.get('image_sources', []):
            if self._download_image(url, board_class, 'direct'):
                downloaded += 1

        # 2. Scrape manufacturer product page
        product_url = board_config.get('product_url')
        if product_url:
            logger.info(f"Scraping product page: {product_url}")
            images = self._scrape_page_images(product_url, board_class)
            for img_url in images[:10]:  # Limit to 10 per page
                if self._download_image(img_url, board_class, 'product_page'):
                    downloaded += 1

        # 3. Scrape wiki/documentation page
        wiki_url = board_config.get('wiki_url')
        if wiki_url:
            logger.info(f"Scraping wiki page: {wiki_url}")
            images = self._scrape_page_images(wiki_url, board_class)
            for img_url in images[:10]:
                if self._download_image(img_url, board_class, 'wiki'):
                    downloaded += 1

        logger.info(f"Downloaded {downloaded} images for {board_class}")
        return downloaded

    def scrape_all(self, board_filter: list = None):
        """Scrape images for all boards in configuration."""
        boards = self.config.get('boards', {})
        total_downloaded = 0

        for board_class, board_config in boards.items():
            if board_filter and board_class not in board_filter:
                continue

            count = self.scrape_board(board_class, board_config)
            total_downloaded += count

            # Rate limiting
            time.sleep(2)

        # Save attribution log
        log_path = self.output_dir / 'attribution.json'
        with open(log_path, 'w') as f:
            json.dump(self.attribution_log, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info(f"COMPLETE: Downloaded {total_downloaded} total images")
        logger.info(f"Attribution log: {log_path}")
        logger.info(f"{'='*50}")

        return total_downloaded


def main():
    parser = argparse.ArgumentParser(description='Scrape dev board images')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml config file')
    parser.add_argument('--output', type=str, default='../dataset/raw',
                        help='Output directory for images')
    parser.add_argument('--boards', type=str, nargs='+',
                        help='Specific boards to scrape (default: all)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    output_dir = script_dir / args.output

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    scraper = ImageScraper(str(config_path), str(output_dir))
    scraper.scrape_all(args.boards)


if __name__ == '__main__':
    main()
