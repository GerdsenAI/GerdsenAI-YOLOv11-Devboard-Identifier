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
    python 01_scrape_images.py --boards teensy_41  # Scrape only Teensy 4.1
    python 01_scrape_images.py --boards teensy_41 --aggressive  # More thorough scraping
"""

import os
import sys
import yaml
import requests
import hashlib
import logging
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse, urljoin, quote_plus
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
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Minimum image dimensions
MIN_WIDTH = 200
MIN_HEIGHT = 150  # Allow slightly shorter images for board photos

# Maximum images per board (to prevent runaway scraping)
MAX_IMAGES_PER_BOARD = 50


class ImageScraper:
    def __init__(self, config_path: str, output_dir: str, aggressive: bool = False):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aggressive = aggressive

        # Track downloaded images to avoid duplicates
        self.downloaded_hashes = set()

        # Load existing hashes from output directory
        self._load_existing_hashes()

        # Attribution log
        self.attribution_log = []

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _load_config(self, config_path: str) -> dict:
        """Load board configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_existing_hashes(self):
        """Load hashes of existing images to avoid re-downloading."""
        for class_dir in self.output_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        try:
                            with open(img_file, 'rb') as f:
                                img_hash = hashlib.md5(f.read()).hexdigest()
                                self.downloaded_hashes.add(img_hash)
                        except:
                            pass
        logger.info(f"Loaded {len(self.downloaded_hashes)} existing image hashes")

    def _get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image deduplication."""
        return hashlib.md5(image_data).hexdigest()

    def _is_valid_image(self, image_data: bytes, url: str = "") -> bool:
        """Check if image meets minimum requirements."""
        try:
            img = Image.open(BytesIO(image_data))
            width, height = img.size

            # Check dimensions
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False

            # Skip very small files (likely icons or placeholders)
            if len(image_data) < 5000:  # Less than 5KB
                return False

            # Skip images that are likely icons/logos based on aspect ratio
            aspect = width / height if height > 0 else 0
            if aspect > 5 or aspect < 0.15:  # Too wide or too tall
                return False

            # Check if it's actually an image (not HTML error page)
            img.verify()

            return True
        except Exception as e:
            logger.debug(f"Invalid image: {e}")
            return False

    def _is_board_related(self, url: str, board_name: str) -> bool:
        """Check if URL likely contains a board image (not icon, logo, etc.)."""
        url_lower = url.lower()

        # Skip common non-product images
        skip_patterns = [
            'logo', 'icon', 'favicon', 'banner', 'button', 'arrow',
            'cart', 'checkout', 'payment', 'shipping', 'social',
            'facebook', 'twitter', 'instagram', 'youtube', 'pinterest',
            'avatar', 'profile', 'user', 'account', 'login',
            'advertisement', 'promo', 'sale', 'discount',
            'placeholder', 'loading', 'spinner', 'ajax',
            'thumb_', '_thumb', '_small', '_tiny', '_icon',
            '1x1', 'pixel', 'spacer', 'blank', 'transparent',
            '.svg', '.gif',  # Usually not product photos
        ]

        for pattern in skip_patterns:
            if pattern in url_lower:
                return False

        # Prefer URLs that mention the board or product
        board_keywords = board_name.lower().replace('_', ' ').split()
        url_has_keyword = any(kw in url_lower for kw in board_keywords)

        # Also accept generic product image patterns
        product_patterns = ['product', 'item', 'catalog', 'media', 'assets', 'parts']
        url_has_product_pattern = any(p in url_lower for p in product_patterns)

        return url_has_keyword or url_has_product_pattern or '/images/' in url_lower

    def _download_image(self, url: str, board_class: str, source_type: str) -> bool:
        """Download a single image and save it."""
        try:
            # Skip data URIs
            if url.startswith('data:'):
                return False

            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type and 'octet-stream' not in content_type:
                logger.debug(f"Not an image (content-type: {content_type}): {url}")
                return False

            image_data = response.content

            # Check if valid image
            if not self._is_valid_image(image_data, url):
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

            logger.info(f"Downloaded: {filename} from {source_type}")
            return True

        except requests.exceptions.RequestException as e:
            logger.debug(f"Request failed {url}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return False

    def _scrape_page_images(self, url: str, board_class: str) -> list:
        """Scrape all images from a webpage."""
        images = []
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all img tags with various src attributes
            for img in soup.find_all('img'):
                for attr in ['src', 'data-src', 'data-lazy-src', 'data-original',
                            'data-zoom-image', 'data-large-image', 'data-image']:
                    src = img.get(attr)
                    if src and not src.startswith('data:'):
                        img_url = urljoin(url, src)
                        if self._is_board_related(img_url, board_class):
                            images.append(img_url)

                # Check srcset for high-res images
                srcset = img.get('srcset', '')
                if srcset:
                    for part in srcset.split(','):
                        src_part = part.strip().split()[0]
                        if src_part and not src_part.startswith('data:'):
                            img_url = urljoin(url, src_part)
                            if self._is_board_related(img_url, board_class):
                                images.append(img_url)

            # Find images in <a> tags (often link to high-res versions)
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(ext in href.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    img_url = urljoin(url, href)
                    if self._is_board_related(img_url, board_class):
                        images.append(img_url)

            # Check for gallery/carousel data attributes
            for elem in soup.find_all(attrs={'data-gallery': True}):
                try:
                    gallery_data = json.loads(elem['data-gallery'])
                    if isinstance(gallery_data, list):
                        for item in gallery_data:
                            if isinstance(item, dict) and 'src' in item:
                                images.append(urljoin(url, item['src']))
                except:
                    pass

            # Look for JSON-LD product data
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Handle single product
                        if 'image' in data:
                            img = data['image']
                            if isinstance(img, str):
                                images.append(img)
                            elif isinstance(img, list):
                                images.extend(img)
                except:
                    pass

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        return unique_images

    def _search_bing_images(self, query: str, max_results: int = 20) -> list:
        """Search Bing for images (works without API key via scraping)."""
        images = []
        try:
            search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1"
            response = self.session.get(search_url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find image URLs in the page
                for img in soup.find_all('img', {'class': 'mimg'}):
                    src = img.get('src') or img.get('data-src')
                    if src and src.startswith('http'):
                        images.append(src)

                # Also try to find full-size image URLs in data attributes
                for a in soup.find_all('a', {'class': 'iusc'}):
                    m = a.get('m')
                    if m:
                        try:
                            data = json.loads(m)
                            if 'murl' in data:
                                images.append(data['murl'])
                        except:
                            pass

                logger.info(f"Bing search '{query}' found {len(images)} images")

        except Exception as e:
            logger.debug(f"Bing search failed: {e}")

        return images[:max_results]

    def _search_duckduckgo_images(self, query: str, max_results: int = 20) -> list:
        """Search DuckDuckGo for images."""
        images = []
        try:
            # DuckDuckGo image search requires a token first
            token_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            response = self.session.get(token_url, timeout=15)

            # Extract vqd token
            vqd_match = re.search(r'vqd=["\']([^"\']+)["\']', response.text)
            if vqd_match:
                vqd = vqd_match.group(1)

                # Now search for images
                img_url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(query)}&vqd={vqd}&f=,,,,,&p=1"
                img_response = self.session.get(img_url, timeout=15)

                if img_response.status_code == 200:
                    data = img_response.json()
                    for result in data.get('results', []):
                        if 'image' in result:
                            images.append(result['image'])

                logger.info(f"DuckDuckGo search '{query}' found {len(images)} images")

        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")

        return images[:max_results]

    def scrape_board(self, board_class: str, board_config: dict) -> int:
        """Scrape all images for a single board class."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping: {board_config.get('display_name', board_class)}")
        logger.info(f"{'='*60}")

        downloaded = 0

        # Count existing images
        class_dir = self.output_dir / board_class
        existing_count = len(list(class_dir.glob('*.[jp][pn][g]'))) if class_dir.exists() else 0
        logger.info(f"Existing images: {existing_count}")

        # 1. Download from direct image sources (highest priority)
        logger.info("Phase 1: Direct image sources...")
        for url in board_config.get('image_sources', []):
            if downloaded >= MAX_IMAGES_PER_BOARD:
                break
            if self._download_image(url, board_class, 'direct'):
                downloaded += 1
            time.sleep(0.5)  # Rate limiting

        # 2. Scrape manufacturer product page
        product_url = board_config.get('product_url')
        if product_url and downloaded < MAX_IMAGES_PER_BOARD:
            logger.info(f"Phase 2: Product page - {product_url}")
            images = self._scrape_page_images(product_url, board_class)
            logger.info(f"  Found {len(images)} candidate images")
            for img_url in images[:15]:
                if downloaded >= MAX_IMAGES_PER_BOARD:
                    break
                if self._download_image(img_url, board_class, 'product_page'):
                    downloaded += 1
                time.sleep(0.3)

        # 3. Scrape wiki/documentation page
        wiki_url = board_config.get('wiki_url')
        if wiki_url and downloaded < MAX_IMAGES_PER_BOARD:
            logger.info(f"Phase 3: Wiki page - {wiki_url}")
            images = self._scrape_page_images(wiki_url, board_class)
            logger.info(f"  Found {len(images)} candidate images")
            for img_url in images[:10]:
                if downloaded >= MAX_IMAGES_PER_BOARD:
                    break
                if self._download_image(img_url, board_class, 'wiki'):
                    downloaded += 1
                time.sleep(0.3)

        # 4. Scrape extra URLs (retailers, etc.)
        extra_urls = board_config.get('extra_scrape_urls', [])
        if extra_urls and downloaded < MAX_IMAGES_PER_BOARD:
            logger.info(f"Phase 4: Extra sources ({len(extra_urls)} URLs)...")
            for url in extra_urls:
                if downloaded >= MAX_IMAGES_PER_BOARD:
                    break
                logger.info(f"  Scraping: {url}")
                images = self._scrape_page_images(url, board_class)
                logger.info(f"    Found {len(images)} candidate images")
                for img_url in images[:10]:
                    if downloaded >= MAX_IMAGES_PER_BOARD:
                        break
                    if self._download_image(img_url, board_class, 'retailer'):
                        downloaded += 1
                    time.sleep(0.3)
                time.sleep(1)  # Rate limiting between sites

        # 5. Image search (if aggressive mode or low image count)
        search_queries = board_config.get('search_queries', [])
        if search_queries and (self.aggressive or (existing_count + downloaded) < 10):
            logger.info(f"Phase 5: Image search ({len(search_queries)} queries)...")
            for query in search_queries:
                if downloaded >= MAX_IMAGES_PER_BOARD:
                    break

                # Try Bing first (usually better results)
                images = self._search_bing_images(query, max_results=15)
                for img_url in images:
                    if downloaded >= MAX_IMAGES_PER_BOARD:
                        break
                    if self._download_image(img_url, board_class, 'bing_search'):
                        downloaded += 1
                    time.sleep(0.5)

                # Also try DuckDuckGo
                if downloaded < MAX_IMAGES_PER_BOARD:
                    images = self._search_duckduckgo_images(query, max_results=10)
                    for img_url in images:
                        if downloaded >= MAX_IMAGES_PER_BOARD:
                            break
                        if self._download_image(img_url, board_class, 'ddg_search'):
                            downloaded += 1
                        time.sleep(0.5)

                time.sleep(2)  # Rate limiting between searches

        logger.info(f"\nDownloaded {downloaded} new images for {board_class}")
        logger.info(f"Total images now: {existing_count + downloaded}")
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

            # Rate limiting between boards
            time.sleep(3)

        # Save attribution log
        log_path = self.output_dir / 'attribution.json'
        existing_log = []
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    existing_log = json.load(f)
            except:
                pass

        existing_log.extend(self.attribution_log)
        with open(log_path, 'w') as f:
            json.dump(existing_log, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE: Downloaded {total_downloaded} total images")
        logger.info(f"Attribution log: {log_path}")
        logger.info(f"{'='*60}")

        return total_downloaded


def main():
    parser = argparse.ArgumentParser(description='Scrape dev board images')
    parser.add_argument('--config', type=str, default='../config/boards.yaml',
                        help='Path to boards.yaml config file')
    parser.add_argument('--output', type=str, default='../dataset/raw',
                        help='Output directory for images')
    parser.add_argument('--boards', type=str, nargs='+',
                        help='Specific boards to scrape (default: all)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Enable aggressive scraping (image search)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    output_dir = script_dir / args.output

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    scraper = ImageScraper(str(config_path), str(output_dir), aggressive=args.aggressive)
    scraper.scrape_all(args.boards)


if __name__ == '__main__':
    main()
