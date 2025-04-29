#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# imajin.py by YonKuma
#
# To the extent possible under law, the person who associated CC0 with
# imajin.py has waived all copyright and related or neighboring rights
# to imajin.py.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
# -----------------------------------------------------------------------------

"""
imajin.py

A search tool for .epub and .mokuro files supporting exact and fuzzy matching.
- Supports structured searching of chapters (EPUB) and pages (Mokuro).
- Highlights matched snippets with optional fuzzy Japanese language matching using MeCab.
- Accepts a single file or an entire directory (with optional recursive search).
"""

import argparse
import zipfile
import posixpath
import os
import re
import sys
import json
from collections import defaultdict
from typing import List, Optional, Any
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

try:
    import MeCab
    mecab = MeCab.Tagger()
except (ImportError, RuntimeError) as e:
    mecab = None
    print("[WARNING] MeCab not found. Fuzzy matching for Japanese conjugations will be disabled.", file=sys.stderr)

class VolumeLoadError(Exception):
    """Custom exception for errors loading Epub or Mokuro volumes."""
    pass

# Abstract classes for Volume and Chapter
class Volume(ABC):
    """Abstract base class representing a searchable volume."""
    @abstractmethod
    def get_sections(self):
        """Return all sections in the volume."""
        pass

    @abstractmethod
    def get_total_length(self):
        """Return total text length of the volume."""
        pass

    @abstractmethod
    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        pass

class Section(ABC):
    """Abstract base class representing a searchable section of text."""
    @abstractmethod
    def get_text(self):
        """Return the raw text of the section."""
        pass

    @abstractmethod
    def get_display_type(self) -> str:
        """Return a name identifying this section (chapter title, page number, etc.)."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Return a name identifying this section (chapter title, page number, etc.)."""
        pass

    @abstractmethod
    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
        """Extract a snippet centered around a match. Returns (snippet_text, match_start_in_snippet, match_end_in_snippet)."""
        pass


# Concrete classes for EPUB
class EpubVolume(Volume):
    """Volume class for handling EPUB files."""

    def __init__(self, epub_path: str):
        self.epub_path: str = epub_path
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                self.manifest, self.spine, self.id_to_label, self.rootfile_path = parse_epub_metadata(zf)
                self.chapters, self.total_text_length = self._extract_spine_documents(zf)
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
            raise VolumeLoadError(f"Failed to load EPUB '{path}': {e}") from e

    def _extract_spine_documents(self, zip_file: zipfile.ZipFile) -> tuple[List['EpubChapter'], int]:
        chapters = []
        total_text_length = 0

        for idref in self.spine:
            href = self.manifest.get(idref)
            if not href:
                continue
            full_path = posixpath.join(posixpath.dirname(self.rootfile_path), href)
            try:
                content_raw = zip_file.read(full_path)
            except KeyError as e:
                print(f"[WARNING] Missing document '{full_path}' in EPUB: {self.get_filename()}\n\tResults may be incomplete", file=sys.stderr)
                continue
            content = BeautifulSoup(content_raw, 'html.parser')
            raw_text = content.get_text()
            chapters.append(EpubChapter(self, href, raw_text, content))
            total_text_length += len(raw_text)

        return chapters, total_text_length

    def get_sections(self) -> List['EpubChapter']:
        """Return all sections in the volume."""
        return self.chapters

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return os.path.basename(self.epub_path)

class EpubChapter(Section):
    """Section class representing a chapter inside an EPUB volume."""

    def __init__(self, volume: EpubVolume, href: str, raw_text: str, content: BeautifulSoup):
        self.volume: EpubVolume = volume
        self.href: str = href
        self.raw_text: str = raw_text
        self.content: BeautifulSoup = content
        self._cached_chapter_name: Optional[str] = None

    def get_text(self) -> str:
        """Return the raw text of the section."""
        return self.raw_text

    def find_chapter_name(self) -> str:
        if self._cached_chapter_name is not None:
            return self._cached_chapter_name

        id_to_label = getattr(self.volume, 'id_to_label', {})
        spine_docs = self.volume.get_sections()
        idx_in_spine = spine_docs.index(self)

        if idx_in_spine > 0:
            prev_doc = spine_docs[idx_in_spine - 1]
            prev_content = prev_doc.content
            if prev_content:
                prev_body = prev_content.find('body')
                if prev_body:
                    full_text = ''.join(prev_body.stripped_strings).strip()
                    for tag in prev_body.descendants:
                        if tag.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            pure_text = ''.join(tag.stripped_strings).strip()
                            if pure_text and self.is_innermost_block(tag):
                                if pure_text == full_text:
                                    self._cached_chapter_name = pure_text
                                    return pure_text

        headings = self.content.find_all(re.compile('^h[1-6]$'))
        if headings:
            self._cached_chapter_name = headings[0].get_text(strip=True)
            return self._cached_chapter_name

        href_base = posixpath.basename(self.href).split('#')[0]
        if (href_base, None) in id_to_label:
            self._cached_chapter_name = id_to_label[(href_base, None)]
            return self._cached_chapter_name

        body = self.content.find('body')
        if body:
            first_text = extract_first_text_block(body)
            if first_text:
                self._cached_chapter_name = first_text
                return self._cached_chapter_name

        self._cached_chapter_name = 'Unknown'
        return 'Unknown'

    @staticmethod
    def is_innermost_block(tag: BeautifulSoup) -> bool:
        block_tags = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        for child in tag.descendants:
            if child.name in block_tags:
                child_text = ''.join(child.stripped_strings).strip()
                if child_text:
                    return False
        return True

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
        return make_snippet(self.raw_text, idx=idx, length=length)

    def get_display_type(self) -> str:
        return "Chapter"

    def get_display_name(self) -> str:
        return self.find_chapter_name()

class MokuroVolume(Volume):
    """Volume class for handling Mokuro (manga JSON) files."""

    def __init__(self, path: str):
        self.path: str = path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data: dict = json.load(f)
        except (FileNotFoundError, PermissionError, OSError, JSONDecodeError) as e:
            raise VolumeLoadError(f"Failed to load Mokuro file '{path}': {e}") from e
        self.pages: List[MokuroPage] = [MokuroPage(self, i, page_data) for i, page_data in enumerate(self.data['pages'])]
        self.total_text_length: int = sum(len(page.get_text()) for page in self.pages)

    def get_sections(self) -> List[Section]:
        """Return all sections in the volume."""
        return self.pages

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return os.path.basename(self.path)


class MokuroPage(Section):
    """Section class representing a page inside a Mokuro volume."""

    def __init__(self, volume: MokuroVolume, index: int, page_data: dict):
        self.volume: MokuroVolume = volume
        self.index: int = index
        self.page_data: dict = page_data
        self.page_number: int = index + 1
        self.text: str = self._build_text()

    def _build_text(self) -> str:
        blocks = []
        for block in self.page_data.get('blocks', []):
            lines = block.get('lines', [])
            block_text = ' '.join(lines)
            blocks.append(block_text)
        return '\n'.join(blocks)

    def get_text(self) -> str:
        """Return the raw text of the section."""
        return self.text

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
        return make_snippet(self.text, idx=idx, length=length)
        
    def get_display_type(self) -> str:
        return "Page"

    def get_display_name(self) -> str:
        return f"{self.page_number}"

class OutputManager:
    def __init__(self, mode: str = 'text'):
        self.mode = mode

    def output_results(self, results: List['Result']) -> None:
        """Output the given results immediately."""
        if self.mode == 'json':
            output = [result.to_dict(mode=self.mode) for result in results]
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            self._output_text_or_markdown(results)

    def _output_text_or_markdown(self, results: List['Result']) -> None:
        grouped = self._group_results_by_volume(results)

        for volume, matches in grouped.items():
            self._print_volume_header(volume)

            for result in matches:
                result_dict = result.to_dict(mode=self.mode)
                for field_name, field_value in result_dict.items():
                    if field_name == 'Volume':
                        continue
                    if field_name == 'Snippet' and isinstance(field_value, dict):
                        print(f"**Snippet**: {field_value['highlighted']}  " if self.mode == 'markdown'
                              else f"Snippet: {field_value['highlighted']}")
                    else:
                        print(f"**{field_name}**: {field_value}  " if self.mode == 'markdown'
                              else f"{field_name}: {field_value}")
                print()

    def _group_results_by_volume(self, results: List[dict[str, str]]) -> dict:
        from collections import defaultdict
        grouped = defaultdict(list)
        for result in results:
            volume = result.volume.get_filename()
            grouped[volume].append(result)
        return grouped

    def _print_volume_header(self, volume_name: str) -> None:
        """Print the header for a volume."""
        if self.mode == 'text':
            print(f"========== {volume_name} ==========")
            print()
        elif self.mode == 'markdown':
            print(f"# {volume_name}")
            print()

class Result:
    def __init__(self, volume: 'Volume', section: 'Section', snippet: tuple[str, int, int], absolute_position: int):
        self.volume = volume
        self.section = section
        self.snippet = snippet  # (snippet_text, match_start, match_end)
        self.absolute_position = absolute_position

    def to_dict(self, mode: str = 'text') -> dict[str, str | dict[str, str | int]]:
        """Return a dictionary version of the result for output."""
        snippet_text, match_start, match_end = self.snippet
        matched_text = snippet_text[match_start:match_end]
        highlighted = self._highlight_snippet(snippet_text, match_start, match_end, mode)

        location_field = None
        if isinstance(self.volume, EpubVolume):  # Only EPUBs get location
            location = (self.absolute_position / self.volume.get_total_length()) * 100
            location_field = f"{location:.2f}%"

        result = {}

        result["Volume"] = self.volume.get_filename();

        result[self.section.get_display_type()] = self.section.get_display_name();

        if location_field:
            result["Location"] = location_field

        result["Snippet"] = {
            "text": snippet_text,
            "start": match_start,
            "length": match_end - match_start,
            "match": matched_text,
            "highlighted": highlighted,
        }

        return result


    def _highlight_snippet(self, text: str, start: int, end: int, mode: str) -> str:
        """Apply appropriate highlight styling depending on output mode."""
        if mode == 'markdown' or mode == 'json':
            return text[:start] + "**" + text[start:end] + "**" + text[end:]
        elif mode == 'text':
            return text[:start] + "\033[1m" + text[start:end] + "\033[0m" + text[end:]
        else:
            # Default for unknown mode — no fancy highlighting
            return text



# Helper functions
def fuzzy_match(text: str, search_word: str) -> tuple[int, int]:
    """Fuzzy match by comparing base form tokens of search_word and text."""
    if mecab is None:
        return -1, 0

    parsed_search = mecab.parse(search_word)
    if parsed_search is None:
        return -1, 0

    search_bases = []
    for line in parsed_search.split('\n'):
        if line == 'EOS' or line.strip() == '' or '\t' not in line:
            continue
        surface, features = line.split('\t')
        features_list = features.split(',')
        base = features_list[6] if len(features_list) > 6 else surface
        if base == '*':
            return -1, 0
        search_bases.append(base)

    if not search_bases:
        return -1, 0

    parsed_text = mecab.parse(text)
    if parsed_text is None:
        return -1, 0

    text_tokens = []
    cursor = 0  # Real character position inside text

    for line in parsed_text.split('\n'):
        if line == 'EOS' or line.strip() == '' or '\t' not in line:
            continue
        surface, features = line.split('\t')
        features_list = features.split(',')
        base = features_list[6] if len(features_list) > 6 else surface

        # Find actual surface in text starting from current cursor
        found_pos = text.find(surface, cursor)
        if found_pos == -1:
            return -1, 0

        text_tokens.append((surface, base, found_pos))
        cursor = found_pos + len(surface)

    if not text_tokens:
        return -1, 0

    search_length = len(search_bases)

    for i in range(len(text_tokens) - search_length + 1):
        window_bases = [base for (_, base, _) in text_tokens[i:i + search_length]]
        if window_bases == search_bases:
            start_pos = text_tokens[i][2]
            last_surface, _, last_start = text_tokens[i + search_length - 1]
            end_pos = last_start + len(last_surface)
            return start_pos, end_pos - start_pos

    return -1, 0

def find_matches(text: str, search_word: str, use_fuzzy: bool = True) -> List[tuple[int, int]]:
    """Find all exact and fuzzy matches in a given text."""
    matches = []
    start = 0
    while start < len(text):
        idx = text.find(search_word, start)
        if idx == -1:
            break
        matches.append((idx, len(search_word)))
        start = idx + len(search_word)

    if use_fuzzy and mecab:
        start = 0
        while start < len(text):
            fuzzy_idx, fuzzy_len = fuzzy_match(text[start:], search_word)
            if fuzzy_idx == -1:
                break
            absolute_idx = start + fuzzy_idx
            matches.append((absolute_idx, fuzzy_len))
            start = absolute_idx + fuzzy_len

    return sorted(set(matches))

def make_snippet(text: str, idx: int = 0, length: int = 0, max_expand: int = 300) -> tuple[str, int, int]:
    """Create a snippet including the full sentence containing the match.

    max_expand limits how much to expand backward/forward to find sentence boundaries.
    """
    if idx == -1:
        return '', 0, 0

    sentence_endings = set('。．？！!?')
    text_length = len(text)

    # Search backward for start of the sentence
    snippet_start = idx
    back_steps = 0
    while snippet_start > 0 and back_steps < max_expand:
        if text[snippet_start] in sentence_endings:
            snippet_start += 1  # Move past the ending punctuation
            break
        snippet_start -= 1
        back_steps += 1
    snippet_start = max(0, snippet_start)

    # Search forward for end of the sentence
    snippet_end = idx + length
    forward_steps = 0
    while snippet_end < text_length and forward_steps < max_expand:
        if text[snippet_end] in sentence_endings:
            snippet_end += 1  # Include the ending punctuation
            break
        snippet_end += 1
        forward_steps += 1
    snippet_end = min(text_length, snippet_end)

    snippet = text[snippet_start:snippet_end]
    match_start_in_snippet = idx - snippet_start
    match_end_in_snippet = match_start_in_snippet + length

    return snippet, match_start_in_snippet, match_end_in_snippet

def extract_first_text_block(body: BeautifulSoup) -> Optional[str]:
    """Extract the first meaningful block of text from HTML body."""
    classes = {}
    for tag in body.find_all(True):
        cls = tag.get('class')
        if cls:
            for c in cls:
                classes[c] = classes.get(c, 0) + 1
    unique_classes = {c for c, count in classes.items() if count == 1}
    for tag in body.descendants:
        if tag.name and tag.string and tag.string.strip():
            parent = tag.parent
            cls = parent.get('class')
            if cls and any(c in unique_classes for c in cls):
                return tag.string.strip()
            else:
                return None
    return None

def parse_epub_metadata(zip_file: zipfile.ZipFile) -> tuple[dict, List[str], dict, str]:
    """Parse EPUB metadata and return manifest, spine, labels, and rootfile path."""
    try:
        container = BeautifulSoup(zip_file.read('META-INF/container.xml'), 'xml')
    except KeyError as e:
        raise VolumeLoadError("Missing META-INF/container.xml in EPUB archive") from e

    try:
        rootfile_path = container.find('rootfile')['full-path']
    except (AttributeError, TypeError, KeyError) as e:
        raise VolumeLoadError("Malformed container.xml: could not find rootfile path") from e
    
    try:
        opf = BeautifulSoup(zip_file.read(rootfile_path), 'xml')
    except KeyError as e:
        raise VolumeLoadError(f"Missing rootfile '{rootfile_path}' in EPUB archive") from e

    manifest = {item['id']: item['href'] for item in opf.find_all('item')}
    spine = [item['idref'] for item in opf.find_all('itemref')]

    id_to_label = {}
    ncx_path = None
    for item in opf.find_all('item'):
        if item.get('media-type') == 'application/x-dtbncx+xml':
            ncx_path = item['href']
            break

    if ncx_path:
        ncx_full_path = posixpath.join(posixpath.dirname(rootfile_path), ncx_path)
        try:
            ncx = BeautifulSoup(zip_file.read(ncx_full_path), 'xml')
            for navpoint in ncx.find_all('navPoint'):
                src = navpoint.content['src']
                label = navpoint.navLabel.text.strip()
                if '#' in src:
                    file_part, frag = src.split('#', 1)
                    id_to_label[(posixpath.basename(file_part), frag)] = label
                else:
                    id_to_label[(posixpath.basename(src), None)] = label
        except (KeyError, AttributeError, TypeError):
            pass # Silently allow a failed ncx read

    return manifest, spine, id_to_label, rootfile_path

def search_volume(volume: Volume, search_word: str, use_fuzzy: bool = True) -> List[dict[str, str]]:
    """Search a volume and return a list of match result dictionaries."""
    results = []
    current_text_position = 0

    for section in volume.get_sections():
        matches = find_matches(section.get_text(), search_word, use_fuzzy)

        for idx, match_len in matches:
            snippet = section.extract_snippet(idx, match_len)
            absolute_position = current_text_position + idx
            result = Result(volume, section, snippet, absolute_position)
            results.append(result)

        current_text_position += len(section.get_text())

    return results

def safe_search_volume(path, search_word, use_fuzzy):
    try:
        if path.endswith('.epub'):
            volume = EpubVolume(path)
        elif path.endswith('.mokuro'):
            volume = MokuroVolume(path)
        else:
            return None
        return search_volume(volume, search_word, use_fuzzy=use_fuzzy)
    except VolumeLoadError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return None

def parse_args() -> tuple[str, str, bool, bool, str]:
    """Parse command-line arguments using argparse with grouped help."""
    parser = argparse.ArgumentParser(
        description="imajin.py — Search inside EPUB (.epub) and Mokuro (.mokuro) files with optional fuzzy Japanese matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional arguments
    parser.add_argument(
        "search_word",
        help="The word or phrase to search for."
    )
    parser.add_argument(
        "target_path",
        help="A single file (.epub or .mokuro) or a directory containing such files."
    )

    # Optional flags
    options = parser.add_argument_group("Options")

    options.add_argument(
        "--no-fuzzy",
        action="store_false",
        dest="use_fuzzy",
        help="Disable fuzzy matching (only exact matches)."
    )

    options.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search subdirectories if a directory is specified."
    )

    options.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format style."
    )

    args = parser.parse_args()

    return args.search_word, args.target_path, args.use_fuzzy, args.recursive, args.format


if __name__ == '__main__':
    search_word, target_path, use_fuzzy, recursive, output_format = parse_args()

    try:
        if os.path.isdir(target_path):
            if recursive:
                paths = []
                for root, dirs, files in os.walk(target_path):
                    for f in files:
                        if f.endswith(('.epub', '.mokuro')):
                            paths.append(os.path.join(root, f))
            else:
                paths = [
                    os.path.join(target_path, f)
                    for f in os.listdir(target_path)
                    if f.endswith(('.epub', '.mokuro'))
                ]
        elif os.path.isfile(target_path) and target_path.endswith(('.epub', '.mokuro')):
            paths = [target_path]
        else:
            print(f"Error: '{target_path}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"[ERROR] Failed accessing path '{target_path}': {e}", file=sys.stderr)
        sys.exit(1)

    output_manager = OutputManager(mode=output_format)

    all_results = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for path in paths:
            if path.endswith('.epub') or path.endswith('.mokuro'):
                futures.append(executor.submit(safe_search_volume, path, search_word, use_fuzzy))

        # Collect results
        for future in futures:
            volume_results = future.result()
            all_results.extend(volume_results)

    output_manager.output_results(all_results)


