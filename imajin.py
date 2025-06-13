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

A search tool for .epub and .mokuro files supporting exact and smart matching.
- Supports structured searching of chapters (EPUB) and pages (Mokuro).
- Highlights matched snippets with optional smart Japanese language matching using MeCab.
- Accepts a single file or an entire directory (with optional recursive search).
"""

import argparse
import logging
import zipfile
import re
import sys
import json
import warnings
from json import JSONDecodeError
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Iterable, Any, TypeVar, Callable
from typing_extensions import ParamSpec
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

__version__ = "v1.4.0b1"

sys.stdout.reconfigure(encoding='utf-8') # type: ignore
sys.stderr.reconfigure(encoding='utf-8') # type: ignore

P = ParamSpec("P")
R = TypeVar("R")

def safe_tag(tag: Any) -> Optional[Tag]:
    return tag if isinstance(tag, Tag) else None

def safe_str_attr(tag: Tag, attr: str) -> Optional[str]:
    val = tag.get(attr)
    return val if isinstance(val, str) else None

class VolumeLoadError(Exception):
    """Custom exception for errors loading Epub or Mokuro volumes."""
    pass

class ParseError(Exception):
    """Custom exception for errors parsing Epub or Mokuro volumes."""
    pass

class TokenizationError(Exception):
    """Custom exception for errors tokenizing a string."""
    pass

class UnsupportedFormatError(Exception):
    """Custom exception for an invalid file format."""
    pass

# Abstract classes for Volume and Chapter
class Volume(ABC):
    """Abstract base class representing a searchable volume."""
    @abstractmethod
    def get_sections(self) -> List["Section"]:
        """Return all sections in the volume."""
        pass

    @abstractmethod
    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        pass

    @abstractmethod
    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        pass

class Section(ABC):
    """Abstract base class representing a searchable section of text."""
    @abstractmethod
    def get_text(self) -> str:
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
        with zipfile.ZipFile(epub_path, 'r') as zf:
            self.manifest, self.spine, self.toc, self.rootfile_path = self._parse_epub_metadata(zf)
            self.sections, self.total_text_length = self._extract_spine_documents(zf)
        self.chapters = self._locate_chapter_names()
        logging.debug(f"Volume {self.get_filename()}: Chapters mapped:\n{self.chapters}\n")

    def _extract_spine_documents(self, zip_file: zipfile.ZipFile) -> tuple[dict[str, 'EpubSection'], int]:
        sections = {}
        total_text_length = 0

        for idref in self.spine:
            href = self.manifest.get(idref)
            if not href:
                continue
            full_path = PurePosixPath(self.rootfile_path).parent / href
            try:
                content_raw = zip_file.read(str(full_path))
            except KeyError:
                logging.warning(f"Volume {self.get_filename()}: Missing document '{full_path}'\n\tResults may be incomplete")
                continue
            content = BeautifulSoup(content_raw, 'html.parser')
            raw_text = content.get_text()
            sections[idref] = EpubSection(self, href, raw_text, content)
            logging.debug(f"Volume {self.get_filename()}: Loaded section {href} with {len(raw_text)} characters")
            total_text_length += len(raw_text)

        return sections, total_text_length

    def _locate_chapter_names(self) -> dict[str, str]:
        current_chapter = "Front Matter"
        chapters = {}
        # Check toc for chapter names
        if self.toc:
            for id in self.spine:
                if section := self.sections.get(id):
                    if id in self.toc:
                        current_chapter = self.toc[id]
                    chapters[id] = current_chapter
                    section.set_chapter_name(current_chapter)
            if chapters:
                logging.debug(f"Volume {self.get_filename()}: Chapters found by index")
        else:
            for id in self.spine:
                if section := self.sections.get(id):
                    if chapter_name := section.search_for_chapter_name():
                        current_chapter = chapter_name
                    chapters[id] = current_chapter
                    section.set_chapter_name(current_chapter)
            if chapters:
                logging.debug(f"Volume {self.get_filename()}: Chapters found by text search")
        return chapters

    def _parse_epub_metadata(self, zip_file: zipfile.ZipFile) -> tuple[dict[str, str], tuple[str, ...], dict[str, str], str]:
        """Parse EPUB metadata and return manifest, spine, labels, and rootfile path."""
        rootfile_path = self._extract_rootfile_path(zip_file)
        opf = self._load_opf(zip_file, rootfile_path)
        manifest = self._parse_manifest(opf)
        spine = self._parse_spine(opf)
        toc = self._parse_toc(opf, manifest, rootfile_path, zip_file)
        return manifest, spine, toc, rootfile_path

    def _extract_rootfile_path(self, zip_file: zipfile.ZipFile) -> str:
        try:
            container = BeautifulSoup(zip_file.read('META-INF/container.xml'), 'xml')
            rootfile_element = safe_tag(container.find('rootfile'))
            path = rootfile_element['full-path'] if rootfile_element else None
            if not isinstance(path, str):
                raise ParseError("Invalid rootfile path")
            return path
        except (KeyError, ParseError):
            raise VolumeLoadError(f"Volume {self.get_filename()}: Malformed container.xml: missing rootfile")

    def _load_opf(self, zip_file: zipfile.ZipFile, rootfile_path: str) -> Tag:
        try:
            return BeautifulSoup(zip_file.read(rootfile_path), 'xml')
        except KeyError:
            raise VolumeLoadError(f"Volume {self.get_filename()}: Missing rootfile '{rootfile_path}'")

    def _parse_manifest(self, opf: Tag) -> dict[str, str]:
        """Parse the manifest from the OPF file.

        The manifest is a list of all the files in the ebook, referenced by an
        id value. The spine uses the same id values.
        """
        manifest = {}
        for item in opf.find_all('item'):
            if tag := safe_tag(item):
                mid = safe_str_attr(tag, 'id')
                href = safe_str_attr(tag, 'href')
                if mid and href:
                    manifest[mid] = href
                else:
                    logging.warning(f"Volume {self.get_filename()}: Malformed manifest item.")
        return manifest

    def _parse_spine(self, opf: Tag) -> tuple[str, ...]:
        """Parse the spine from the OPF file.

        The spine lists out the display order for the ebook. By cross-referencing
        the ids in the spine with the files from the manifest, we can get a list
        of files in display order.
        """
        return tuple(
            ref for item in opf.find_all('itemref')
            if (tag := safe_tag(item)) and (ref := safe_str_attr(tag, 'idref'))
        )

    def _parse_toc(self, opf: Tag, manifest: dict[str, str], rootfile_path: str, zip_file: zipfile.ZipFile) -> dict[str, str]:
        """Parse the chapter listing from the toc file.
        
        There are two separate toc styles for epubs, the older ncx and newer
        nav files. While EPub 3+ books should use nav files, it's not 100%
        guaranteed, so just try to get a nav toc and fall back to ncx.

        Returns:
            dict[str,str]: A mapping of the manifest id to the chapter name, if available.
        """
        inverse_manifest = {v: k for k, v in manifest.items()}
        content_path = str(PurePosixPath(rootfile_path).parent)

        if nav_item := safe_tag(opf.find('item', attrs={'properties': 'nav'})):
            nav_path = safe_str_attr(nav_item, 'href')
            if nav_path:
                toc = self._parse_nav_toc(nav_path, content_path, zip_file, inverse_manifest)
                logging.debug(f"Parsed nav chapters:\n{toc}")
                return toc

        if ncx_item := safe_tag(opf.find('item', attrs={'media-type': 'application/x-dtbncx+xml'})):
            ncx_path = safe_str_attr(ncx_item, 'href')
            if ncx_path:
                toc = self._parse_ncx_toc(ncx_path, content_path, zip_file, inverse_manifest)
                logging.debug(f"Parsed ncx chapters:\n{toc}")
                return toc

        return {}

    def _parse_nav_toc(self, nav_path: str, content_path: str, zip_file: zipfile.ZipFile, inverse_manifest: dict[str, str]) -> dict[str, str]:
        toc: dict[str, str] = {}
        nav_full_path = PurePosixPath(content_path) / nav_path
        try:
            nav = BeautifulSoup(zip_file.read(str(nav_full_path)), 'xml')
        except (KeyError):
            logging.warning(f"Volume {self.get_filename()}: Index nav file {nav_path} missing or invalid")
            return toc

        try:
            toc_nav = safe_tag(nav.find('nav', attrs={'epub:type': 'toc'}))
            if not toc_nav:
                raise ParseError("Failed toc_nav parse")

            ol = safe_tag(toc_nav.find('ol'))
            if not ol:
                raise ParseError("Failed ol parse")
        except (ParseError):
            logging.warning(f"Volume {self.get_filename()}: Index nav file {nav_path} invalid")
            return toc

        for li in ol.find_all('li', recursive=False):
            if li_tag := safe_tag(li):
                if a_tag := safe_tag(li_tag.find('a', recursive=True)):
                    href: Optional[str] = safe_str_attr(a_tag, 'href')
                    label: str = a_tag.get_text(strip=True)

                    if (href is None):
                        logging.warning(f"Volume {self.get_filename()}: Nav chapter skipped. Chapter data may be incomplete")
                        continue

                    if '#' in href:
                        file_part, _ = href.split('#', 1)
                    else:
                        file_part = href

                    if mid := inverse_manifest.get(file_part):
                        toc[mid] = label
                    else:
                        logging.warning(f"Volume {self.get_filename()}: Chapter file {file_part} missing from manifest. Chapter data may be incomplete")
                        continue

        return toc

    def _parse_ncx_toc(self, ncx_path: str, content_path: str, zip_file: zipfile.ZipFile, inverse_manifest: dict[str, str]) -> dict[str, str]:
        toc: dict[str, str] = {}
        ncx_full_path = PurePosixPath(content_path) / ncx_path
        try:
            ncx = BeautifulSoup(zip_file.read(str(ncx_full_path)), 'xml')
        except (KeyError):
            logging.warning(f"Volume {self.get_filename()}: Index ncx file {ncx_path} missing or invalid")
            return toc

        for navpoint in ncx.find_all('navPoint'):
            if navpoint_tag := safe_tag(navpoint):
                href: Optional[str] = None
                if content := safe_tag(navpoint_tag.find('content')):
                    href = safe_str_attr(content, 'src')

                label_tag = getattr(navpoint_tag, "navLabel", None)
                label = label_tag.text.strip() if label_tag and hasattr(label_tag, "text") else None

                if (href is None) or (label is None):
                    logging.debug(f"Volume {self.get_filename()}: Invalid navpoint label. Chapter data may be incomplete")
                    continue

                if '#' in href:
                    file_part, _ = href.split('#', 1)
                else:
                    file_part = href

                if mid := inverse_manifest.get(file_part):
                    toc[mid] = label
                else:
                    logging.warning(f"Volume {self.get_filename()}: Chapter file {file_part} missing from manifest. Chapter data may be incomplete")
                    continue
        logging.debug(f"Volume {self.get_filename()}: Index ncx file found")

        return toc

    def get_sections(self) -> List['Section']:
        """Return all sections in the volume."""
        return list(self.sections.values())

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return Path(self.epub_path).name

class EpubSection(Section):
    """Section class representing a chapter inside an EPUB volume."""

    def __init__(self, volume: EpubVolume, href: str, raw_text: str, content: Tag):
        self.volume: EpubVolume = volume
        self.href: str = href
        self.raw_text: str = raw_text
        self.content: Tag = content
        self._cached_chapter_name: Optional[str] = None

    def get_text(self) -> str:
        """Return the raw text of the section."""
        return self.raw_text

    def search_for_chapter_name(self) -> Optional[str]:
        """Search this section for something that looks like a chapter name"""
        body = safe_tag(self.content.find('body'))
        if body and (first_text_tag := self._get_first_innermost_block(body)):
            if self.looks_like_chapter_name(first_text_tag, body):
                first_text = ' '.join(first_text_tag.stripped_strings).strip()
                return first_text
        return None

    def _get_first_innermost_block(self, body: Tag) -> Optional[Tag]:
        """Return the first innermost block-level tag with meaningful text."""
        block_tags = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

        for item in body.descendants:
            if tag := safe_tag(item):
                if tag.name not in block_tags:
                    continue

                pure_text = ''.join(tag.stripped_strings).strip()
                if not pure_text:
                    continue

                # Check if tag contains any nested block-level tags with content
                for child in tag.descendants:
                    if (child_tag := safe_tag(child)) and (child_tag.name in block_tags):
                        child_text = ''.join(child_tag.stripped_strings).strip()
                        if child_text:
                            break
                else:
                    return tag  # Return the full tag, not the text

        return None

    @staticmethod
    def looks_like_chapter_name(tag: Tag, body: Tag) -> bool:
        """Return True if the tag or any of its descendants has a unique class or is a heading tag."""
        heading_tags = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

        # Count all class frequencies in the body
        class_counts: dict[str, int] = {}
        for node in body.find_all(True):
            if (node_tag := safe_tag(node)) and (cls := node_tag.get_attribute_list('class')):
                for c in cls:
                    if c_str := c:
                        class_counts[c] = class_counts.get(c_str, 0) + 1
        unique_classes = {c for c, count in class_counts.items() if count == 1}

        # Check the tag and its descendants
        for node in [tag] + list(tag.descendants):
            if isinstance(node, Tag):
                # Condition 1: tag is a heading
                if node.name in heading_tags:
                    return True
                # Condition 2: tag has a unique class
                cls = node.get_attribute_list('class')
                if cls and any(c in unique_classes for c in cls):
                    return True

        return False

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
        return make_snippet(self.raw_text, idx=idx, length=length)

    def get_display_type(self) -> str:
        return "Chapter"

    def set_chapter_name(self, str: str) -> None:
        self._cached_chapter_name = str

    def get_display_name(self) -> str:
        if self._cached_chapter_name is not None:
            return self._cached_chapter_name
        else:
            return "Unknown"

class MokuroVolume(Volume):
    """Volume class for handling Mokuro (manga JSON) files."""

    def __init__(self, path: str):
        self.path: str = path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except (FileNotFoundError, PermissionError, OSError, JSONDecodeError) as e:
            raise VolumeLoadError(f"Failed to load Mokuro file '{path}': {e}") from e
        self.pages: List[Section] = [MokuroPage(self, i, page_data) for i, page_data in enumerate(self.data['pages'])]
        self.total_text_length: int = sum(len(page.get_text()) for page in self.pages)

    def get_sections(self) -> List[Section]:
        """Return all sections in the volume."""
        return self.pages

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return Path(self.path).name


class MokuroPage(Section):
    """Section class representing a page inside a Mokuro volume."""

    def __init__(self, volume: MokuroVolume, index: int, page_data: dict[str, Any]):
        self.volume: MokuroVolume = volume
        self.index = index
        self.page_data = page_data
        self.page_number = index + 1
        self.text: str = self._build_text()

    def _build_text(self) -> str:
        blocks = []
        for block in self.page_data.get('blocks', []):
            lines = block.get('lines', [])
            block_text = ' '.join(lines)
            blocks.append(block_text)
        return '\t'.join(blocks)

    def get_text(self) -> str:
        """Return the raw text of the section."""
        return self.text

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
        return make_snippet(self.text, idx=idx, length=length)
        
    def get_display_type(self) -> str:
        return "Page"

    def get_display_name(self) -> str:
        return f"{self.page_number}"

class SrtVolume(Volume):
    """Volume class for handling srt subtitle files"""

    def __init__(self, path: str):
        self.path: str = path
        self.entries: List[Section] = []
        try:
            with open(self.path, 'r', encoding='utf-8-sig') as f:
                self._parse_srt_file(f)
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise VolumeLoadError(f"Failed to load srt file '{path}': {e}") from e
        self.total_text_length: int = sum(len(entry.get_text()) for entry in self.entries)

    def _parse_srt_file(self, file: Iterable[str]) -> None:
        block = []
        for line in file:
            line = line.strip()
            if line:
                block.append(line)
            else:
                self._parse_entry(block)
                block = []
        if block:
            self._parse_entry(block)

    def _parse_entry(self, block: list[str]) -> None:
        if len(block) < 2:
            return
        times = block[1]
        start_str, end_str = times.split(' --> ')
        text = '\t'.join(block[2:])
        self.entries.append(SrtEntry(start_str, end_str, text))

    def get_sections(self) -> List[Section]:
        """Return all sections in the volume."""
        return self.entries

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return Path(self.path).name

class SrtEntry(Section):
    """Section class representing an srt entry"""

    def __init__(self, start: str, end: str, text: str):
        self.start = start
        self.end = end
        self.text = text

    def get_text(self) -> str:
        return self.text

    def get_display_type(self) -> str:
        return "Timestamp"

    def get_display_name(self) -> str:
        return self.start

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
       return make_snippet(self.text, idx=idx, length=length)

class AssVolume(Volume):
    """Volume class for handling ass subtitle files"""

    def __init__(self, path: str):
        self.path: str = path
        self.lines: List[Section] = []
        try:
            with open(self.path, 'r', encoding='utf-8-sig') as f:
                self._parse_ass_file(f)
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise VolumeLoadError(f"Failed to load srt file '{path}': {e}") from e
        self.total_text_length: int = sum(len(line.get_text()) for line in self.lines)

    def _parse_ass_file(self, file: Iterable[str]) -> None:
        in_events: bool = False
        for line in file:
            line = line.strip()

            if line.startswith("[Events]"):
                in_events = True
                continue
            elif line.startswith("["):
                in_events = False
                continue

            if in_events:
                if line.startswith("Format:"):
                    self.dialogue_format = self._parse_format(line)
                    logging.debug(f"Dialogue format: {self.dialogue_format}")
                elif line.startswith("Dialogue:"):
                    self._parse_dialogue(line)

    def _parse_format(self, line: str) -> list[str]:
        # Example: Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        _, fields = line.split(":", 1)
        return [field.strip().lower() for field in fields.split(",")]

    def _parse_dialogue(self, line: str) -> None:
        # Example: Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,こんにちは。
        _, data = line.split(":", 1)
        values = [v.strip() for v in data.split(",", len(self.dialogue_format) - 1)]
        field_map = dict(zip(self.dialogue_format, values))

        start = field_map.get("start")
        end = field_map.get("end")
        text = self._clean_ass_text(field_map.get("text", ""))

        if (start is not None) and (end is not None) and (text is not None):
            self.lines.append(AssLine(start, end, text))
        else:
            logging.warning(f"Volume {self.get_filename()}: Skipped malformed dialogue line:\n{str}")

    def _clean_ass_text(self, text: str) -> str:
        # Remove override tags like {\i1}, {\pos(...)} etc., and handle line breaks
        text = re.sub(r"{.*?}", "", text)
        return text.replace(r"\N", "\n").replace(r"\n", "\t").strip()

    def get_sections(self) -> List[Section]:
        """Return all sections in the volume."""
        return self.lines

    def get_total_length(self) -> int:
        """Return total text length of the volume."""
        return self.total_text_length

    def get_filename(self) -> str:
        """Return the base filename of the volume."""
        return Path(self.path).name

class AssLine(Section):
    """Section class representing an ass dialogue line"""

    def __init__(self, start: str, end: str, text: str):
        self.start = start
        self.end = end
        self.text = text

    def get_text(self) -> str:
        return self.text

    def get_display_type(self) -> str:
        return "Timestamp"

    def get_display_name(self) -> str:
        return self.start

    def extract_snippet(self, idx: int, length: int) -> tuple[str, int, int]:
       return make_snippet(self.text, idx=idx, length=length) 

class OutputManager:
    def __init__(self, mode: str = 'text'):
        self.mode = mode

    def output_global_header(self) -> None:
        if self.mode == 'json':
            print("[")

    def output_volume_results(self, results: List['Result'], first: bool = False) -> None:
        """Output all results for a single volume"""
        if self.mode == 'json':
            if not first:
                print(",")
            for result in results:
                self._print_indented(json.dumps(result.to_dict(mode=self.mode), ensure_ascii=False, indent=2))
        else:
            self._output_text_or_markdown(results)

    def output_global_footer(self) -> None:
        if self.mode == 'json':
            print("\n]")

    def _print_indented(self, text: str) -> None:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            end = '\n' if i < len(lines) - 1 else ''
            print('  ' + line, end=end)

    def _output_text_or_markdown(self, results: List['Result']) -> None:
        first = True

        for result in results:
            if first:
                first = False
                self._print_volume_header(result.volume.get_filename())
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

    def to_dict(self, mode: str = 'text') -> dict[str, Any]:
        """Return a dictionary version of the result for output."""
        snippet_text, match_start, match_end = self.snippet
        matched_text = snippet_text[match_start:match_end]
        highlighted = self._highlight_snippet(snippet_text, match_start, match_end, mode)

        location_field = None
        if isinstance(self.volume, EpubVolume):  # Only EPUBs get location
            location = (self.absolute_position / self.volume.get_total_length()) * 100
            location_field = f"{location:.2f}%"

        result: dict[str, Any] = {}

        result["Volume"] = self.volume.get_filename()

        result[self.section.get_display_type()] = self.section.get_display_name()

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

# Cache the function's results since they should be universal
@lru_cache(maxsize=1)
def detect_feature_separator_and_index(mecab: Any) -> tuple[str, int]:
    """
    Detect the separator character and index for the base form feature.
    Returns a tuple (separator, index).
    """
    parsed = mecab.parse("食べ")

    logging.debug(f"MeCab 食べ parse:\n{parsed}")

    if parsed is None:
        return ',', 6

    for line in parsed.splitlines():
        if line == 'EOS' or not line.strip():
            continue

        parts = line.split('\t', maxsplit=1)
        if len(parts) != 2:
            continue
        surface, feature_str = parts

        # Try both separators
        for sep in [',', '\t']:
            fields = feature_str.split(sep)
            for i, field in enumerate(fields):
                if field == "食べる":
                    sepString = 'tab' if sep == '\t' else sep
                    logging.debug(f"Identified MeCab dictionary separator: {sepString}")
                    logging.debug(f"Identified MeCab base index: {i}")
                    return sep, i

    logging.warning("Unknown MeCab dictionary configuration. Fuzzy search may not function correctly")
    # Default fallback if detection fails
    return ',', 6

def get_base_form(line: str, mecab: Any) -> tuple[str, str]:
    """
    Extract surface and base form from a MeCab line, using detected format.
    """

    parts = line.split('\t', maxsplit=1)
    if len(parts) != 2:
        return line, line

    surface, feature_str = parts
    separator, base_form_index = detect_feature_separator_and_index(mecab)

    fields = feature_str.split(separator)
    if (
        len(fields) > base_form_index
        and fields[base_form_index] not in ('', '*')
    ):
        return surface, fields[base_form_index]

    return surface, surface

def memoize_search_term(func: Callable[P, R], search_term: str) -> Callable[P, R]:
    """Cache the results of fetching mecab base for the user's search term"""
    cache = {}

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not args:
            return func(*args, **kwargs)

        word = args[0]
        if word == search_term:
            if word not in cache:
                result = cache[word] = func(*args, **kwargs)
                if isinstance(result, Iterable):
                    logging.debug(f"Tokenized search term: {', '.join(' | '.join(map(str, t)) for t in result)}")
            return cache[word]
        return func(*args, **kwargs)

    return wrapper

def tokenize_with_positions(text: str, mecab: Any) -> Optional[List[tuple[str, str, int]]]:
    """Tokenize a text string into a tuple of (surface, base, found_pos)"""
    parsed_text = mecab.parse(text)
    if parsed_text is None:
        return None

    text_tokens = []
    cursor = 0  # Real character position inside text

    for line in parsed_text.splitlines():
        if line == 'EOS' or line.strip() == '':
            continue
        surface, base = get_base_form(line, mecab)
        # Search for the unprocessed text in the original corpus
        # Workaround to get an accurate text location
        found_pos = text.find(surface, cursor)
        if found_pos == -1:
            raise TokenizationError(f"Tokenization failed finding string '{cursor}' in string '{surface}'")
        text_tokens.append((surface, base, found_pos))
        cursor = found_pos + len(surface)
    return text_tokens

def smart_match(text: str, search_term: str, mecab: Any) -> tuple[int, int]:
    """Fuzzy match by comparing base form tokens of search_term and text."""

    # Tokenize the search and fetch bases
    search_bases = None
    if search_tokens := tokenize_with_positions(search_term, mecab):
        search_bases = [base for (_, base, _) in search_tokens]

    if not search_bases:
        return -1, 0

    # Tokenize the corpus text to compare bases
    text_tokens = tokenize_with_positions(text, mecab)

    if not text_tokens:
        return -1, 0

    search_length = len(search_bases)

    # Do a sliding window search matching the bases against each other
    for i in range(len(text_tokens) - search_length + 1):
        window_bases = [base for (_, base, _) in text_tokens[i:i + search_length]]
        if window_bases == search_bases:
            start_pos = text_tokens[i][2]
            last_surface, _, last_start = text_tokens[i + search_length - 1]
            end_pos = last_start + len(last_surface)
            return start_pos, end_pos - start_pos

    return -1, 0


def find_matches(text: str, search_term: str, mecab: Optional[Any], match: str = "both") -> List[tuple[int, int]]:
    """Find all exact and smart matches in a given text."""
    matches = []
    start = 0
    use_exact = match == "both" or match == "exact"
    use_smart = match == "both" or match == "smart"

    # Do exact string search
    if use_exact:
        while start < len(text):
            idx = text.find(search_term, start)
            if idx == -1:
                break
            matches.append((idx, len(search_term)))
            start = idx + len(search_term)

    # Do smart string search
    if use_smart and mecab:
        start = 0
        while start < len(text):
            smart_idx, smart_len = smart_match(text[start:], search_term, mecab)
            if smart_idx == -1:
                break
            absolute_idx = start + smart_idx
            matches.append((absolute_idx, smart_len))
            start = absolute_idx + smart_len

    return sorted(set(matches))

def find_sentence_bounds(text: str, idx: int, length: int, max_expand: int = 300) -> tuple[int, int]:
    """Return start and end indices for a sentence-sized snippet around the match."""
    sentence_endings = set('。．？！!?\n')
    text_length = len(text)

    # Find the start
    snippet_start = idx
    for _ in range(max_expand):
        if snippet_start <= 0 or text[snippet_start] in sentence_endings:
            snippet_start += 1 if snippet_start < len(text) else 0
            break
        snippet_start -= 1
    snippet_start = max(0, snippet_start)

    # Find the end
    snippet_end = idx + length
    for _ in range(max_expand):
        if snippet_end >= text_length or text[snippet_end] in sentence_endings:
            snippet_end += 1 if snippet_end < text_length else 0
            break
        snippet_end += 1
    snippet_end = min(text_length, snippet_end)

    return snippet_start, snippet_end


def adjust_match_indices(unstripped: str, match_start: int, match_end: int) -> tuple[str, int, int]:
    """Strip the snippet and adjust match indices accordingly."""
    leading_ws = len(unstripped) - len(unstripped.lstrip())
    stripped = unstripped.strip()
    adjusted_start = max(0, match_start - leading_ws)
    adjusted_end = max(adjusted_start, match_end - leading_ws)
    adjusted_start = min(adjusted_start, len(stripped))
    adjusted_end = min(adjusted_end, len(stripped))
    return stripped, adjusted_start, adjusted_end


def make_snippet(text: str, idx: int = 0, length: int = 0, max_expand: int = 300) -> tuple[str, int, int]:
    """Generate a sentence-based snippet from text containing the match at idx."""
    if idx == -1:
        return '', 0, 0

    start, end = find_sentence_bounds(text, idx, length, max_expand)
    unstripped = text[start:end]
    match_start = idx - start
    match_end = match_start + length
    return adjust_match_indices(unstripped, match_start, match_end)

def search_volume(volume: Volume, search_term: str, mecab: Optional[Any], use_fuzzy: bool = True, match: str = "both") -> List[Result]:
    """Search a volume and return a list of match result dictionaries."""
    results = []
    current_text_position = 0

    logging.info(f"Searching volume {volume.get_filename()}")

    for section in volume.get_sections():
        matches = find_matches(text=section.get_text(), search_term=search_term, mecab=mecab, match=match)

        for idx, match_len in matches:
            snippet = section.extract_snippet(idx, match_len)
            absolute_position = current_text_position + idx
            result = Result(volume, section, snippet, absolute_position)
            results.append(result)

        current_text_position += len(section.get_text())

    return results

def safe_search_volume(path: str, search_term: str, mecab: Optional[Any], match: str = "both", use_fuzzy: bool = True) -> List[Result]:
    volume: Optional[Volume] = None
    if path.endswith('.epub'):
        volume = EpubVolume(path)
    elif path.endswith('.mokuro'):
        volume = MokuroVolume(path)
    elif path.endswith('.srt'):
        volume = SrtVolume(path)
    elif path.endswith('.ass'):
        volume = AssVolume(path)
    else:
        raise UnsupportedFormatError(f"{path} is not in a supported format")
    return search_volume(volume=volume, search_term=search_term, mecab=mecab, match=match)

def get_cli_args() -> argparse.Namespace:
    """Parse command-line arguments using argparse with grouped help."""
    parser = argparse.ArgumentParser(
        description="imajin.py — Search inside EPUB (.epub) and Mokuro (.mokuro) files with optional smart Japanese matching.",
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
        "-r", "--recursive",
        action="store_true",
        help="Recursively search subdirectories if a directory is specified."
    )

    options.add_argument(
        "--format",
        choices=["text", "markdown", "md", "json"],
        default="text",
        help="Output format style."
    )

    options.add_argument(
        "-v", "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="Increase output verbosity (e.g. -v, -vv, -vvv)"
    )

    options.add_argument(
        "--match",
        choices=["both", "exact", "smart"],
        default="both",
        help=(
            "Matching strategy: "
            "'exact' matches only the literal string, "
            "'smart' uses MeCab tokenization and deconjugation, "
            "'both' combines the results from both search strategies."
        )
    )

    options.add_argument(
        "--no-fuzzy",
        action="store_false",
        dest="use_fuzzy",
        help="(DEPRECATED) Use --mach=exact instead. Will be removed in a future release"
    )

    args = parser.parse_args()

    if args.format == "md":
        args.format = "markdown"

    if not args.use_fuzzy and args.match == "both":
        args.match = "exact"

    if not args.use_fuzzy:
        print("WARNING: --no-fuzzy is deprecated and will be removed in a future release. Use --match=exact instead.", file=sys.stderr)

    return args

def get_args(
    *,
    search_word: Optional[str] = None,
    target_path: Optional[str] = None,
    recursive: Optional[bool] = None,
    output_format: Optional[str] = None,
    verbosity: Optional[int] = None,
    use_exact: Optional[bool] = None,
    match: Optional[str] = None
) -> argparse.Namespace:
    if search_word is None or target_path is None:
        return get_cli_args()
    return argparse.Namespace(
        search_word = search_word,
        target_path = target_path,
        recursive = False if recursive is None else recursive,
        format="text" if output_format is None else output_format,
        verbosity=0 if verbosity is None else verbosity,
        match="both" if match is None else match
    )

def resolve_paths(target_path: str, recursive: bool, supported_formats: tuple[str, ...]) -> tuple[str, ...]:
    target = Path(target_path)
    if target.is_dir():
        try:
            if recursive:
                return tuple(
                    str(f)
                    for f in target.rglob("*")
                    if f.is_file() and f.suffix in supported_formats
                )
            else:
                return tuple(
                    str(f)
                    for f in target.iterdir()
                    if f.is_file() and f.suffix in supported_formats
                )
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise VolumeLoadError(f"Failed accessing directory '{target_path}': {e}") from e

    elif target.is_file() and target.suffix in supported_formats:
        return tuple(target_path)
    else:
        raise ValueError(f"'{target_path}' is not a valid file or directory.")

def imajin_search(args: argparse.Namespace) -> None:
    try:
        global tokenize_with_positions

        # Map verbosity to logging levels
        log_level = {
            0: logging.WARNING,   # default
            1: logging.INFO,      # -v
            2: logging.DEBUG,     # -vv
        }.get(args.verbosity, logging.DEBUG)  # -vvv or more

        logging.basicConfig(
            level=log_level,
            format="%(levelname)s: %(message)s"
        )

        logging.info(f"Search term: {args.search_word}")
        logging.info(f"Target path: {args.target_path}")
        logging.info(f"Recursive: {args.recursive}, Match: {args.match}, Format: {args.format}")

        mecab = None

        if args.use_fuzzy:
            try:
                import MeCab  # type: ignore
                mecab = MeCab.Tagger()
                # Monkey patch get_base_form to cache calling mecab on the user's search
                tokenize_with_positions = memoize_search_term(tokenize_with_positions, args.search_word)
            except (ImportError, RuntimeError):
                logging.info("MeCab not found. Fuzzy matching for Japanese conjugations will be disabled.")
            else:
                logging.info("MeCab enabled")

        supported_formats = ('.epub', '.mokuro', '.srt', '.ass')

        try:
            paths = resolve_paths(args.target_path, args.recursive, supported_formats)
        except (VolumeLoadError, ValueError) as e:
            logging.error(str(e))
            sys.exit(1)

        output_manager = OutputManager(mode=args.format)

        with ThreadPoolExecutor() as executor:
            try:
                futures = []
                for path in paths:
                    if path.endswith(supported_formats):
                        futures.append(executor.submit(safe_search_volume, path, args.search_word, mecab, args.match))

                output_manager.output_global_header()
                first = True
                # Collect results
                for future in as_completed(futures):
                    try:
                        volume_results = future.result()
                    except (UnsupportedFormatError, VolumeLoadError) as e:
                        logging.error(f"{e}")
                        continue
                    if volume_results:
                        output_manager.output_volume_results(volume_results, first=first)
                        first = False
                output_manager.output_global_footer()
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Terminating...")
        sys.exit(0)

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is not None and not isinstance(args, argparse.Namespace):
        raise TypeError("Expected argparse.Namespace or None for `main()` argument.")
    if args is None:
        args = get_args()
    imajin_search(args)

if __name__ == '__main__':
    main()