# imajin.py

**imajin.py** is a search tool for `.epub` and `.mokuro` files, designed to help you find example sentences in your Japanese books and manga.

---


## Features

- Search across unencrypted `.epub` (ebooks) and `.mokuro` (manga) files
- Supports fuzzy matching for Japanese conjugations (optional)
- Supports searching individual words or phrases
- Structured output: text, markdown, or JSON
- Recursively search directories of books and manga
- Clean highlighted snippets showing surrounding context

---

## Installation

1. Install Python 3.9+ if not already installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)
- [lxml](https://pypi.org/project/lxml/)
- [mecab-python3](https://github.com/SamuraiT/mecab-python3) (optional; fuzzy matching will be disabled if MeCab is not available)

---

## Usage

```bash
python imajin.py [options] <search_word> <file_or_directory>
```

### Positional Arguments
| Argument | Description |
|:---------|:------------|
| `<search_word>` | The word or phrase you want to find. |
| `<file_or_directory>` | A single `.epub` or `.mokuro` file, or a directory containing them. |

### Options
| Option | Description |
|:-------|:------------|
| `--no-fuzzy` | Disable fuzzy matching (only exact matches). |
| `-r`, `--recursive` | Recursively search subdirectories if a directory is specified. |
| `--format {text,markdown,json}` | Choose output format (default: `text`). |
| `-h`, `--help` | Show help message and exit. |

---

## Examples

Search for the word "慌ただしい" inside your book collection:

```bash
python imajin.py 慌ただしい ./books/
```

Find exact matches only, searching all subdirectories:

```bash
python imajin.py --no-fuzzy -r 慌ただしい ./novel-library/
```

Get markdown-formatted results:

```bash
python imajin.py 慌ただしい ./books/ --format markdown
```

Save the results in a JSON file for further processing:

```bash
python imajin.py 慌ただしい ./manga-collection/ --format json > results.json
```

---

## Saving Results

To save your search results to a file, redirect the output:

```bash
python imajin.py 慌ただしい ./books/ --format markdown > examples.md
```

This method works for all output formats (text, markdown, or JSON).

---

## Notes

- If MeCab is not installed, fuzzy matching will be automatically disabled.

---

## License

This project is released under the [CC0 1.0 Universal Public Domain Dedication](LICENSE.txt).
