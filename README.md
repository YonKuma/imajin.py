# imajin.py

**imajin.py** is a search tool for `.epub`, `.mokuro`, `.srt` and `.ass` files, designed to help you find example sentences in your Japanese media.

---

## Index

- [Features](#features)
- [Installation](#installation)  
  - [Install with pip (preferred method)](#install-with-pip-preferred-method)  
  - [Download and install script](#download-and-install-script)  
  - [Dependencies](#dependencies)
- [Usage](#usage)  
  - [Positional Arguments](#positional-arguments)  
  - [Options](#options)
- [Examples](#examples)  
  - [Installed](#installed)  
  - [As a script](#as-a-script)
- [Saving Results](#saving-results)
- [Notes](#notes)
- [License](#license)

---

## Features

- Search across unencrypted `.epub` (ebooks), `.mokuro` (manga), and `.srt` and `.ass` (subtitled) files
- Supports smart matching for Japanese conjugations (optional)
- Supports searching individual words or phrases
- Structured output: text, markdown, or JSON
- Recursively search directories of books, manga, and subtitles
- Clean highlighted snippets showing surrounding context

---

## Installation

### Install with pip (preferred method)

Recommended because it will be easier to update going forward, especially if you prefer to have the script installed.

1. Install Python 3.9+ if not allready installed
2. Install imajin using pip:

    ```bash
    pip install imajin
    ```

    You should be all set up to use `imajin` directly from the command line. 

### Download and install script

Not recommended, because it's harder to update.

1. Install Python 3.9+ if not already installed.
2. Download the source folder, either by downloading the Source code from the [Latest Release](https://github.com/YonKuma/imajin.py/releases/latest) or using `git clone https://github.com/YonKuma/imajin.py.git`
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   You can now use `imajin.py` as a python script. For example, if you're in the source directory, you can use `python imajin.py -r 蹴散らす ~/Documents/Manga`, but you'll need to use a qualified path to the files you want to search
4. (Optional) You can install imajin by copying the script to somewhere in your PATH and making it executable

    ```bash
    sudo cp imajin.py /usr/local/bin/imajin
    sudo chmod +x /usr/local/bin/imajin
    ```

### Dependencies

Dependencies:
- [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)
- [lxml](https://pypi.org/project/lxml/)
- [mecab-python3](https://github.com/SamuraiT/mecab-python3) (optional; smart matching will be disabled if MeCab is not available)
- unidic-lite (optional; used for smart matching. Other dictionaries should work but are harder to install)

---

## Usage

Installed:

```bash
imajin [options] <search_word> <file_or_directory>
```

As a script:

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
| `--match {exact, smart, both}` | Choose whether to use exact match only search (faster), smart search only (slower, less false positives, matches different conjucations), or the union of both (default: `both`). |
| `-r`, `--recursive` | Recursively search subdirectories if a directory is specified. |
| `--format {text,markdown,md,json}` | Choose output format (default: `text`). |
| `-h`, `--help` | Show help message and exit. |

---

## Examples

### Installed

Search for the word "慌ただしい" inside your book collection:

```bash
imajin 慌ただしい ./books/
```

Find exact matches only, searching all subdirectories:

```bash
imajin --match=exact -r 慌ただしい ./novel-library/
```

Get markdown-formatted results:

```bash
imajin 慌ただしい ./books/ --format md
```

Save the results in a JSON file for further processing:

```bash
imajin 慌ただしい ./manga-collection/ --format json > results.json
```

### As a script

Search for the word "慌ただしい" inside your book collection:

```bash
python imajin.py 慌ただしい ./books/
```

Find smart matches only, searching all subdirectories:

```bash
python imajin.py --match=smart -r 慌ただしい ./novel-library/
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
imajin 慌ただしい ./books/ --format md > examples.md
```

This method works for all output formats (text, markdown/md, or JSON).

---

## Notes

- If MeCab is not installed, smart matching will be automatically disabled.

---

## License

This project is released under the [CC0 1.0 Universal Public Domain Dedication](LICENSE.txt).
