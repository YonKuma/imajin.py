# Changelog

## [Unreleased]

### Changed

-- **Breaking**: Main now takes an argparse Namespace object rather than individual values ([f66fcd3](https://github.com/YonKuma/imajin.py/commit/1d3568839451c9a8bdf1990a10405021a968c126))

### Added

- Search Srt and Ass subtitle files ([f9b93a3](https://github.com/YonKuma/imajin.py/commit/fb235ba23f583e62958646bbb113d48ac4a41fa8))
- Selet search mode: exact, smart, and both ([f66fcd3](https://github.com/YonKuma/imajin.py/commit/1d3568839451c9a8bdf1990a10405021a968c126))

### Fixed

- Force UTF-8 encoding ([6e613d5](https://github.com/YonKuma/imajin.py/commit/f66fcd3c767cdfeccc0315b75fe673afa78537b4))

## [v1.3.3](https://github.com/YonKuma/imajin.py/releases/tag/v1.3.3) - 2025-05-28

### Fixed

- Fix Python 3.9 syntax crash

## [v1.3.2](https://github.com/YonKuma/imajin.py/releases/tag/v1.3.2) - 2025-05-28

### Fixed

- Fix Python 3.9 syntax crash

## [v1.3.1](https://github.com/YonKuma/imajin.py/releases/tag/v1.3.1) - 2025-05-04

### Fixed

- Fix build system. No user facing changes

## [v1.3.0](https://github.com/YonKuma/imajin.py/releases/tag/v1.3.0) - 2025-05-04

### Added

- Add build system. No user facing changes

## [v1.2.0](https://github.com/YonKuma/imajin.py/releases/tag/v1.2.0) - 2025-05-04

### Added

- Add ability to install from `pip` using `pip install imajin`

## [v1.1.1](https://github.com/YonKuma/imajin.py/releases/tag/v1.1.1) - 2025-05-03

### Fixed

- Fix bug preventing volumes from outputting as they finish searching

## [v1.1.0](https://github.com/YonKuma/imajin.py/releases/tag/v1.1.0) - 2025-05-03

### Changed

- Improve chapter identification

### Added

- Parse Epub 3 nav files for chapter titles
- Output volumes as they finish searching

### Fix

- Fix bug that caused some image captions to be identified as chapter titles

## [v1.0.1](https://github.com/YonKuma/imajin.py/releases/tag/v1.0.1) - 2025-04-30

### Fix

- Fix bug that caused MeCab tokenization to fail for some MeCab configurations

## [v1.0.0](https://github.com/YonKuma/imajin.py/releases/tag/v1.0.0) - 2025-04-29

### Added

- Search Epub files for matching Japanese words
- Search Mokuro files for matching Japanese words
- Use MeCab to search using tokenization and deconjugation