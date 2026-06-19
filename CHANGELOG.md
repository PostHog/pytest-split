# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- New `--split-granularity` option (`item` default, or `file`). `file` assigns
  whole test files to groups and skips other groups' files via
  `pytest_ignore_collect` — before they are imported — so a shard no longer
  collects (imports) the entire test tree just to run its slice. On large suites
  that whole-tree import is the dominant per-shard cost. File weights feed the
  same makespan partition, so balance is preserved; whole files are never split,
  so within-file ordering is kept. Untimed files are placed deterministically
  (each runs on exactly one shard). Requires pytest >= 7.
- New `optimal_chunks` splitting algorithm. Like `duration_based_chunks` it keeps
  tests in contiguous, in-order groups (so it never scatters tests the way
  `least_duration` does), but it computes the cut points that minimise the
  slowest group's duration instead of using a greedy rule. Useful for suites with
  implicit inter-test ordering (e.g. Django `TestCase`/`TransactionTestCase`)
  where `least_duration` would expose latent ordering dependencies.
- Estimate the duration of tests with no stored timing from their file's average,
  falling back to the directory average and then the global average (previously
  always the global average). This gives much closer estimates on suites with
  uneven per-file runtimes — most relevant for newly added tests, which would
  otherwise all be guessed at the suite-wide mean and skew a single shard.

### Fixed
- Fix malformed bullet points rendering in GitHub Pages documentation

## [0.11.0] - 2026-02-03
### Added
- Support for pytest 9.x
- Support for Python 3.14

### Changed
- Migrated `poetry.dev-dependencies` to `poetry.group.dev.dependencies`

### Removed
- Support for Python 3.8 and 3.9 (end-of-life)

## [0.10.0] - 2024-10-16
### Added
- Support for Python 3.13.

## [0.9.0] - 2024-06-19
### Changed
- Cruft update to get up to date with the parent cookiecutter template

### Removed
- Support for Python 3.7

## [0.8.2] - 2024-01-29
### Added
- Support for pytest 8.x
- Python 3.12 to CI test matrix

## [0.8.1] - 2023-04-12
### Changed
- Introduce Ruff
- Fixed usage of [deprecated pytest API](https://docs.pytest.org/en/latest/deprecations.html#configuring-hook-specs-impls-using-markers)

### Added
- Python 3.11 to CI test matrix

## [0.8.0] - 2022-04-22
### Fixed
- The `least_duration` algorithm should now split deterministically regardless of starting test order.
  This should fix the main problem when running with test-randomization packages such as `pytest-randomly` or `pytest-random-order`
  See #52

## [0.7.0] - 2022-03-13
### Added
- Support for pytest 7.x, see https://github.com/jerry-git/pytest-split/pull/47

## [0.6.0] - 2022-01-10
### Added
- PR template
- Test against 3.10
- Compatibility with IPython Notebooks

## [0.5.0] - 2021-11-09
### Added
- Wolt cookiecutter + cruft setup, see https://github.com/jerry-git/pytest-split/pull/33

## [0.4.0] - 2021-11-09
### Changed
- Durations file content in prettier format, see https://github.com/jerry-git/pytest-split/pull/31

[Unreleased]: https://github.com/jerry-git/pytest-split/compare/0.11.0...master
[0.11.0]: https://github.com/jerry-git/pytest-split/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/jerry-git/pytest-split/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/jerry-git/pytest-split/compare/0.8.2...0.9.0
[0.8.2]: https://github.com/jerry-git/pytest-split/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/jerry-git/pytest-split/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/jerry-git/pytest-split/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/jerry-git/pytest-split/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/jerry-git/pytest-split/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/jerry-git/pytest-split/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/jerry-git/pytest-split/tree/0.4.0

