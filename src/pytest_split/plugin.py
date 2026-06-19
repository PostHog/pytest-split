import fnmatch
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _pytest.config import create_terminal_writer, hookimpl
from _pytest.reports import TestReport

from pytest_split import algorithms
from pytest_split.ipynb_compatibility import ensure_ipynb_compatibility

if TYPE_CHECKING:
    from _pytest import nodes
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.main import ExitCode  # type: ignore[attr-defined]


# Ugly hack for freezegun compatibility: https://github.com/spulec/freezegun/issues/286
STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD = 60 * 10  # seconds

# `--split-granularity=file` relies on the `collection_path` ignore-collect hook,
# which pytest added in 7.0.
MIN_PYTEST_FOR_FILE_GRANULARITY = 7


def pytest_addoption(parser: "Parser") -> None:
    """
    Declare pytest-split's options.
    """
    group = parser.getgroup(
        "Split tests into groups which execution time is about the same. "
        "Run with --store-durations to store information about test execution times."
    )
    group.addoption(
        "--store-durations",
        dest="store_durations",
        action="store_true",
        help="Store durations into '--durations-path'.",
    )
    group.addoption(
        "--durations-path",
        dest="durations_path",
        help=(
            "Path to the file in which durations are (to be) stored, "
            "default is .test_durations in the current working directory"
        ),
        default=os.path.join(os.getcwd(), ".test_durations"),
    )
    group.addoption(
        "--splits",
        dest="splits",
        type=int,
        help="The number of groups to split the tests into",
    )
    group.addoption(
        "--group",
        dest="group",
        type=int,
        help="The group of tests that should be executed (first one is 1)",
    )
    group.addoption(
        "--splitting-algorithm",
        dest="splitting_algorithm",
        type=str,
        help=f"Algorithm used to split the tests. Choices: {algorithms.Algorithms.names()}",
        default="duration_based_chunks",
        choices=algorithms.Algorithms.names(),
    )
    group.addoption(
        "--split-granularity",
        dest="split_granularity",
        type=str,
        help=(
            "What to split on. 'item' (default) splits individual tests after the "
            "whole suite is collected. 'file' assigns whole test files to groups and "
            "skips other groups' files before they are imported -- this avoids "
            "collecting (importing) the entire tree on every shard, which is the "
            "dominant per-shard cost on a large suite."
        ),
        default="item",
        choices=("item", "file"),
    )
    group.addoption(
        "--clean-durations",
        dest="clean_durations",
        action="store_true",
        help=(
            "Removes the test duration info for tests which are not present "
            "while running the suite with '--store-durations'."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: "Config") -> "int | ExitCode | None":
    """
    Validate options.
    """
    group = config.getoption("group")
    splits = config.getoption("splits")

    if splits is None and group is None:
        return None

    if splits and group is None:
        raise pytest.UsageError("argument `--group` is required")

    if group and splits is None:
        raise pytest.UsageError("argument `--splits` is required")

    if splits < 1:
        raise pytest.UsageError("argument `--splits` must be >= 1")

    if group < 1 or group > splits:
        raise pytest.UsageError(f"argument `--group` must be >= 1 and <= {splits}")

    pytest_major = int(pytest.__version__.split(".", 1)[0])
    if (
        config.option.split_granularity == "file"
        and pytest_major < MIN_PYTEST_FOR_FILE_GRANULARITY
    ):
        # File granularity prunes via the `collection_path` ignore-collect hook,
        # added in pytest 7. On older pytest the hook would be silently dropped
        # (every shard would then run every file), so fail loudly instead.
        raise pytest.UsageError(
            "`--split-granularity=file` requires pytest>=7; "
            "use the default `--split-granularity=item` on older pytest"
        )

    return None


def pytest_configure(config: "Config") -> None:
    """
    Enable the plugins we need.
    """
    if config.option.splits and config.option.group:
        if config.option.split_granularity == "file":
            config.pluginmanager.register(
                PytestSplitFilePlugin(config), "pytestsplitfileplugin"
            )
        else:
            config.pluginmanager.register(
                PytestSplitPlugin(config), "pytestsplitplugin"
            )

    if config.option.store_durations:
        config.pluginmanager.register(
            PytestSplitCachePlugin(config), "pytestsplitcacheplugin"
        )


class Base:
    def __init__(self, config: "Config") -> None:
        """
        Load durations and set up a terminal writer.

        This logic is shared for both the split- and cache plugin.
        """
        self.config = config
        self.writer = create_terminal_writer(self.config)

        try:
            with open(config.option.durations_path) as f:
                self.cached_durations = json.loads(f.read())
        except FileNotFoundError:
            self.cached_durations = {}

        # This code provides backwards compatibility after we switched
        # from saving durations in a list-of-lists to a dict format
        # Remove this when bumping to v1
        if isinstance(self.cached_durations, list):
            self.cached_durations = dict(self.cached_durations)


class PytestSplitPlugin(Base):
    def __init__(self, config: "Config"):
        super().__init__(config)

        if not self.cached_durations:
            message = self.writer.markup(
                "\n[pytest-split] No test durations found. Pytest-split will "
                "split tests evenly when no durations are found. "
                "\n[pytest-split] You can expect better results in consequent runs, "
                "when test timings have been documented.\n"
            )
            self.writer.line(message)

    @hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: "Config", items: "list[nodes.Item]"
    ) -> None:
        """
        Collect and select the tests we want to run, and deselect the rest.
        """
        splits: int = config.option.splits
        group_idx: int = config.option.group

        algo = algorithms.Algorithms[config.option.splitting_algorithm].value
        groups = algo(splits, items, self.cached_durations)
        group = groups[group_idx - 1]

        ensure_ipynb_compatibility(group, items)

        items[:] = group.selected
        config.hook.pytest_deselected(items=group.deselected)

        self.writer.line(
            self.writer.markup(
                f"\n\n[pytest-split] Splitting tests with algorithm: {config.option.splitting_algorithm}"
            )
        )
        self.writer.line(
            self.writer.markup(
                f"[pytest-split] Running group {group_idx}/{splits} (estimated duration: {group.duration:.2f}s)\n"
            )
        )


def _under_any(path: str, prefixes: "list[str]") -> bool:
    """True if ``path`` is one of, or under, any of ``prefixes`` (a rootdir
    prefix of ``""``/``"."`` matches everything)."""
    for prefix in prefixes:
        if prefix in ("", "."):
            return True
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


class PytestSplitFilePlugin(Base):
    """
    File-granularity splitting, in two stages.

    Stage 1 -- ``pytest_ignore_collect`` (pre-import): whole files are grouped into
    buckets, and a shard skips every bucket's files but its own. Pruning here,
    before pytest imports a file, is what avoids collecting the whole tree on every
    shard -- the dominant per-shard cost on a large suite.

    Stage 2 -- ``pytest_collection_modifyitems`` (post-import): within its bucket, a
    shard runs the normal item-level algorithm over the bucket's collected items
    and keeps only its slice. This recovers fine balance and splits any single file
    too heavy for one shard, while staying contiguous -- so it's as order-safe as
    the item-level algorithm it reuses.

    The durations universe is scoped to what this run actually collects (its target
    paths, minus ``--ignore``, minus files no longer on disk), so a multi-segment CI
    can't pollute one segment's partition with another segment's files or with stale
    keys for deleted tests.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__(config)
        self.splits: int = config.option.splits
        self.group_idx: int = config.option.group - 1  # 0-based
        # nodeids are relative to rootpath. resolve() so the symlinked-tmp case
        # (macOS /tmp -> /private/tmp) doesn't break the relative_to below.
        self.rootpath = config.rootpath.resolve()
        self.test_file_patterns: list[str] = config.getini("python_files")
        self.plan = algorithms.bucket_plan(self._scoped_durations(config), self.splits)
        self.my_bucket = self.plan.shard_bucket[self.group_idx]
        self.my_within = self.plan.shard_within[self.group_idx]
        self.my_bucket_width = self.plan.bucket_widths[self.my_bucket]

    def _as_rel(self, target: str) -> str:
        """A pytest target (path arg or --ignore) as a rootdir-relative posix str."""
        path = Path(target)
        if not path.is_absolute():
            path = self.rootpath / path
        try:
            return path.resolve().relative_to(self.rootpath).as_posix()
        except ValueError:
            return Path(target).as_posix().rstrip("/")

    def _scoped_durations(self, config: "Config") -> "dict[str, float]":
        """Durations restricted to files this run collects: under a target path,
        not ``--ignore``-d, and still present on disk (drops stale keys)."""
        targets = [self._as_rel(a) for a in (config.args or [])]
        ignores = [self._as_rel(i) for i in (config.getoption("ignore") or [])]
        ignore_globs = config.getoption("ignore_glob") or []
        exists: dict[str, bool] = {}
        scoped: dict[str, float] = {}
        for nodeid, dur in self.cached_durations.items():
            path = algorithms._path_of(nodeid)  # noqa: SLF001  (same-package helper)
            if targets and not _under_any(path, targets):
                continue
            if _under_any(path, ignores):
                continue
            if any(fnmatch.fnmatch(path, glob) for glob in ignore_globs):
                continue
            present = exists.get(path)
            if present is None:
                present = (self.rootpath / path).exists()
                exists[path] = present
            if not present:
                continue
            scoped[nodeid] = dur
        return scoped

    def _is_test_file(self, name: str) -> bool:
        return any(
            fnmatch.fnmatch(name, pattern) for pattern in self.test_file_patterns
        )

    def _bucket_of(self, rel_path: str) -> int:
        bucket = self.plan.file_bucket.get(rel_path)
        if bucket is None:
            # New file with no stored timing: place it in a bucket deterministically.
            bucket = algorithms.stable_group(rel_path, self.plan.num_buckets)
        return bucket

    def pytest_ignore_collect(self, collection_path: "Path") -> "bool | None":
        # Stage 1, the pre-import gate. Only decide for test files; returning None
        # for directories lets pytest recurse, and for non-test files (conftest.py,
        # helpers) leaves them untouched.
        if not self._is_test_file(collection_path.name):
            return None
        try:
            rel_path = collection_path.relative_to(self.rootpath).as_posix()
        except ValueError:  # pragma: no cover
            # Fast path failed: rootpath/collection_path can disagree on symlinks.
            # Resolve the file (slow path only) and retry before giving up.
            try:
                rel_path = (
                    collection_path.resolve().relative_to(self.rootpath).as_posix()
                )
            except ValueError:
                return None  # genuinely outside rootdir -- don't second-guess pytest
        if self._bucket_of(rel_path) != self.my_bucket:
            return True  # another bucket's file: skip it before it is imported
        return None  # our bucket: collect, then stage 2 splits it among our shards

    @hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: "Config", items: "list[nodes.Item]"
    ) -> None:
        # Stage 2: split this bucket's collected items across the shards assigned to
        # the bucket, with the configured item-level algorithm, and keep our slice.
        # A bucket that owns a single shard needs no further split.
        if self.my_bucket_width <= 1:
            return
        algo = algorithms.Algorithms[config.option.splitting_algorithm].value
        group = algo(self.my_bucket_width, items, self.cached_durations)[self.my_within]
        ensure_ipynb_compatibility(group, items)
        items[:] = group.selected
        config.hook.pytest_deselected(items=group.deselected)

    def pytest_report_collectionfinish(self) -> "list[str]":
        return [
            self.writer.markup(
                f"[pytest-split] Splitting by file: bucket "
                f"{self.my_bucket + 1}/{self.plan.num_buckets}, group "
                f"{self.group_idx + 1}/{self.splits}"
            )
        ]


class PytestSplitCachePlugin(Base):
    """
    The cache plugin writes durations to our durations file.
    """

    def pytest_sessionfinish(self) -> None:
        """
        Method is called by Pytest after the test-suite has run.
        https://github.com/pytest-dev/pytest/blob/main/src/_pytest/main.py#L308
        """
        terminal_reporter = self.config.pluginmanager.get_plugin("terminalreporter")
        test_durations: dict[str, float] = {}

        for test_reports in terminal_reporter.stats.values():  # type: ignore[union-attr]
            for test_report in test_reports:
                if isinstance(test_report, TestReport):
                    # These ifs be removed after this is solved: # https://github.com/spulec/freezegun/issues/286
                    if test_report.duration < 0:
                        continue  # pragma: no cover
                    if (
                        test_report.when in ("teardown", "setup")
                        and test_report.duration
                        > STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD
                    ):
                        # Ignore not legit teardown durations
                        continue  # pragma: no cover

                    # Add test durations to map
                    if test_report.nodeid not in test_durations:
                        test_durations[test_report.nodeid] = 0
                    test_durations[test_report.nodeid] += test_report.duration

        if self.config.option.clean_durations:
            self.cached_durations = dict(test_durations)
        else:
            for k, v in test_durations.items():
                self.cached_durations[k] = v

        with open(self.config.option.durations_path, "w") as f:
            json.dump(self.cached_durations, f, sort_keys=True, indent=4)

        message = self.writer.markup(
            f"\n\n[pytest-split] Stored test durations in {self.config.option.durations_path}"
        )
        self.writer.line(message)
