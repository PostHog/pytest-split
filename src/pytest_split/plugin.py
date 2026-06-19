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
    File-granularity splitting that keeps item-level balance, with no artifact.

    The item-level optimal split is computed offline from the per-item durations
    (see :func:`algorithms.item_plan`). From it each shard learns, before pytest
    imports anything, *which files its tests live in* -- so:

    * ``pytest_ignore_collect`` (pre-import) skips every file this shard has no
      tests in. Almost every file belongs to one shard, so a shard imports only
      ~1/N of the tree instead of all of it -- that's the collection saving, and
      the dominant per-shard cost on a large suite.
    * ``pytest_collection_modifyitems`` (post-import) splits the few *boundary*
      files shared by two shards: each shard spends a precomputed weight budget in
      the file's real collection order, so the cut is runtime-contiguous (order-
      safe, like the item-level algorithm) and the two shards tile the file
      exactly.

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
        self.plan = algorithms.item_plan(self._scoped_durations(config), self.splits)

    def _as_rel(self, target: str) -> str:
        """A pytest target (path arg or --ignore) as a rootdir-relative posix str."""
        target = target.split("::", 1)[0]  # drop any `::test_name` selector
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

        # The accept/reject decision is per *file*, so cache it and pay the
        # prefix checks + stat once per file rather than once per test nodeid.
        keep_file: dict[str, bool] = {}

        def in_scope(path: str) -> bool:
            return (
                (not targets or _under_any(path, targets))
                and not _under_any(path, ignores)
                and not any(fnmatch.fnmatch(path, glob) for glob in ignore_globs)
                and (self.rootpath / path).exists()  # drops stale keys
            )

        scoped: dict[str, float] = {}
        for nodeid, dur in self.cached_durations.items():
            path = algorithms._path_of(nodeid)  # noqa: SLF001  (same-package helper)
            keep = keep_file.get(path)
            if keep is None:
                keep = keep_file[path] = in_scope(path)
            if keep:
                scoped[nodeid] = dur
        return scoped

    def _is_test_file(self, name: str) -> bool:
        return any(
            fnmatch.fnmatch(name, pattern) for pattern in self.test_file_patterns
        )

    def _collects_file(self, rel_path: str) -> bool:
        """Whether this shard has any tests in ``rel_path`` (so must import it)."""
        shards = self.plan.file_shards.get(rel_path)
        if shards is None:
            # New file with no stored timing: place it on one shard deterministically.
            return algorithms.stable_group(rel_path, self.plan.num_shards) == (
                self.group_idx
            )
        return self.group_idx in shards

    def pytest_ignore_collect(self, collection_path: "Path") -> "bool | None":
        # The pre-import gate. Only decide for test files; returning None for
        # directories lets pytest recurse, and for non-test files (conftest.py,
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
        if self._collects_file(rel_path):
            return None  # our file: collect it (boundary files then split below)
        return True  # no tests of ours here: skip it before it is imported

    @hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: "Config", items: "list[nodes.Item]"
    ) -> None:
        # Keep this shard's items. A file we own outright keeps all its items; a
        # boundary file (shared with a neighbour) is split by the cut rule in
        # algorithms.assign_file_items -- each shard spends its weight budget in
        # the file's real collection order, so the cut is runtime-contiguous and
        # the two shards tile the file exactly.
        plan = self.plan
        kept: list[nodes.Item] = []
        deselected: list[nodes.Item] = []

        by_file: dict[str, list[nodes.Item]] = {}
        for item in items:
            by_file.setdefault(
                algorithms._path_of(item.nodeid),  # noqa: SLF001  (same-package helper)
                [],
            ).append(item)

        for path, file_items in by_file.items():
            if path not in plan.file_offset:
                # Untimed file: keep it only if we are its hash owner. ignore_collect
                # already gates this for walked files, but files named explicitly on
                # the command line bypass ignore_collect, so decide authoritatively
                # here to stay exactly-once.
                owner = algorithms.stable_group(path, plan.num_shards)
                (kept if owner == self.group_idx else deselected).extend(file_items)
                continue
            shards = algorithms.assign_file_items(
                plan, self.cached_durations, path, [item.nodeid for item in file_items]
            )
            for item, shard in zip(file_items, shards, strict=True):
                (kept if shard == self.group_idx else deselected).append(item)

        items[:] = kept
        config.hook.pytest_deselected(items=deselected)

    def pytest_report_collectionfinish(self) -> "list[str]":
        return [
            self.writer.markup(
                f"[pytest-split] Splitting by file (item-plan), group "
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
