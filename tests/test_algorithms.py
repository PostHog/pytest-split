import hashlib
import itertools
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.nodes import Item

from pytest_split.algorithms import (
    AlgorithmBase,
    Algorithms,
    _get_items_with_durations,
    _path_of,
    assign_file_items,
    item_plan,
    stable_group,
)

item = namedtuple("item", "nodeid")  # noqa: PYI024


class TestAlgorithms:
    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_test(self, algo_name):
        durations = {"a": 1, "b": 1, "c": 1}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value
        first, second, third = algo(splits=3, items=items, durations=durations)

        # each split should have one test
        assert first.selected == [item("a")]
        assert first.deselected == [item("b"), item("c")]
        assert first.duration == 1

        assert second.selected == [item("b")]
        assert second.deselected == [item("a"), item("c")]
        assert second.duration == 1

        assert third.selected == [item("c")]
        assert third.deselected == [item("a"), item("b")]
        assert third.duration == 1

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_in_durations_but_missing_from_items(
        self, algo_name
    ):
        durations = {"a": 1, "b": 1}
        items = [item(x) for x in ["a"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == []

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_with_missing_durations(self, algo_name):
        durations = {"a": 1}
        items = [item(x) for x in ["a", "b"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == [item("b")]

    def test__split_test_handles_large_duration_at_end(self):
        """NOTE: only least_duration does this correctly"""
        durations = {"a": 1, "b": 1, "c": 1, "d": 3}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms["least_duration"].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("d")]
        assert second.selected == [item(x) for x in ["a", "b", "c"]]

    def test__optimal_chunks_minimises_makespan_where_greedy_does_not(self):
        # duration_based_chunks front-loads: it fills group 1 to a,b (=9) and
        # leaves c alone (=4), so the slowest group is 9. The optimal contiguous
        # split is a | b,c (slowest group = 8). optimal_chunks finds it, while
        # still keeping tests in their original, contiguous order.
        durations = {"a": 5, "b": 4, "c": 4}
        items = [item(x) for x in ["a", "b", "c"]]

        greedy = Algorithms["duration_based_chunks"].value(2, items, durations)
        optimal = Algorithms["optimal_chunks"].value(2, items, durations)

        # greedy front-loads -> [a, b] | [c]; optimal balances -> [a] | [b, c]
        assert [g.selected for g in greedy] == [
            [item("a"), item("b")],
            [item("c")],
        ]
        assert [g.selected for g in optimal] == [
            [item("a")],
            [item("b"), item("c")],
        ]
        # the whole point: optimal's slowest group finishes sooner than greedy's
        assert max(g.duration for g in optimal) < max(g.duration for g in greedy)

    def test__optimal_chunks_keeps_groups_contiguous(self):
        # The Django-safety guarantee: every group is a contiguous slice of the
        # original collection order, so no group ever interleaves tests that were
        # not neighbours. A heavy test in the middle must not get scattered.
        durations = {"a": 1, "b": 1, "c": 10, "d": 1, "e": 1, "f": 1}
        items = [item(x) for x in ["a", "b", "c", "d", "e", "f"]]

        groups = Algorithms["optimal_chunks"].value(3, items, durations)

        flattened = [it for g in groups for it in g.selected]
        assert flattened == items  # same order, nothing reshuffled or dropped
        # and each group is a run of consecutive original items
        for g in groups:
            indices = [items.index(it) for it in g.selected]
            assert indices == list(range(indices[0], indices[0] + len(indices)))

    def test__missing_duration_uses_file_average_not_global(self):
        # global average here is 5.5s; a new test should inherit its own file's
        # average instead, so a new test in a slow file isn't under-budgeted.
        fast_avg, slow_avg = 1.0, 10.0
        durations = {
            "fast/test_a.py::test_1": fast_avg,
            "fast/test_a.py::test_2": fast_avg,
            "slow/test_b.py::test_1": slow_avg,
            "slow/test_b.py::test_2": slow_avg,
        }
        items = [item(nodeid) for nodeid in durations] + [
            item("fast/test_a.py::test_new"),
            item("slow/test_b.py::test_new"),
        ]
        estimated = {
            it.nodeid: dur for it, dur in _get_items_with_durations(items, durations)
        }
        assert estimated["fast/test_a.py::test_new"] == fast_avg
        assert estimated["slow/test_b.py::test_new"] == slow_avg

    def test__missing_duration_falls_back_to_directory_average(self):
        # a brand-new file in a known directory inherits the directory average.
        durations = {
            "dir/test_a.py::test_1": 2.0,
            "dir/test_b.py::test_1": 4.0,
        }
        items = [item(nodeid) for nodeid in durations] + [
            item("dir/test_new_file.py::test_1")
        ]
        estimated = {
            it.nodeid: dur for it, dur in _get_items_with_durations(items, durations)
        }
        directory_average = (2.0 + 4.0) / 2
        assert estimated["dir/test_new_file.py::test_1"] == directory_average

    def test__missing_duration_falls_back_to_global_average(self):
        # a new test in an entirely new directory falls back to the global average.
        durations = {
            "a/test_x.py::test_1": 2.0,
            "b/test_y.py::test_1": 4.0,
        }
        items = [item(nodeid) for nodeid in durations] + [
            item("brand/new/test_z.py::test_1")
        ]
        estimated = {
            it.nodeid: dur for it, dur in _get_items_with_durations(items, durations)
        }
        global_average = (2.0 + 4.0) / 2
        assert estimated["brand/new/test_z.py::test_1"] == global_average

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            ("duration_based_chunks", [[item("a"), item("b")], [item("c"), item("d")]]),
            ("least_duration", [[item("a"), item("c")], [item("b"), item("d")]]),
            ("optimal_chunks", [[item("a"), item("b")], [item("c"), item("d")]]),
        ],
    )
    def test__split_tests_calculates_avg_test_duration_only_on_present_tests(
        self, algo_name, expected
    ):
        # If the algo includes test e's duration to calculate the averge then
        # a will be expected to take a long time, and so 'a' will become its
        # own group. Intended behaviour is that a gets estimated duration 1 and
        # this will create more balanced groups.
        durations = {"b": 1, "c": 1, "d": 1, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            (
                "duration_based_chunks",
                [[item("a"), item("b"), item("c"), item("d"), item("e")], []],
            ),
            (
                "least_duration",
                [[item("e")], [item("a"), item("b"), item("c"), item("d")]],
            ),
            (
                "optimal_chunks",
                [[item("a"), item("b"), item("c"), item("d")], [item("e")]],
            ),
        ],
    )
    def test__split_tests_maintains_relative_order_of_tests(self, algo_name, expected):
        durations = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d", "e"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    def test__split_tests_same_set_regardless_of_order(self):
        """NOTE: only least_duration does this correctly"""
        tests = ["a", "b", "c", "d", "e", "f", "g"]
        durations = {t: 1 for t in tests}
        items = [item(t) for t in tests]
        algo = Algorithms["least_duration"].value
        for n in (2, 3, 4):
            selected_each: list[set[Item]] = [set() for _ in range(n)]
            for order in itertools.permutations(items):
                splits = algo(splits=n, items=order, durations=durations)
                for i, group in enumerate(splits):
                    if not selected_each[i]:
                        selected_each[i] = set(group.selected)
                    assert selected_each[i] == set(group.selected)

    def test__algorithms_members_derived_correctly(self):
        for a in Algorithms.names():
            assert issubclass(Algorithms[a].value.__class__, AlgorithmBase)


class MyAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class MyOtherAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class TestAbstractAlgorithm:
    def test__hash__returns_correct_result(self):
        algo = MyAlgorithm()
        assert algo.__hash__() == hash(algo.__class__.__name__)

    def test__hash__returns_same_hash_for_same_class_instances(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__hash__() == algo2.__hash__()

    def test__hash__returns_different_hash_for_different_classes(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__hash__() != algo2.__hash__()

    def test__eq__returns_true_for_same_instance(self):
        algo = MyAlgorithm()
        assert algo.__eq__(algo) is True

    def test__eq__returns_false_for_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__eq__(algo2) is False

    def test__eq__returns_true_for_same_algorithm_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__eq__(algo2) is True

    def test__eq__returns_false_for_non_algorithm_object(self):
        algo = MyAlgorithm()
        other = "not an algorithm"
        assert algo.__eq__(other) is NotImplemented


class TestStableGroup:
    def test_is_deterministic(self):
        assert stable_group("posthog/test_foo.py", 7) == stable_group(
            "posthog/test_foo.py", 7
        )

    def test_stays_within_range(self):
        splits = 8
        for i in range(50):
            group = stable_group(f"pkg/test_{i}.py", splits)
            assert 0 <= group < splits

    def test_does_not_depend_on_pythonhashseed(self):
        # blake2b, not the salted built-in hash() -- so the value is fixed and
        # every CI shard process maps a new file to the same group.
        splits = 16
        path = "products/new/test_thing.py"
        expected = (
            int.from_bytes(
                hashlib.blake2b(path.encode(), digest_size=8).digest(), "big"
            )
            % splits
        )
        assert stable_group(path, splits) == expected


def _tile(plan, durations, items_in_order):
    """Replay the plugin's modifyitems tiling (via the shared cut rule) and return
    {nodeid: shard}, grouping by file in the given collection order."""
    by_file: dict[str, list[str]] = {}
    for nodeid in items_in_order:
        by_file.setdefault(_path_of(nodeid), []).append(nodeid)
    assignment: dict[str, int] = {}
    for path, nodeids in by_file.items():
        if path not in plan.file_offset:  # untimed file -> its hash owner
            for nodeid in nodeids:
                assignment[nodeid] = stable_group(path, plan.num_shards)
            continue
        shards = assign_file_items(plan, durations, path, nodeids)
        assignment.update(zip(nodeids, shards, strict=True))
    return assignment


class TestItemPlan:
    def test_most_files_go_to_a_single_shard(self):
        # Even-weight files spread across shards: each file's items land on one
        # shard, so its footprint has length 1 (only boundary files have 2).
        durations = {f"f{i:02d}/test.py::t": 1.0 for i in range(40)}
        plan = item_plan(durations, splits=4)
        singletons = sum(1 for s in plan.file_shards.values() if len(s) == 1)
        assert singletons >= len(plan.file_shards) - 4  # at most ~splits boundary files

    def test_oversized_file_spans_multiple_shards(self):
        # A file heavier than one shard's budget must be touched by several shards
        # (its items get split across them).
        durations = {f"big/test.py::t{i}": 1.0 for i in range(100)}
        durations.update({f"f{i}/test.py::t": 1.0 for i in range(10)})
        plan = item_plan(durations, splits=8)
        assert len(plan.file_shards["big/test.py"]) > 1

    def test_tiling_covers_every_item_within_footprint(self):
        # The core invariant: every item is assigned to exactly one shard, and to
        # a shard that collects its file -- regardless of collection order.
        durations = {
            f"pkg{i % 7}/test_{i % 3}.py::test_{i}": float(i % 5 + 1)
            for i in range(120)
        }
        splits = 5
        plan = item_plan(durations, splits)
        for reverse in (False, True):  # order-independence of the invariants
            order = sorted(durations, reverse=reverse)
            assignment = _tile(plan, durations, order)
            assert set(assignment) == set(durations)  # coverage, no dup
            for nodeid, shard in assignment.items():
                path = nodeid.split("::", 1)[0]
                assert shard in plan.file_shards[path]  # footprint consistency
            assert {assignment[n] for n in durations} == set(range(splits))

    def test_assign_clamps_untimed_items_into_the_footprint(self):
        # A timed file's footprint is computed from its stored items only. New
        # untimed items added later must clamp back into that footprint, never
        # spilling onto a shard that didn't collect the file (which would silently
        # drop them).
        durations = {f"{name}.py::t": 1.0 for name in "abcd"}
        plan = item_plan(durations, splits=2)
        footprint = plan.file_shards["b.py"]
        # one stored item plus three new untimed ones, heavy enough that their
        # tiled midpoints would cross the cut without the clamp
        nodeids = ["b.py::t", "b.py::new1", "b.py::new2", "b.py::new3"]
        shards = assign_file_items(plan, durations, "b.py", nodeids)
        assert all(shard in footprint for shard in shards)

    def test_zero_duration_files_still_spread_across_shards(self):
        # All-zero stored durations must not collapse every file onto one shard
        # (the 1µs weight floor keeps them at distinct cumulative positions).
        splits = 4
        durations = {f"f{i:02d}/test.py::t": 0.0 for i in range(40)}
        plan = item_plan(durations, splits)
        used = {shard for shards in plan.file_shards.values() for shard in shards}
        assert used == set(range(splits))

    def test_no_durations_leaves_everything_to_the_hash(self):
        splits = 3
        plan = item_plan({}, splits)
        assert plan.num_shards == splits
        assert plan.file_shards == {}  # nothing timed; placed by hash at collect time
