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
