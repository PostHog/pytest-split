import enum
import hashlib
import heapq
from abc import ABC, abstractmethod
from bisect import bisect_right
from itertools import accumulate, pairwise
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest import nodes


class TestGroup(NamedTuple):
    selected: "list[nodes.Item]"
    deselected: "list[nodes.Item]"
    duration: float


class AlgorithmBase(ABC):
    """Abstract base class for the algorithm implementations."""

    @abstractmethod
    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        pass

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgorithmBase):
            return NotImplemented
        return self.__class__.__name__ == other.__class__.__name__


class LeastDurationAlgorithm(AlgorithmBase):
    """
    Split tests into groups by runtime.
    It walks the test items, starting with the test with largest duration.
    It assigns the test with the largest runtime to the group with the smallest duration sum.

    The algorithm sorts the items by their duration. Since the sorting algorithm is stable, ties will be broken by
    maintaining the original order of items. It is therefore important that the order of items be identical on all nodes
    that use this plugin. Due to issue #25 this might not always be the case.

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return:
        List of groups
    """

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)

        # add index of item in list
        items_with_durations_indexed = [
            (*tup, i) for i, tup in enumerate(items_with_durations)
        ]

        # Sort by name to ensure it's always the same order
        items_with_durations_indexed = sorted(
            items_with_durations_indexed, key=lambda tup: str(tup[0])
        )

        # sort in ascending order
        sorted_items_with_durations = sorted(
            items_with_durations_indexed, key=lambda tup: tup[1], reverse=True
        )

        selected: list[list[tuple[nodes.Item, int]]] = [[] for _ in range(splits)]
        deselected: list[list[nodes.Item]] = [[] for _ in range(splits)]
        duration: list[float] = [0 for _ in range(splits)]

        # create a heap of the form (summed_durations, group_index)
        heap: list[tuple[float, int]] = [(0, i) for i in range(splits)]
        heapq.heapify(heap)
        for item, item_duration, original_index in sorted_items_with_durations:
            # get group with smallest sum
            summed_durations, group_idx = heapq.heappop(heap)
            new_group_durations = summed_durations + item_duration

            # store assignment
            selected[group_idx].append((item, original_index))
            duration[group_idx] = new_group_durations
            for i in range(splits):
                if i != group_idx:
                    deselected[i].append(item)

            # store new duration - in case of ties it sorts by the group_idx
            heapq.heappush(heap, (new_group_durations, group_idx))

        groups = []
        for i in range(splits):
            # sort the items by their original index to maintain relative ordering
            # we don't care about the order of deselected items
            s = [
                item
                for item, original_index in sorted(selected[i], key=lambda tup: tup[1])
            ]
            group = TestGroup(
                selected=s, deselected=deselected[i], duration=duration[i]
            )
            groups.append(group)
        return groups


class DurationBasedChunksAlgorithm(AlgorithmBase):
    """
    Split tests into groups by runtime.
    Ensures tests are split into non-overlapping groups.
    The original list of test items is split into groups by finding boundary indices i_0, i_1, i_2
    and creating group_1 = items[0:i_0], group_2 = items[i_0, i_1], group_3 = items[i_1, i_2], ...

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return: List of TestGroup
    """

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)
        time_per_group = sum(map(itemgetter(1), items_with_durations)) / splits

        selected: list[list[nodes.Item]] = [[] for i in range(splits)]
        deselected: list[list[nodes.Item]] = [[] for i in range(splits)]
        duration: list[float] = [0 for i in range(splits)]

        group_idx = 0
        for item, item_duration in items_with_durations:
            if duration[group_idx] >= time_per_group:
                group_idx += 1

            selected[group_idx].append(item)
            for i in range(splits):
                if i != group_idx:
                    deselected[i].append(item)
            duration[group_idx] += item_duration

        return [
            TestGroup(
                selected=selected[i], deselected=deselected[i], duration=duration[i]
            )
            for i in range(splits)
        ]


class OptimalChunksAlgorithm(AlgorithmBase):
    """
    Split tests into contiguous groups with provably optimal balance.

    Like ``duration_based_chunks``, this keeps tests in their original collection
    order and only ever cuts the sequence into non-overlapping, contiguous slices
    (``group_1 = items[0:i_0]``, ``group_2 = items[i_0:i_1]``, ...). It therefore
    maintains *absolute order*: it never scatters tests across groups the way
    ``least_duration`` does. That makes it safe for suites with implicit
    inter-test ordering (a very common situation with Django's ``TestCase`` /
    ``TransactionTestCase``, where DB state and auto-increment IDs leak between
    neighbouring tests).

    The difference from ``duration_based_chunks`` is *where* the cuts go.
    ``duration_based_chunks`` uses a naive greedy rule (start a new group as soon
    as the current one reaches ``total / splits``), which front-loads early groups
    and can leave the slowest group much heavier than necessary. This algorithm
    instead computes the cut points that minimise the duration of the *slowest*
    group (the makespan) -- which is exactly what determines CI wall-clock time.

    This is the classic "linear partition" / "split array largest sum" /
    "painter's partition" problem. The optimum is found by binary-searching the
    smallest feasible makespan ``C`` (every group sum <= ``C``) and checking
    feasibility with a greedy left-to-right fill -- ``O(n log(sum))``. The result
    is reconstructed deterministically, so every CI node computes the same groups.

    :param splits: How many groups we're splitting in.
    :param items: Test items passed down by Pytest.
    :param durations: Our cached test runtimes. Assumes contains timings only of relevant tests
    :return: List of TestGroup
    """

    SCALE = 1_000_000  # seconds -> microseconds, so we can partition on exact ints

    def __call__(
        self, splits: int, items: "list[nodes.Item]", durations: "dict[str, float]"
    ) -> "list[TestGroup]":
        items_with_durations = _get_items_with_durations(items, durations)
        weights = [round(d * self.SCALE) for _, d in items_with_durations]
        boundaries = _optimal_boundaries(weights, splits)
        return _groups_from_boundaries(items_with_durations, boundaries)


def _greedy_boundaries(weights: "list[int]", capacity: int) -> "list[int]":
    """Left-to-right greedy fill at `capacity`; returns boundary indices [0, ..., n].

    Assumes `capacity >= max(weights)`, so every single item fits in a group.
    """
    boundaries = [0]
    current = 0
    for i, w in enumerate(weights):
        if current + w > capacity:
            boundaries.append(i)
            current = w
        else:
            current += w
    boundaries.append(len(weights))
    return boundaries


def _feasible_groups(weights: "list[int]", capacity: int) -> int:
    """Number of contiguous groups a greedy fill needs at `capacity`."""
    return len(_greedy_boundaries(weights, capacity)) - 1


def _best_split_point(weights: "list[int]", start: int, end: int) -> int:
    """Index in (start, end) that splits the slice into the most balanced halves."""
    total = sum(weights[start:end])
    left = 0
    best_cut = start + 1
    best_max = float("inf")
    for i in range(start, end - 1):
        left += weights[i]
        worst_half = max(left, total - left)
        if worst_half < best_max:
            best_max = worst_half
            best_cut = i + 1
    return best_cut


def _optimal_boundaries(weights: "list[int]", splits: int) -> "list[int]":
    """Boundary indices for `splits` contiguous groups minimising the slowest group.

    Returns ``splits + 1`` indices ``[0, b_1, ..., b_{splits-1}, n]``.
    """
    n = len(weights)
    if splits <= 1:
        return [0, n]
    if n <= splits:
        # One test per group as far as they go, then trailing empty groups.
        return list(range(n + 1)) + [n] * (splits - n)

    # Binary search the smallest makespan that fits in `splits` groups.
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if _feasible_groups(weights, mid) <= splits:
            hi = mid
        else:
            lo = mid + 1
    capacity = lo

    # Greedy fill at the optimal capacity may use fewer than `splits` groups.
    # Splitting any group into two contiguous halves keeps every group <= capacity,
    # so we can always reach exactly `splits` groups without hurting the makespan.
    boundaries = _greedy_boundaries(weights, capacity)
    while len(boundaries) - 1 < splits:
        splittable = [
            idx
            for idx in range(len(boundaries) - 1)
            if boundaries[idx + 1] - boundaries[idx] >= 2  # noqa: PLR2004
        ]
        if not splittable:
            # Every group is a single test already; pad with empty trailing groups.
            boundaries.append(n)
            continue
        # Split the heaviest group; smaller halves stay <= capacity, so the
        # makespan can't get worse.
        biggest = max(
            splittable,
            key=lambda idx: sum(weights[boundaries[idx] : boundaries[idx + 1]]),
        )
        start, end = boundaries[biggest], boundaries[biggest + 1]
        boundaries.insert(biggest + 1, _best_split_point(weights, start, end))
    return boundaries


def _groups_from_boundaries(
    items_with_durations: "list[tuple[nodes.Item, float]]", boundaries: "list[int]"
) -> "list[TestGroup]":
    """Build one TestGroup per ``[boundaries[i], boundaries[i + 1])`` slice."""
    items = [item for item, _ in items_with_durations]
    groups = []
    for start, end in pairwise(boundaries):
        groups.append(
            TestGroup(
                selected=items[start:end],
                deselected=items[:start] + items[end:],
                duration=sum(d for _, d in items_with_durations[start:end]),
            )
        )
    return groups


def _path_of(nodeid: str) -> str:
    """The test file portion of a node id (everything before the first '::')."""
    return nodeid.split("::", 1)[0]


def _dir_of(path: str) -> str:
    """The directory portion of a test file path ('' for a top-level file)."""
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _group_average_durations(
    durations: "dict[str, float]", key_of: "Callable[[str], str]"
) -> "dict[str, float]":
    """Average known duration per group, where a group is ``key_of(nodeid)``."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for nodeid, dur in durations.items():
        key = key_of(nodeid)
        totals[key] = totals.get(key, 0.0) + dur
        counts[key] = counts.get(key, 0) + 1
    return {key: total / counts[key] for key, total in totals.items()}


def _get_items_with_durations(
    items: "list[nodes.Item]", durations: "dict[str, float]"
) -> "list[tuple[nodes.Item, float]]":
    durations = _remove_irrelevant_durations(items, durations)
    global_average = _get_avg_duration_per_test(durations)
    file_average = _group_average_durations(durations, _path_of)
    dir_average = _group_average_durations(
        durations, lambda nodeid: _dir_of(_path_of(nodeid))
    )

    def estimate(nodeid: str) -> float:
        # A test with no stored timing is most like its file neighbours, then its
        # directory, then the suite as a whole. Falling back through those levels
        # beats the global average on skewed suites, where it badly under- or
        # over-budgets a freshly added test (the common case in a fast-moving repo).
        path = _path_of(nodeid)
        if path in file_average:
            return file_average[path]
        directory = _dir_of(path)
        if directory in dir_average:
            return dir_average[directory]
        return global_average

    return [
        (
            item,
            durations[item.nodeid]
            if item.nodeid in durations
            else estimate(item.nodeid),
        )
        for item in items
    ]


def _get_avg_duration_per_test(durations: "dict[str, float]") -> float:
    if durations:
        avg_duration_per_test = sum(durations.values()) / len(durations)
    else:
        # If there are no durations, give every test the same arbitrary value
        avg_duration_per_test = 1
    return avg_duration_per_test


def _remove_irrelevant_durations(
    items: "list[nodes.Item]", durations: "dict[str, float]"
) -> "dict[str, float]":
    # Filtering down durations to relevant ones ensures the avg isn't skewed by irrelevant data
    test_ids = [item.nodeid for item in items]
    durations = {name: durations[name] for name in test_ids if name in durations}
    return durations


class ItemPlan(NamedTuple):
    """Plan for file-level splitting with item-level balance and no artifact.

    The item-level optimal split is computed offline from the per-item durations.
    That gives two things a shard uses without ever collecting the whole tree:

    * ``file_shards`` -- the *footprint*: which shards a file's items land on.
      Almost every file -> one shard; only the few boundary files -> two (or
      more, for a file heavier than one shard). ``pytest_ignore_collect`` keeps a
      file only if this shard is in its footprint -- that's the collection saving.
    * ``thresholds2`` + ``file_offset`` -- the *budget*: cumulative-weight cut
      points (doubled, so the per-item midpoint stays integer). At collection time
      each shard spends its weight budget in the file's real collection order, so
      a shared (boundary) file is split runtime-contiguously -- order-safe,
      exactly like the item-level algorithm.

    Weights are integer microseconds, so every shard computes byte-identical cuts
    (no float drift across processes). Indices are 0-based (``--group`` minus one).
    """

    num_shards: int
    file_shards: "dict[str, list[int]]"  # file -> shards whose items touch it
    file_offset: "dict[str, int]"  # file -> cumulative weight of all earlier files
    file_avg: "dict[str, int]"  # file -> mean weight per item (for its untimed items)
    thresholds2: "list[int]"  # 2 * each internal cut weight (for midpoint tiling)


def item_plan(durations: "dict[str, float]", splits: int) -> "ItemPlan":
    """Compute the offline item-level split and derive footprint + weight budgets.

    ``durations`` should already be scoped to the files this run collects. See
    :class:`ItemPlan` for how the result is used.
    """
    scale = OptimalChunksAlgorithm.SCALE
    items = sorted(durations)
    # Floor at 1µs so a zero-duration test still occupies a distinct cumulative
    # position; otherwise a whole file of ~0s tests collapses onto one shard.
    weights = [max(1, round(durations[nodeid] * scale)) for nodeid in items]

    # Items are grouped by file (the nodeid sort is path-first), so a single pass
    # gives each file's cumulative-weight offset, total weight and mean item weight.
    file_offset: dict[str, int] = {}
    file_weight: dict[str, int] = {}
    file_count: dict[str, int] = {}
    pos = 0
    for nodeid, weight in zip(items, weights, strict=True):
        path = _path_of(nodeid)
        if path not in file_offset:
            file_offset[path] = pos
        file_weight[path] = file_weight.get(path, 0) + weight
        file_count[path] = file_count.get(path, 0) + 1
        pos += weight
    file_avg = {path: file_weight[path] // file_count[path] for path in file_weight}

    # Item-level optimal cut points as cumulative-weight thresholds, doubled so
    # the per-item midpoint (see assign_file_items) stays an exact integer.
    boundaries = _optimal_boundaries(weights, splits)
    prefix = list(accumulate(weights, initial=0))  # prefix[i] = sum(weights[:i])
    thresholds = [prefix[b] for b in boundaries[1:-1]]  # the N-1 internal cuts
    thresholds2 = [2 * threshold for threshold in thresholds]

    # Footprint: a file spans weight [offset, offset + weight); the shards that
    # touch it are those whose ranges overlap that span.
    file_shards: dict[str, list[int]] = {}
    for path, offset in file_offset.items():
        end = offset + file_weight[path]
        first = bisect_right(thresholds, offset)
        last = bisect_right(thresholds, max(offset, end - 1))
        file_shards[path] = list(range(first, last + 1))

    return ItemPlan(splits, file_shards, file_offset, file_avg, thresholds2)


def assign_file_items(
    plan: "ItemPlan", durations: "dict[str, float]", path: str, nodeids: "list[str]"
) -> "list[int]":
    """The cut rule: the shard each of ``path``'s items belongs to, in the order
    given (its real collection order).

    A file owned outright returns the same shard for every item; a boundary file
    is split where each item's cumulative-weight *midpoint* crosses a threshold --
    integer math, so every shard agrees on the boundary item and the file tiles
    exactly. Untimed items (new tests not yet in ``durations``) use the file's
    mean item weight; the result is clamped to the file's footprint so a late
    untimed item can never land on a shard that didn't collect the file.

    ``path`` must be a timed file (present in ``plan.file_offset``).
    """
    offset2 = 2 * plan.file_offset[path]
    avg = plan.file_avg.get(path, 0)
    footprint = plan.file_shards[path]
    lo, hi = footprint[0], footprint[-1]
    shards = []
    cumulative = 0
    for nodeid in nodeids:
        dur = durations.get(nodeid)
        # Same 1µs floor as item_plan, so the cut matches the offline plan.
        weight = (
            max(1, round(dur * OptimalChunksAlgorithm.SCALE))
            if dur is not None
            else avg
        )
        midpoint2 = offset2 + 2 * cumulative + weight
        cumulative += weight
        shard = bisect_right(plan.thresholds2, midpoint2)
        shards.append(min(max(shard, lo), hi))
    return shards


def stable_group(path: str, splits: int) -> int:
    """Deterministically map an (untimed) file path to a group, 0-based.

    Used for files with no stored duration -- new tests, which a fast-moving repo
    adds daily. A content hash, not the built-in ``hash()``: every CI shard runs
    in its own process with a randomised hash seed, so ``hash()`` would send the
    same new file to a different group on each shard and the file would be run
    on several shards or none. blake2b is identical across processes.
    """
    digest = hashlib.blake2b(path.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % splits


class Algorithms(enum.Enum):
    duration_based_chunks = DurationBasedChunksAlgorithm()
    least_duration = LeastDurationAlgorithm()
    optimal_chunks = OptimalChunksAlgorithm()

    @staticmethod
    def names() -> "list[str]":
        return [x.name for x in Algorithms]
