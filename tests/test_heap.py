"""
Heap
----
"""

import itertools
import random
from unittest import TestCase

from src import heap


class TestHeap(TestCase):
    def setUp(self):
        self.x = random.randint(1, 10)
        self.n = random.randint(0, 1000)
        self.a = random.randint(-1000, 0)
        self.b = random.randint(0, 1000)

    def test_insert(self):
        """
        Test 'insert' operation.
        """
        for _ in range(self.x):
            h = create_random_array(n=self.n, a=self.a, b=self.b)
            heap.heapify(h)

            # Insert a random element into the heap.
            e = random.randint(-1000, 1000)
            heap.insert(h, e)

        assert is_heap(h)

    def test_delete_min(self):
        """
        Test 'delete-min' operation.
        """
        for _ in range(self.x):
            h = create_random_array(n=self.n, a=self.a, b=self.b)
            exp_min = min(h)
            heap.heapify(h)

            # Delete the smallest element from the heap.
            act_min = heap.delete_min(h)

            assert act_min == exp_min
            assert is_heap(h)

    def test_replace_min(self):
        """
        Test 'replace-min' operation.
        """
        for _ in range(self.x):
            h = create_random_array(n=self.n, a=self.a, b=self.b)
            exp_min = min(h)
            heap.heapify(h)

            # Replace the smallest element from the heap with new element e.
            e = random.randint(-1000, 1000)
            act_min = heap.replace_min(h, e)

            assert act_min == exp_min
            assert is_heap(h)

    def test_make_heap(self):
        """
        Test 'make-heap' operation.
        """
        for _ in range(self.x):
            h = create_random_array(n=self.n, a=self.a, b=self.b)
            heap.heapify(h)

            assert is_heap(h)

    def test_k_way_merge(self):
        """
        Test k-way merge algorithm.
        """
        for _ in range(self.x):
            k = random.randint(0, 1000)
            iterables = create_k_sorted_arrays(k=k, a=self.a, b=self.b)
            exp = sorted(itertools.chain(*iterables.copy()))

            assert list(heap.k_way_merge(*iterables)) == exp

    def test_kth_smallest(self):
        """
        Test kth_smallest solution.
        """
        for _ in range(self.x):
            # Pick a 'k' within the range of 'l' [1:self.n], inclusive. k
            # cannot be 0. k represents the kth element in a one-based array.
            # To get the kth element for a zero-based array, use k - 1.
            self.n = 1000
            k = random.randint(1, self.n)
            l = create_random_array(n=self.n, a=self.a, b=self.b)
            exp = sorted(l.copy())[k - 1]

            assert heap.kth_smallest(l=l, k=k) == exp


def create_random_array(n: int, a: int, b: int) -> list[int]:
    """
    Creates a random array of size `n` with integers between `a` and `b`, both
    inclusive.
    """
    return [random.randint(a, b) for _ in range(n)]


def create_sorted_array(n: int, a: int, b: int) -> list[int]:
    """
    Creates a sorted array of size `n` with integers between `a` and `b`, both
    inclusive.
    """
    return sorted([random.randint(a, b) for _ in range(n)])


def create_k_sorted_arrays(k: int, a: int, b: int) -> list[list[int]]:
    """
    Creates k sorted arrays of random size between `a` and `b` with integers
    between `a` and `b`, both inclusive.
    """
    n = random.randint(a, b)
    return [create_sorted_array(n=n, a=a, b=b) for _ in range(k)]


def is_heap(heap: list[int]) -> bool:
    """
    Returns true if the given heap is a valid min-heap.

    A heap is an array for which a[i] <= a[2*i+1] and a[i] <= a[2*i+2] for all
    i for zero-based arrays. a[0] is always the smallest element.
    """
    n = len(heap)
    for i in reversed(range(n // 2)):
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and heap[i] > heap[left]:
            return False
        if right < n and heap[i] > heap[right]:
            return False
    return True
