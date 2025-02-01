"""
Heap
----

A heap is a binary tree in which each element has a key (or sometimes priority)
that is less than (or greater than) the keys of its children. This property is
called the heap property or heap invariant. Heaps can be used to implement
priority queues.

Though it is possible to build a heap using structs and pointers, in practice,
heaps are stored in arrays, with an implicit pointer structure determined by
array indices.

Formally, a heap is an array for which a[i] <= a[2*i+1] and a[i] <= a[2*i+2]
for all i for zero-based arrays. a[0] is always the smallest element (or
largest for max-heaps).

Sift down vs. sift up
---------------------
An implementation of a heap requires two critical operations: sift down and
sift up. With sift down, a node is moved to a position in heap where it
satisfies the heap property by moving it "down" the heap. This process involves
comparing the node to its children and performing a swap if it is larger (or
smaller for max-heaps). With sift up, a node is moved to a position in heap
where it satisfies the heap property by moving it "up" the heap. This process
involves comparing the node to its parent and performing a swap if it is
smaller (or larger for max-heaps). Both the sift down and sift up operations
are used in higher-level heap operations to reposition nodes in the heap. Both
operations have a worst-case time complexity of O(log n), since it requires
log(n) operations to sift a node down or up a heap of height log(n).

Heapify (heapification)
-----------------------
Building a heap from an array is the first step in the heapsort algorithm and
involves reorganizing the array into a heap. Starting with the first non-leaf
node (i.e., the parent of the last element in the array), nodes are sifted down
the heap until they satisfy the heap invariant. An important characteristic of
bottom-up heapification is that it can be done using O(n) operations, that is,
in linear time. This is due to the fact that the number of operations forms a
geometric series that sums to 1:

  n/4 * 1 + n/8 * 2 + n/16 * 3 + ... + 1 * log n

Building a heap from an empty array, that is, inserting each new element into
the heap has O(nlog n) performance (log(n) comparisons/swaps for n elements).
For this reason, heapification is always done from bottom-up.

NOTE: You also cannot simply float down all nodes starting from the root, as
this does not guarantee the heap property will be maintained. Instead, the root
node must be removed and floated up to its correct position in the heap from
the bottom, which is always an O(log n) operation.

k-way Merge
-----------
k-way merge is a merge algorithm that, as the name suggests, merges k sorted
lists into a single sorted list. It can be efficiently solved in O(nlog k) time
using a heap, where n is the total number of elements for all k lists.

kth smallest (largest)
----------------------
The kth smallest (or largest) problem can be efficiently solved using a
max-heap (or min-heap). Maintaining a heap of size k, the kth smallest value is
always the root of the max-heap. To retrieve all k smallest elements, simply
return the sorted heap.
"""

from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar


class Comparable(Protocol):
    """
    Defines a protocol for comparable types.
    """

    def __lt__(self, other) -> bool: ...


# Create a TypeVar bound to Comparable
T = TypeVar("T", bound=Comparable)


def insert(heap: list[T], e: T) -> None:
    """
    Insert an element into the heap and reestablish the heap invariant.

    The insert operation appends an element to the end of the heap, then sifts
    it up to its correct position. It has a worst-case time complexity of
    O(log n).
    """
    # Add the element to the end of the heap. This may violate the heap
    # invariant.
    heap.append(e)
    # Reestablish the heap invariant. `heap` is a heap for all indices < i,
    # where i is the newly added element (n - 1), which may have violated the
    # heap invariant. The heap invariant is restored by calling the sift up
    # operation, which moves the element at index i to its correct position in
    # the heap.
    _sift_up(heap=heap, i=len(heap) - 1)


def insert_max(heap: list[T], e: T) -> None:
    """
    Same as `insert()`, but for a max-heap.
    """
    heap.append(e)
    _sift_up_max(heap=heap, i=len(heap) - 1)


def delete_min(heap: list[T]) -> T:
    """
    Remove the smallest element from the heap and reestablish the heap
    invariant.

    The delete-min operation removes the smallest element (heap[0] or the root)
    from the heap. Since removing an element from an array requires shifting
    all subsequent elements to the left by one (an O(n) operation), we can
    leverage a strategy used in heapsort, which swaps the root with the last
    element in the heap, removes the last element from the array (O(1)), then
    floats down the new root to its proper position in the heap (O(log n). The
    resulting operation has a time complexity of O(log n).

    NOTE: If we were to repeat the delete-min operation for all elements in the
    heap, the sequence of removed elements would yield a sorted array. This is
    the basis for the heapsort algorithm.
    """
    i = len(heap) - 1
    heap[0], heap[i] = heap[i], heap[0]  # Raises IndexError if heap is empty.
    _min = heap.pop()
    _sift_down(heap, 0)
    return _min


def delete_max(heap: list[T]) -> T:
    """
    Same as `delete_min()`, but for a max-heap.
    """
    i = len(heap) - 1
    heap[0], heap[i] = heap[i], heap[0]  # Raises IndexError if heap is empty.
    _min = heap.pop()
    _sift_down_max(heap, 0)
    return _min


def replace_min(heap: list[T], e: T) -> T:
    """
    Remove the smallest element from the heap and reestablish the heap
    invariant using the new element e.

    The replace-min operation removes the smallest element (heap[0] or the
    root) from the heap and reestablishes the heap invariant using the new
    element. Technically, this is more efficient than a delete-min + insert
    operation, which is O(log n) + O(log n).
    """
    _min, heap[0] = heap[0], e  # Raises IndexError if heap is empty.
    _sift_down(heap, 0)
    return _min


def replace_max(heap: list[T], e: T) -> T:
    """
    Same as `replace_min()`, but for a max-heap.
    """
    _min, heap[0] = heap[0], e  # Raises IndexError if heap is empty.
    _sift_down_max(heap, 0)
    return _min


def make_heap(heap: list[T]) -> None:
    """
    Build a binary min-heap from an array.

    Heapify (or bottom-up heapification) is the process of converting an array
    into a heap. Starting with the first non-leaf node (i.e., the parent of the
    last element in the array), nodes are sifted down the heap until they
    satisfy the heap invariant.
    """
    # For zero-based arrays, the first non-leaf node is given by
    # floor((n-2) / 2). Since the `range` function is exclusive, we simply use
    # n//2, since range(n//2) yields 0..(n//2 - 1).
    for i in reversed(range(len(heap) // 2)):
        _sift_down(heap, i)


def make_heap_max(heap: list[T]) -> None:
    """
    Same as `make_heap()`, but for a max-heap.
    """
    for i in reversed(range(len(heap) // 2)):
        _sift_down_max(heap, i)


def _sift_down(heap: list[T], i: int) -> None:
    """
    Sift down the element at index i to its correct position in the heap.

    The sift down function moves node i to a position that satisfies the heap
    property by satisfying one of three cases:

      1. The node has no children (i.e., it is a leaf node).
      2. The node is less than either its children.
      3. The node is greater than the smallest child and the nodes are
         exchanged.

    The third case may violate the heap property for subtree for which the node
    is now the root. Therefore, the sift down operation is repeated.
    """
    n = len(heap)
    # Keep swapping the parent node with its smallest child until it is the
    # smallest of its children or is a leaf node.
    while True:
        # If the node has no children (i.e., it is a leaf node), it cannot be
        # sifted down any further.
        if _is_leaf(i, n):
            break
        left, right = _left_child(i), _right_child(i)
        # To establish the min-heap property at i, up to three nodes must be
        # compared (the root and one or both of its children). The smallest is
        # swapped with i. If i is the smallest node, the heap invariant is
        # satisfied and no swap occurs.
        smallest = i
        if left < n and heap[left] < heap[smallest]:
            smallest = left
        if right < n and heap[right] < heap[smallest]:
            smallest = right
        if smallest != i:
            heap[i], heap[smallest] = heap[smallest], heap[i]
            i = smallest
        else:
            break


def _sift_down_max(heap: list[T], i: int) -> None:
    """
    Same as `_sift_down()`, but for a max-heap.
    """
    n = len(heap)
    while True:
        if _is_leaf(i, n):
            break
        left, right = _left_child(i), _right_child(i)
        largest = i
        if left < n and heap[left] > heap[largest]:
            largest = left
        if right < n and heap[right] > heap[largest]:
            largest = right
        if largest != i:
            heap[i], heap[largest] = heap[largest], heap[i]
            i = largest
        else:
            break


def _sift_up(heap: list[T], i: int) -> None:
    """
    Sift up the element at index i to its correct position in the heap.

    The sift up function moves node i to a position that satisfies the heap
    property by satisfying one of three cases:

      1. The node has no parent (i.e., it is the root node).
      2. The node is greater than or equal to its parent.
      3. The node is less than its parent and the nodes are exchanged.

    The third case may violate the heap property for subtree for which the node
    is now a child. Therefore the sift up operation is repeated.
    """
    # Keep swapping the child node with its parent until it is greater than its
    # parent or is the root node.
    while True:
        # If the node has no parent (i.e., it is the root node), it cannot be
        # sifted up any further.
        if i == 0:
            break
        parent = _parent(i)
        # To establish the min-heap property at i, the child node is compared
        # to its parent. If the child is less than its parent, the two nodes
        # are swapped. If i is greater than its parent, no swap occurs.
        if heap[i] < heap[parent]:
            heap[i], heap[parent] = heap[parent], heap[i]
            i = parent
        else:
            break


def _sift_up_max(heap: list[T], i: int) -> None:
    """
    Same as `_sift_up()`, but for a max-heap.
    """
    while True:
        if i == 0:
            break
        parent = _parent(i)
        if heap[i] > heap[parent]:
            heap[i], heap[parent] = heap[parent], heap[i]
            i = parent
        else:
            break


def _left_child(i: int) -> int:
    """
    The left child of a node at index i in zero-based array is given by 2*i+1.
    """
    return 2 * i + 1


def _right_child(i: int) -> int:
    """
    The right child of a node at index i in zero-based array is given by 2*i+2.
    """
    return 2 * i + 2


def _parent(i: int) -> int:
    """
    The parent of a node at index i in zero-based array is given by (i-1) // 2.
    """
    return (i - 1) // 2


def _is_leaf(i: int, n: int) -> bool:
    """
    For a complete binary tree represented by a zero-based array of length n,
    the first leaf node is at position n//2. If the node at index i is greater
    or equal to n//2, it is a leaf node.
    """
    return i >= n // 2


def k_way_merge(*iterables: Iterable[T]) -> Iterator[T]:
    """
    k-way merge is a merge algorithm that, as the name suggests, merges k
    sorted lists into a single sorted list. It can be efficiently solved in
    O(nlog k) using a heap.
    """
    # Build a heap using the first element of each list. The generator and
    # iterator pattern is used to keep track of the "head" of each list. This
    # is a very elegant solution, which I stole from the original heapq
    # implementation. Basically, a node in the heap contains its value and its
    # next value, similar to a linked list. Each node has an additional "index"
    # that associates it with an iterable. This is required to resolve tuple
    # comparisons where the values of nodes are the same. When Python compares
    # tuples, it compares elements in order until it finds a difference. Using
    # an index, a node from a list with a lower index will take precedence over
    # a node with a higher index.
    heap: list[tuple[T, int, Iterator[T]]] = []
    for i, it in enumerate(map(iter, iterables)):
        try:
            e = next(it)
            heap.append((e, i, it))
        except StopIteration:
            pass
    heapify(heap)

    # For each selection, we retrieve the smallest element from the heap
    # (heap[0]), which contains a value and an iterator. The value is yielded,
    # and a new value is retrieved from the iterator with a call to `next()`.
    # The root node is removed from the heap and replaced with the next element
    # from the list from which the node was taken. If `next()` raises a
    # StopIteration exception, the iterator has been exhausted meaning the
    # previous element was the final element from the iterable. If this is the
    # case, we simply delete the smallest element from the heap.
    while heap:
        try:
            e, i, it = heap[0]
            yield e
            e = next(it)
            replace_min(heap, (e, i, it))
        except StopIteration:
            delete_min(heap)


def kth_smallest(l: list[T], k: int) -> T:
    """
    The kth smallest (or largest) problem can be efficiently solved using a
    max-heap (or min-heap). Maintaining a heap of size k, the kth smallest
    value is always the root of the max-heap. For all k smallest elements,
    simply return the sorted heap.
    """
    # Build a heap from the first k elements of the list.
    heap: list[T] = []
    for i in range(k):
        heap.append(l[i])
    heapify_max(heap)
    # Iterate over the remaining values in the list. If a value is less than
    # the largest value in the heap, replace it.
    for i in range(k, len(l)):
        if l[i] < heap[0]:
            replace_max(heap, l[i])
    return heap[0]


def kth_largest(l: list[T], k: int) -> T:
    """
    Same as `kth_smallest()`, but for the kth largest value.
    """
    heap: list[T] = []
    for i in range(k):
        heap.append(l[i])
    heapify(heap)
    for i in range(k, len(l)):
        if l[i] > heap[0]:
            replace_min(heap, l[i])
    return heap[0]


# Though the technical names for heap operation are insert, delete-min,
# make-heap, etc., their `heapq` equivalents are provided here.
heappush = insert
heappop = delete_min
heapreplace = replace_min
heapify = make_heap
heapify_max = make_heap_max
