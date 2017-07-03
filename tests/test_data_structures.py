from random import shuffle
import unittest
import random

from algorithms.data_structures import (
    binary_search_tree,
    digraph,
    queue,
    singly_linked_list,
    stack,
    undirected_graph,
    union_find,
    union_find_by_rank,
    union_find_with_path_compression,
    lcp_array
)
from algorithms.data_structures.array_binary_tree_operation import *
from algorithms.data_structures.interval_tree_plus_max import IntervalTreePlusMax
from algorithms.data_structures.interval_tree_plus_sum import IntervalTreePlusSum

class TestBinarySearchTree(unittest.TestCase):
    """
    Test Binary Search Tree Implementation
    """
    key_val = [
        ("a", 1), ("b", 2), ("c", 3),
        ("d", 4), ("e", 5), ("f", 6),
        ("g", 7), ("h", 8), ("i", 9)
    ]

    def shuffle_list(self, ls):
        shuffle(ls)
        return ls

    def test_size(self):
        # Size starts at 0
        self.bst = binary_search_tree.BinarySearchTree()
        self.assertEqual(self.bst.size(), 0)
        # Doing a put increases the size to 1
        self.bst.put("one", 1)
        self.assertEqual(self.bst.size(), 1)
        # Putting a key that is already in doesn't change size
        self.bst.put("one", 1)
        self.assertEqual(self.bst.size(), 1)
        self.bst.put("one", 2)
        self.assertEqual(self.bst.size(), 1)

        self.bst = binary_search_tree.BinarySearchTree()
        size = 0
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
            size += 1
            self.assertEqual(self.bst.size(), size)

        shuffled = self.shuffle_list(self.key_val[:])

        self.bst = binary_search_tree.BinarySearchTree()
        size = 0
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
            size += 1
            self.assertEqual(self.bst.size(), size)

    def test_is_empty(self):
        self.bst = binary_search_tree.BinarySearchTree()
        self.assertTrue(self.bst.is_empty())
        self.bst.put("a", 1)
        self.assertFalse(self.bst.is_empty())

    def test_get(self):
        self.bst = binary_search_tree.BinarySearchTree()
        # Getting a key not in BST returns None
        self.assertEqual(self.bst.get("one"), None)

        # Get with a present key returns proper value
        self.bst.put("one", 1)
        self.assertEqual(self.bst.get("one"), 1)

        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
            self.assertEqual(self.bst.get(k), v)

        shuffled = self.shuffle_list(self.key_val[:])

        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
            self.assertEqual(self.bst.get(k), v)

    def test_contains(self):
        self.bst = binary_search_tree.BinarySearchTree()
        self.assertFalse(self.bst.contains("a"))
        self.bst.put("a", 1)
        self.assertTrue(self.bst.contains("a"))

    def test_put(self):
        self.bst = binary_search_tree.BinarySearchTree()

        # When BST is empty first put becomes root
        self.bst.put("bbb", 1)
        self.assertEqual(self.bst.root.key, "bbb")
        self.assertEqual(self.bst.root.left, None)

        # Adding a key greater than root doesn't update the left tree
        # but does update the right
        self.bst.put("ccc", 2)
        self.assertEqual(self.bst.root.key, "bbb")
        self.assertEqual(self.bst.root.left, None)
        self.assertEqual(self.bst.root.right.key, "ccc")

        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("bbb", 1)
        # Adding a key less than root doesn't update the right tree
        # but does update the left
        self.bst.put("aaa", 2)
        self.assertEqual(self.bst.root.key, "bbb")
        self.assertEqual(self.bst.root.right, None)
        self.assertEqual(self.bst.root.left.key, "aaa")

        self.bst = binary_search_tree.BinarySearchTree()
        size = 0
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
            size += 1
            self.assertEqual(self.bst.get(k), v)
            self.assertEqual(self.bst.size(), size)

        self.bst = binary_search_tree.BinarySearchTree()

        shuffled = self.shuffle_list(self.key_val[:])

        size = 0
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
            size += 1
            self.assertEqual(self.bst.get(k), v)
            self.assertEqual(self.bst.size(), size)

    def test_min_key(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val[::-1]:
            k, v = pair
            self.bst.put(k, v)
            self.assertEqual(self.bst.min_key(), k)

        shuffled = self.shuffle_list(self.key_val[:])

        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
        self.assertEqual(self.bst.min_key(), "a")

    def test_max_key(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
            self.assertEqual(self.bst.max_key(), k)

        shuffled = self.shuffle_list(self.key_val[:])

        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
        self.assertEqual(self.bst.max_key(), "i")

    def test_floor_key(self):
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.bst.put("c", 3)
        self.bst.put("e", 5)
        self.bst.put("g", 7)
        self.assertEqual(self.bst.floor_key("a"), "a")
        self.assertEqual(self.bst.floor_key("b"), "a")
        self.assertEqual(self.bst.floor_key("g"), "g")
        self.assertEqual(self.bst.floor_key("h"), "g")

        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("c", 3)
        self.bst.put("e", 5)
        self.bst.put("a", 1)
        self.bst.put("g", 7)
        self.assertEqual(self.bst.floor_key("a"), "a")
        self.assertEqual(self.bst.floor_key("b"), "a")
        self.assertEqual(self.bst.floor_key("g"), "g")
        self.assertEqual(self.bst.floor_key("h"), "g")

    def test_ceiling_key(self):
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.bst.put("c", 3)
        self.bst.put("e", 5)
        self.bst.put("g", 7)
        self.assertEqual(self.bst.ceiling_key("a"), "a")
        self.assertEqual(self.bst.ceiling_key("b"), "c")
        self.assertEqual(self.bst.ceiling_key("g"), "g")
        self.assertEqual(self.bst.ceiling_key("f"), "g")

        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("c", 3)
        self.bst.put("e", 5)
        self.bst.put("a", 1)
        self.bst.put("g", 7)
        self.assertEqual(self.bst.ceiling_key("a"), "a")
        self.assertEqual(self.bst.ceiling_key("b"), "c")
        self.assertEqual(self.bst.ceiling_key("g"), "g")
        self.assertEqual(self.bst.ceiling_key("f"), "g")

    def test_select_key(self):
        shuffled = self.shuffle_list(self.key_val[:])

        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
        self.assertEqual(self.bst.select_key(0), "a")
        self.assertEqual(self.bst.select_key(1), "b")
        self.assertEqual(self.bst.select_key(2), "c")

    def test_rank(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)

        self.assertEqual(self.bst.rank("a"), 0)
        self.assertEqual(self.bst.rank("b"), 1)
        self.assertEqual(self.bst.rank("c"), 2)
        self.assertEqual(self.bst.rank("d"), 3)

        shuffled = self.shuffle_list(self.key_val[:])
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)

        self.assertEqual(self.bst.rank("a"), 0)
        self.assertEqual(self.bst.rank("b"), 1)
        self.assertEqual(self.bst.rank("c"), 2)
        self.assertEqual(self.bst.rank("d"), 3)

    def test_delete_min(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
        for i in range(self.bst.size() - 1):
            self.bst.delete_min()
            self.assertEqual(self.bst.min_key(), self.key_val[i+1][0])
        self.bst.delete_min()
        self.assertEqual(self.bst.min_key(), None)

        shuffled = self.shuffle_list(self.key_val[:])
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
        for i in range(self.bst.size() - 1):
            self.bst.delete_min()
            self.assertEqual(self.bst.min_key(), self.key_val[i+1][0])
        self.bst.delete_min()
        self.assertEqual(self.bst.min_key(), None)

    def test_delete_max(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
        for i in range(self.bst.size() - 1, 0, -1):
            self.bst.delete_max()
            self.assertEqual(self.bst.max_key(), self.key_val[i-1][0])
        self.bst.delete_max()
        self.assertEqual(self.bst.max_key(), None)

        shuffled = self.shuffle_list(self.key_val[:])

        for pair in shuffled:
            k, v = pair
            self.bst.put(k, v)
        for i in range(self.bst.size() - 1, 0, -1):
            self.bst.delete_max()
            self.assertEqual(self.bst.max_key(), self.key_val[i-1][0])
        self.bst.delete_max()
        self.assertEqual(self.bst.max_key(), None)

    def test_delete(self):
        # delete key from an empty bst
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.delete("a")
        self.assertEqual(self.bst.root, None)
        self.assertEqual(self.bst.size(), 0)

        # delete key not present in bst
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.bst.delete("b")
        self.assertEqual(self.bst.root.key, "a")
        self.assertEqual(self.bst.size(), 1)

        # delete key when bst only contains one key
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.assertEqual(self.bst.root.key, "a")
        self.bst.delete("a")
        self.assertEqual(self.bst.root, None)
        self.assertEqual(self.bst.size(), 0)

        # delete parent key when it only has a left child
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("b", 2)
        self.bst.put("a", 1)
        self.assertEqual(self.bst.root.left.key, "a")
        self.bst.delete("b")
        self.assertEqual(self.bst.root.key, "a")
        self.assertEqual(self.bst.size(), 1)

        # delete parent key when it only has a right child
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.bst.put("b", 2)
        self.assertEqual(self.bst.root.right.key, "b")
        self.bst.delete("a")
        self.assertEqual(self.bst.root.key, "b")
        self.assertEqual(self.bst.size(), 1)

        # delete left child key
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("b", 2)
        self.bst.put("a", 1)
        self.assertEqual(self.bst.root.left.key, "a")
        self.bst.delete("a")
        self.assertEqual(self.bst.root.key, "b")
        self.assertEqual(self.bst.size(), 1)

        # delete right child key
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("a", 1)
        self.bst.put("b", 2)
        self.assertEqual(self.bst.root.right.key, "b")
        self.bst.delete("b")
        self.assertEqual(self.bst.root.key, "a")
        self.assertEqual(self.bst.size(), 1)

        # delete parent key when it has a left and right child
        self.bst = binary_search_tree.BinarySearchTree()
        self.bst.put("b", 2)
        self.bst.put("a", 1)
        self.bst.put("c", 3)
        self.bst.delete("b")
        self.assertEqual(self.bst.root.key, "c")
        self.assertEqual(self.bst.size(), 2)

    def test_keys(self):
        self.bst = binary_search_tree.BinarySearchTree()
        for pair in self.key_val:
            k, v = pair
            self.bst.put(k, v)
        self.assertEqual(
            self.bst.keys(),
            ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        )


class TestDirectedGraph(unittest.TestCase):
    """
    Test Undirected Graph Implementation
    """
    def test_directed_graph(self):

        # init
        self.dg0 = digraph.Digraph()
        self.dg1 = digraph.Digraph()
        self.dg2 = digraph.Digraph()
        self.dg3 = digraph.Digraph()

        # populating
        self.dg1.add_edge(1, 2)

        self.dg1_rev = self.dg1.reverse()  # reverse

        self.dg2.add_edge(1, 2)
        self.dg2.add_edge(1, 2)

        self.dg3.add_edge(1, 2)
        self.dg3.add_edge(1, 2)
        self.dg3.add_edge(3, 1)

        # test adj
        self.assertTrue(2 in self.dg1.adj(1))
        self.assertEqual(len(self.dg1.adj(1)), 1)
        self.assertTrue(1 not in self.dg1.adj(2))
        self.assertEqual(len(self.dg1.adj(2)), 0)

        self.assertTrue(1 in self.dg1_rev.adj(2))
        self.assertEqual(len(self.dg1_rev.adj(2)), 1)
        self.assertTrue(2 not in self.dg1_rev.adj(1))
        self.assertEqual(len(self.dg1_rev.adj(1)), 0)

        self.assertTrue(2 in self.dg2.adj(1))
        self.assertEqual(len(self.dg2.adj(1)), 2)
        self.assertTrue(1 not in self.dg2.adj(2))
        self.assertEqual(len(self.dg2.adj(2)), 0)

        self.assertTrue(2 in self.dg3.adj(1))
        self.assertTrue(1 in self.dg3.adj(3))
        self.assertEqual(len(self.dg3.adj(1)), 2)
        self.assertTrue(1 not in self.dg3.adj(2))
        self.assertEqual(len(self.dg3.adj(2)), 0)
        self.assertTrue(3 not in self.dg3.adj(1))
        self.assertEqual(len(self.dg3.adj(3)), 1)

        # test degree
        self.assertEqual(self.dg1.outdegree(1), 1)
        self.assertEqual(self.dg1.outdegree(2), 0)

        self.assertEqual(self.dg1_rev.outdegree(2), 1)
        self.assertEqual(self.dg1_rev.outdegree(1), 0)

        self.assertEqual(self.dg2.outdegree(1), 2)
        self.assertEqual(self.dg2.outdegree(2), 0)

        self.assertEqual(self.dg3.outdegree(1), 2)
        self.assertEqual(self.dg3.outdegree(2), 0)
        self.assertEqual(self.dg3.outdegree(3), 1)

        # test vertices
        self.assertEqual(list(self.dg0.vertices()), [])
        self.assertEqual(len(self.dg0.vertices()), 0)

        self.assertTrue(1 in self.dg1.vertices())
        self.assertTrue(2 in self.dg1.vertices())
        self.assertEqual(len(self.dg1.vertices()), 2)

        self.assertTrue(2 in self.dg1_rev.vertices())
        self.assertTrue(1 in self.dg1_rev.vertices())
        self.assertEqual(len(self.dg1_rev.vertices()), 2)

        self.assertTrue(1 in self.dg2.vertices())
        self.assertTrue(2 in self.dg2.vertices())
        self.assertEqual(len(self.dg2.vertices()), 2)

        self.assertTrue(1 in self.dg3.vertices())
        self.assertTrue(2 in self.dg3.vertices())
        self.assertTrue(3 in self.dg3.vertices())
        self.assertEqual(len(self.dg3.vertices()), 3)

        # test vertex_count
        self.assertEqual(self.dg0.vertex_count(), 0)
        self.assertEqual(self.dg1.vertex_count(), 2)
        self.assertEqual(self.dg1_rev.vertex_count(), 2)
        self.assertEqual(self.dg2.vertex_count(), 2)
        self.assertEqual(self.dg3.vertex_count(), 3)

        # test edge_count
        self.assertEqual(self.dg0.edge_count(), 0)
        self.assertEqual(self.dg1.edge_count(), 1)
        self.assertEqual(self.dg1_rev.edge_count(), 1)
        self.assertEqual(self.dg2.edge_count(), 2)
        self.assertEqual(self.dg3.edge_count(), 3)


class TestQueue(unittest.TestCase):
    """
    Test Queue Implementation
    """
    def test_queue(self):
        self.que = queue.Queue()
        self.que.add(1)
        self.que.add(2)
        self.que.add(8)
        self.que.add(5)
        self.que.add(6)

        self.assertEqual(self.que.remove(), 1)
        self.assertEqual(self.que.size(), 4)
        self.assertEqual(self.que.remove(), 2)
        self.assertEqual(self.que.remove(), 8)
        self.assertEqual(self.que.remove(), 5)
        self.assertEqual(self.que.remove(), 6)
        self.assertEqual(self.que.is_empty(), True)


class TestSinglyLinkedList(unittest.TestCase):
    """
    Test Singly Linked List Implementation
    """

    def test_singly_linked_list(self):
        self.sl = singly_linked_list.SinglyLinkedList()
        self.sl.add(10)
        self.sl.add(5)
        self.sl.add(30)
        self.sl.remove(30)

        self.assertEqual(self.sl.size, 2)
        self.assertEqual(self.sl.search(30), False)
        self.assertEqual(self.sl.search(5), True)
        self.assertEqual(self.sl.search(10), True)
        self.assertEqual(self.sl.remove(5), True)
        self.assertEqual(self.sl.remove(10), True)
        self.assertEqual(self.sl.size, 0)


class TestStack(unittest.TestCase):
    """
    Test Stack Implementation
    """
    def test_stack(self):
        self.sta = stack.Stack()
        self.sta.add(5)
        self.sta.add(8)
        self.sta.add(10)
        self.sta.add(2)

        self.assertEqual(self.sta.remove(), 2)
        self.assertEqual(self.sta.is_empty(), False)
        self.assertEqual(self.sta.size(), 3)


class TestUndirectedGraph(unittest.TestCase):
    """
    Test Undirected Graph Implementation
    """
    def test_undirected_graph(self):

        # init
        self.ug0 = undirected_graph.Undirected_Graph()
        self.ug1 = undirected_graph.Undirected_Graph()
        self.ug2 = undirected_graph.Undirected_Graph()
        self.ug3 = undirected_graph.Undirected_Graph()

        # populating
        self.ug1.add_edge(1, 2)

        self.ug2.add_edge(1, 2)
        self.ug2.add_edge(1, 2)

        self.ug3.add_edge(1, 2)
        self.ug3.add_edge(1, 2)
        self.ug3.add_edge(3, 1)

        # test adj
        self.assertTrue(2 in self.ug1.adj(1))
        self.assertEqual(len(self.ug1.adj(1)), 1)
        self.assertTrue(1 in self.ug1.adj(2))
        self.assertEqual(len(self.ug1.adj(1)), 1)

        self.assertTrue(2 in self.ug2.adj(1))
        self.assertEqual(len(self.ug2.adj(1)), 2)
        self.assertTrue(1 in self.ug2.adj(2))
        self.assertEqual(len(self.ug2.adj(1)), 2)

        self.assertTrue(2 in self.ug3.adj(1))
        self.assertTrue(3 in self.ug3.adj(1))
        self.assertEqual(len(self.ug3.adj(1)), 3)
        self.assertTrue(1 in self.ug3.adj(2))
        self.assertEqual(len(self.ug3.adj(2)), 2)
        self.assertTrue(1 in self.ug3.adj(3))
        self.assertEqual(len(self.ug3.adj(3)), 1)

        # test degree
        self.assertEqual(self.ug1.degree(1), 1)
        self.assertEqual(self.ug1.degree(2), 1)
        self.assertEqual(self.ug2.degree(1), 2)
        self.assertEqual(self.ug2.degree(2), 2)
        self.assertEqual(self.ug3.degree(1), 3)
        self.assertEqual(self.ug3.degree(2), 2)
        self.assertEqual(self.ug3.degree(3), 1)

        # test vertices
        self.assertEqual(list(self.ug0.vertices()), [])
        self.assertEqual(len(self.ug0.vertices()), 0)

        self.assertTrue(1 in self.ug1.vertices())
        self.assertTrue(2 in self.ug1.vertices())
        self.assertEqual(len(self.ug1.vertices()), 2)

        self.assertTrue(1 in self.ug2.vertices())
        self.assertTrue(2 in self.ug2.vertices())
        self.assertEqual(len(self.ug2.vertices()), 2)

        self.assertTrue(1 in self.ug3.vertices())
        self.assertTrue(2 in self.ug3.vertices())
        self.assertTrue(3 in self.ug3.vertices())
        self.assertEqual(len(self.ug3.vertices()), 3)

        # test vertex_count
        self.assertEqual(self.ug0.vertex_count(), 0)
        self.assertEqual(self.ug1.vertex_count(), 2)
        self.assertEqual(self.ug2.vertex_count(), 2)
        self.assertEqual(self.ug3.vertex_count(), 3)

        # test edge_count
        self.assertEqual(self.ug0.edge_count(), 0)
        self.assertEqual(self.ug1.edge_count(), 1)
        self.assertEqual(self.ug2.edge_count(), 2)
        self.assertEqual(self.ug3.edge_count(), 3)


class TestUnionFind(unittest.TestCase):
    """
    Test Union Find Implementation
    """
    def test_union_find(self):
        self.uf = union_find.UnionFind(4)
        self.uf.make_set(4)
        self.uf.union(1, 0)
        self.uf.union(3, 4)

        self.assertEqual(self.uf.find(1), 0)
        self.assertEqual(self.uf.find(3), 4)
        self.assertEqual(self.uf.is_connected(0, 1), True)
        self.assertEqual(self.uf.is_connected(3, 4), True)


class TestUnionFindByRank(unittest.TestCase):
    """
    Test Union Find Implementation
    """
    def test_union_find_by_rank(self):
        self.uf = union_find_by_rank.UnionFindByRank(6)
        self.uf.make_set(6)
        self.uf.union(1, 0)
        self.uf.union(3, 4)
        self.uf.union(2, 4)
        self.uf.union(5, 2)
        self.uf.union(6, 5)

        self.assertEqual(self.uf.find(1), 1)
        self.assertEqual(self.uf.find(3), 3)
        # test tree is created by rank
        self.uf.union(5, 0)
        self.assertEqual(self.uf.find(2), 3)
        self.assertEqual(self.uf.find(5), 3)
        self.assertEqual(self.uf.find(6), 3)
        self.assertEqual(self.uf.find(0), 3)

        self.assertEqual(self.uf.is_connected(0, 1), True)
        self.assertEqual(self.uf.is_connected(3, 4), True)
        self.assertEqual(self.uf.is_connected(5, 3), True)


class TestUnionFindWithPathCompression(unittest.TestCase):
    """
    Test Union Find Implementation
    """
    def test_union_find_with_path_compression(self):
        self.uf = (
            union_find_with_path_compression
            .UnionFindWithPathCompression(5)
        )

        self.uf.make_set(5)
        self.uf.union(0, 1)
        self.uf.union(2, 3)
        self.uf.union(1, 3)
        self.uf.union(4, 5)
        self.assertEqual(self.uf.find(1), 0)
        self.assertEqual(self.uf.find(3), 0)
        self.assertEqual(self.uf.parent(3), 2)
        self.assertEqual(self.uf.parent(5), 4)
        self.assertEqual(self.uf.is_connected(3, 5), False)
        self.assertEqual(self.uf.is_connected(4, 5), True)
        self.assertEqual(self.uf.is_connected(2, 3), True)
        # test tree is created by path compression
        self.uf.union(5, 3)
        self.assertEqual(self.uf.parent(3), 0)

        self.assertEqual(self.uf.is_connected(3, 5), True)


class TestLCPSuffixArrays(unittest.TestCase):
    def setUp(self):
        super(TestLCPSuffixArrays, self).setUp()
        self.case_1 = "aaaaaa"
        self.s_array_1 = [5, 4, 3, 2, 1, 0]
        self.rank_1 = [5, 4, 3, 2, 1, 0]
        self.lcp_1 = [1, 2, 3, 4, 5, 0]

        self.case_2 = "abcabcdd"
        self.s_array_2 = [0, 2, 4, 1, 3, 5, 7, 6]
        self.rank_2 = [0, 3, 1, 4, 2, 5, 7, 6]
        self.lcp_2 = [3, 0, 2, 0, 1, 0, 1, 0]

        self.case_3 = "kmckirrrmppp"
        self.s_array_3 = [3, 4, 0, 2, 1, 11, 10, 9, 5, 8, 7, 6]
        self.rank_3 = [2, 4, 3, 0, 1, 8, 11, 10, 9, 7, 6, 5]
        self.lcp_3 = [0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0]

    def test_lcp_array(self):
        lcp = lcp_array.lcp_array(self.case_1, self.s_array_1, self.rank_1)
        self.assertEqual(lcp, self.lcp_1)

        lcp = lcp_array.lcp_array(self.case_2, self.s_array_2, self.rank_2)
        self.assertEqual(lcp, self.lcp_2)

        lcp = lcp_array.lcp_array(self.case_3, self.s_array_3, self.rank_3)
        self.assertEqual(lcp, self.lcp_3)

    def test_suffix_array(self):
        s_array, rank = lcp_array.suffix_array(self.case_1)
        self.assertEqual(s_array, self.s_array_1)
        self.assertEqual(rank, self.rank_1)

        s_array, rank = lcp_array.suffix_array(self.case_2)
        self.assertEqual(s_array, self.s_array_2)
        self.assertEqual(rank, self.rank_2)

        s_array, rank = lcp_array.suffix_array(self.case_3)
        self.assertEqual(s_array, self.s_array_3)
        self.assertEqual(rank, self.rank_3)

		
class Test_array_binary_tee_operation_tests(unittest.TestCase):
    def test_is_left_son(self):
        expected_left_sons = [2,4,130,240]
        for left_son in expected_left_sons:
            self.assertTrue(is_left_son(left_son))

    def test_is_right_son(self):
        expected_right_sons = [3,5,7,321,999]
        for right_son in expected_right_sons:
            self.assertTrue(is_right_son(right_son))

    def test_get_left_son(self):
        parents = [0,1,2,65,120]
        left_sons_for_test = [0,2,4,130,240]
        for node_id in range(len(parents)):
            self.assertTrue(left_son(parents[node_id]) == left_sons_for_test[node_id])

    def test_get_right_son(self):
        parents = [0,1,2,65,120]
        right_sons_for_test = [1,3,5,131,241]
        for node_id in range(len(parents)):
            self.assertTrue(right_son(parents[node_id]) == right_sons_for_test[node_id])

    def test_get_left_sibiling(self):
        nodes = [7,11,23,65,121]
        left_sibiling_for_test = [6,10,22,64,120]
        for node_id in range(len(nodes)):
            self.assertTrue(left_sibling(nodes[node_id]) == left_sibiling_for_test[node_id])

    def test_get_right_sibiling(self):
        nodes = [6,12,20,60,122]
        right_sibiling_for_test = [7,13,21,61,123]
        for node_id in range(len(nodes)):
            self.assertTrue(right_sibling(nodes[node_id]) == right_sibiling_for_test[node_id])

    def test_get_right_sibiling(self):
        nodes = [6,12,20,61,123]
        expectedParents = [3,6,10,30,61]
        for node_id in range(len(nodes)):
            self.assertTrue(parent(nodes[node_id]) == expectedParents[node_id])


class IntervalPlusMaxBrutalSolution:
    def set_size(self, size):
        self.size = size
        self.a = [0] * self.size

    def insert(self, value, begin_pos, end_pos):
        for i in range(begin_pos, end_pos + 1):
            self.a[i] += value

    def request(self, begin_pos, end_pos):
        result = self.a[begin_pos]
        for i in range(begin_pos, end_pos + 1):
            result = max(result, self.a[i])
        return result

		
class Interval_tree_plus_max_tests(unittest.TestCase):
    def test_setSizePowerOfTwo(self):
        test_object = IntervalTreePlusMax()
        expected_size = 8
        test_object.set_size(expected_size)
        self.assertEqual(test_object.size, expected_size)
        self.assertEqual(len(test_object.tree), expected_size * 2)

    def test_setSizeNonPowerOfTwo(self):
        test_object = IntervalTreePlusMax()
        expected_size = 13
        closest_power_of_two = 16
        test_object.set_size(expected_size)
        self.assertEqual(test_object.size, closest_power_of_two)
        self.assertEqual(len(test_object.tree), closest_power_of_two * 2)

    def test_updateMaxLeaf(self):
        test_object = IntervalTreePlusMax()
        test_object.set_size(8)
        test_object.tree[15].LOAD = 11
        test_object.update_max(15)
        self.assertEqual(test_object.tree[15].MAX, 11)

    def test_updateMaxNotLeaf(self):
        test_object = IntervalTreePlusMax()
        test_object.set_size(8)
        test_object.tree[1].LOAD = 10
        test_object.tree[left_son(1)].MAX = 2
        test_object.tree[right_son(1)].MAX = 7
        test_object.update_max(1)
        self.assertEqual(test_object.tree[1].MAX, 17)

    def test_insert(self):
        test_object = IntervalTreePlusMax()
        test_object.set_size(8)
        test_object.insert(5, 2, 5)
        test_object.insert(3, 0, 4)
        test_object.insert(7, 3, 7)

        self.assertEqual(test_object.tree[1].MAX, 15)
        self.assertEqual(test_object.tree[2].MAX, 15)
        self.assertEqual(test_object.tree[3].MAX, 15)
        self.assertEqual(test_object.tree[4].MAX, 3 )
        self.assertEqual(test_object.tree[5].MAX, 15)
        self.assertEqual(test_object.tree[6].MAX, 15)
        self.assertEqual(test_object.tree[7].MAX, 7 )
        self.assertEqual(test_object.tree[8].MAX, 3 )
        self.assertEqual(test_object.tree[9].MAX, 3 )
        self.assertEqual(test_object.tree[10].MAX, 5 )
        self.assertEqual(test_object.tree[11].MAX, 12)
        self.assertEqual(test_object.tree[12].MAX, 8 )
        self.assertEqual(test_object.tree[13].MAX, 5 )
        self.assertEqual(test_object.tree[14].MAX, 7 )
        self.assertEqual(test_object.tree[15].MAX, 7 )

        self.assertEqual(test_object.tree[1].LOAD, 0 )
        self.assertEqual(test_object.tree[2].LOAD, 0 )
        self.assertEqual(test_object.tree[3].LOAD, 0 )
        self.assertEqual(test_object.tree[4].LOAD, 0 )
        self.assertEqual(test_object.tree[5].LOAD, 3 )
        self.assertEqual(test_object.tree[6].LOAD, 7 )
        self.assertEqual(test_object.tree[7].LOAD, 0 )
        self.assertEqual(test_object.tree[8].LOAD, 3 )
        self.assertEqual(test_object.tree[9].LOAD, 3 )
        self.assertEqual(test_object.tree[10].LOAD, 5 )
        self.assertEqual(test_object.tree[11].LOAD, 12)
        self.assertEqual(test_object.tree[12].LOAD, 8 )
        self.assertEqual(test_object.tree[13].LOAD, 5 )
        self.assertEqual(test_object.tree[14].LOAD, 7 )
        self.assertEqual(test_object.tree[15].LOAD, 7 )

    def test_request(self):
        test_object = IntervalTreePlusMax()
        test_object.set_size(8)
        test_object.insert(5, 2, 5)
        test_object.insert(3, 0, 4)
        test_object.insert(7, 3, 7)

        self.assertEqual(test_object.request(3, 3), 15)
        self.assertEqual(test_object.request(7, 7), 7)
        self.assertEqual(test_object.request(2, 2), 8)
        self.assertEqual(test_object.request(2, 3), 15)
        self.assertEqual(test_object.request(0, 2), 8)

    def test_randomUseCaseCompariveBrute(self):
        test_object = IntervalTreePlusMax()
        brute_object = IntervalPlusMaxBrutalSolution()
        size = random.randint(10, 1000)
        test_object.set_size(size)
        brute_object.set_size(size)
        for i in range(1000):
            if random.random() >= 0.5:
                value = random.randint(-1000, 1000)
                begin = random.randint(0, size - 1)
                end = random.randint(begin, size - 1)
                test_object.insert(value, begin, end)
                brute_object.insert(value, begin, end)
            else:
                begin = random.randint(0, size - 1)
                end = random.randint(begin, size - 1)
                test_answer = test_object.request(begin, end)
                brute_answer = brute_object.request(begin, end)
                self.assertEqual(test_answer, brute_answer)


class IntervalPlusSumBrutalSolution:
    def set_size(self, size):
        self.size = size
        self.a = [0] * self.size

    def insert(self, value, begin_pos, end_pos):
        for i in range(begin_pos, end_pos + 1):
            self.a[i] += value

    def request(self, begin_pos, end_pos):
        result = 0
        for i in range(begin_pos, end_pos + 1):
            result += self.a[i]
        return result

		
class Interval_tree_plus_max_tests(unittest.TestCase):
    def test_setSizePowerOfTwo(self):
        test_object = IntervalTreePlusSum()
        expected_size = 512
        test_object.set_size(expected_size)
        self.assertEqual(test_object.size, expected_size)
        self.assertEqual(len(test_object.tree), expected_size * 2)

    def test_setSizeNonPowerOfTwo(self):
        test_object = IntervalTreePlusSum()
        expected_size = 999
        closest_power_of_two = 1024
        test_object.set_size(expected_size)
        self.assertEqual(test_object.size, closest_power_of_two)
        self.assertEqual(len(test_object.tree), closest_power_of_two * 2)

    def test_updateSubLeaf(self):
        test_object = IntervalTreePlusSum()
        test_object.set_size(16)
        test_object.tree[31].LOAD = 11
        test_object.update_sub(31, 1)
        self.assertEqual(test_object.tree[31].SUB, 11)

    def test_updateSubNotLeaf(self):
        test_object = IntervalTreePlusSum()
        test_object.set_size(16)
        test_object.tree[1].LOAD = 10
        test_object.tree[left_son(1)].SUB = 2
        test_object.tree[right_son(1)].SUB = 7
        test_object.update_sub(1, 16)
        self.assertEqual(test_object.tree[1].SUB, 169)

    def test_insert(self):
        test_object = IntervalTreePlusSum()
        test_object.set_size(8)
        test_object.insert(12, 4, 6)
        test_object.insert(34, 0, 7)
        test_object.insert(2, 1, 1)

        self.assertEqual(test_object.tree[1].SUB, 310)
        self.assertEqual(test_object.tree[2].SUB, 138)
        self.assertEqual(test_object.tree[3].SUB, 172)
        self.assertEqual(test_object.tree[4].SUB, 70)
        self.assertEqual(test_object.tree[5].SUB, 68)
        self.assertEqual(test_object.tree[6].SUB, 92)
        self.assertEqual(test_object.tree[7].SUB, 80)
        self.assertEqual(test_object.tree[8].SUB, 34)
        self.assertEqual(test_object.tree[9].SUB, 36)
        self.assertEqual(test_object.tree[10].SUB, 0)
        self.assertEqual(test_object.tree[11].SUB, 0)
        self.assertEqual(test_object.tree[12].SUB, 12)
        self.assertEqual(test_object.tree[13].SUB, 12)
        self.assertEqual(test_object.tree[14].SUB, 46)
        self.assertEqual(test_object.tree[15].SUB, 34)

        self.assertEqual(test_object.tree[1].LOAD, 0)
        self.assertEqual(test_object.tree[2].LOAD, 0)
        self.assertEqual(test_object.tree[3].LOAD, 0)
        self.assertEqual(test_object.tree[4].LOAD, 0)
        self.assertEqual(test_object.tree[5].LOAD, 34)
        self.assertEqual(test_object.tree[6].LOAD, 34)
        self.assertEqual(test_object.tree[7].LOAD, 0)
        self.assertEqual(test_object.tree[8].LOAD, 34)
        self.assertEqual(test_object.tree[9].LOAD, 36)
        self.assertEqual(test_object.tree[10].LOAD, 0)
        self.assertEqual(test_object.tree[11].LOAD, 0)
        self.assertEqual(test_object.tree[12].LOAD, 12)
        self.assertEqual(test_object.tree[13].LOAD, 12)
        self.assertEqual(test_object.tree[14].LOAD, 46)
        self.assertEqual(test_object.tree[15].LOAD, 34)

    def test_request(self):
        test_object = IntervalTreePlusSum()
        test_object.set_size(8)
        test_object.insert(5, 2, 5)
        test_object.insert(3, 0, 4)
        test_object.insert(7, 3, 7)

        self.assertEqual(test_object.request(3, 3), 15)
        self.assertEqual(test_object.request(7, 7), 7)
        self.assertEqual(test_object.request(2, 2), 8)
        self.assertEqual(test_object.request(2, 3), 23)
        self.assertEqual(test_object.request(0, 2), 14)

    def test_randomUseCaseCompariveBrute(self):
        test_object = IntervalTreePlusSum()
        brute_object = IntervalPlusSumBrutalSolution()
        size = random.randint(10, 1000)
        test_object.set_size(size)
        brute_object.set_size(size)
        for i in range(1000):
            if random.random() >= 0.5:
                value = random.randint(-1000, 1000)
                begin = random.randint(0, size - 1)
                end = random.randint(begin, size - 1)
                test_object.insert(value, begin, end)
                brute_object.insert(value, begin, end)
            else:
                begin = random.randint(0, size - 1)
                end = random.randint(begin, size - 1)
                test_answer = test_object.request(begin, end)
                brute_answer = brute_object.request(begin, end)
                self.assertEqual(test_answer, brute_answer)
				