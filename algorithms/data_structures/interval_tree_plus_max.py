from algorithms.data_structures.array_binary_tree_operation import *

class IntervalTreePlusMax:
    
    class TreeNode:
        def __init__(self, load = 0, max = 0):
            self.LOAD = load
            self.MAX = max

    def set_size(self, size):
        tmp = 1
        while tmp < size:
            tmp <<= 1

        self.size = tmp
        self.tree = [IntervalTreePlusMax.TreeNode() for i in range(self.size * 2)]

    def update_max(self, number):
        if number >= self.size:
            self.tree[number].MAX = self.tree[number].LOAD
        else:
            self.tree[number].MAX = max(self.tree[left_son(number)].MAX, self.tree[right_son(number)].MAX) + self.tree[number].LOAD

    def insert(self, value, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        self.tree[begin_pos].LOAD += value
        self.tree[end_pos].LOAD += value if begin_pos != end_pos else 0

        while is_in_tree(begin_pos):
            if parent(begin_pos) != parent(end_pos):
                    
                    if is_left_son(begin_pos):
                        self.tree[right_sibling(begin_pos)].LOAD += value
                        self.update_max(right_sibling(begin_pos))

                    if is_right_son(end_pos):
                        self.tree[left_sibling(end_pos)].LOAD += value
                        self.update_max(left_sibling(end_pos))
        
            self.update_max(begin_pos)
            self.update_max(end_pos)

            begin_pos = parent(begin_pos)
            end_pos = parent(end_pos)

    def request(self, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        left_max = 0
        right_max = 0

        while is_in_tree(begin_pos):
            left_max += self.tree[begin_pos].LOAD
            right_max += self.tree[end_pos].LOAD

            if parent(begin_pos) != parent(end_pos):
                
                if is_left_son(begin_pos):
                    left_max = max(left_max, self.tree[right_sibling(begin_pos)].MAX)

                if is_right_son(end_pos):
                    right_max = max(right_max, self.tree[left_sibling(end_pos)].MAX)

            begin_pos = parent(begin_pos)
            end_pos = parent(end_pos)
        
        return max(left_max, right_max)