from array_binary_tree_operation import *

class IntervalTreePlusSum:
    
    class TreeNode:
        def __init__(self, load = 0, sub = 0):
            self.LOAD = load
            self.SUB = sub

    def set_size(self, size):
        tmp = 1
        while tmp < size:
            tmp <<= 1

        self.size = tmp
        self.tree = [IntervalTreePlusSum.TreeNode() for i in range(self.size * 2)]

    def update_sub(self, number, length):
        if number >= self.size:
            self.tree[number].SUB = self.tree[number].LOAD
        else:
            self.tree[number].SUB = self.tree[left_son(number)].SUB + self.tree[right_son(number)].SUB + self.tree[number].LOAD * length

    def insert(self, value, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        length = 1
        self.tree[begin_pos].LOAD += value
        self.tree[end_pos].LOAD += value if begin_pos != end_pos else 0

        while is_in_tree(begin_pos):
            if parent(begin_pos) != parent(end_pos):
                    
                    if is_left_son(begin_pos):
                        self.tree[right_sibling(begin_pos)].LOAD += value
                        self.update_sub(right_sibling(begin_pos), length)

                    if is_right_son(end_pos):
                        self.tree[left_sibling(end_pos)].LOAD += value
                        self.update_sub(left_sibling(end_pos), length)
        
            self.update_sub(begin_pos, length)
            self.update_sub(end_pos, length)

            length <<= 1
            begin_pos = parent(begin_pos)
            end_pos = parent(end_pos)

    def request(self, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        left_path = 1
        right_path = 0 if begin_pos == end_pos else 1
        length = 1
        result = 0

        while is_in_tree(begin_pos):
            result += left_path * self.tree[begin_pos].LOAD
            result += right_path * self.tree[end_pos].LOAD

            if parent(begin_pos) != parent(end_pos):
                
                if is_left_son(begin_pos):
                    result += self.tree[right_sibling(begin_pos)].SUB
                    left_path += length
                
                if is_right_son(end_pos):
                    result += self.tree[left_sibling(end_pos)].SUB
                    right_path += length
            
            begin_pos = parent(begin_pos)
            end_pos = parent(end_pos)
            length <<= 1
        
        return result