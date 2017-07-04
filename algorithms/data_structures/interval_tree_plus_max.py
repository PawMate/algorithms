import algorithms.data_structures.array_binary_tree_operation as bto


class IntervalTreePlusMax:

    class TreeNode:
        def __init__(self, load=0, max=0):
            self.LOAD = load
            self.MAX = max

    def set_size(self, size):
        tmp = 1
        while tmp < size:
            tmp <<= 1

        self.size = tmp
        treeSize = self.size * 2
        self.tree = [IntervalTreePlusMax.TreeNode() for i in range(treeSize)]

    def update_max(self, number):
        if number >= self.size:
            self.tree[number].MAX = self.tree[number].LOAD
        else:
            maximalChildren = max(self.tree[bto.left_son(number)].MAX, 
                                  self.tree[bto.right_son(number)].MAX)
            self.tree[number].MAX = maximalChildren + self.tree[number].LOAD

    def insert(self, value, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        self.tree[begin_pos].LOAD += value
        self.tree[end_pos].LOAD += value if begin_pos != end_pos else 0

        while bto.is_in_tree(begin_pos):
            if bto.parent(begin_pos) != bto.parent(end_pos):

                    if bto.is_left_son(begin_pos):
                        self.tree[bto.right_sibling(begin_pos)].LOAD += value
                        self.update_max(bto.right_sibling(begin_pos))

                    if bto.is_right_son(end_pos):
                        self.tree[bto.left_sibling(end_pos)].LOAD += value
                        self.update_max(bto.left_sibling(end_pos))

            self.update_max(begin_pos)
            self.update_max(end_pos)

            begin_pos = bto.parent(begin_pos)
            end_pos = bto.parent(end_pos)


    def request(self, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        left_max = 0
        right_max = 0

        while bto.is_in_tree(begin_pos):
            left_max += self.tree[begin_pos].LOAD
            right_max += self.tree[end_pos].LOAD

            if bto.parent(begin_pos) != bto.parent(end_pos):
                
                if bto.is_left_son(begin_pos):
                    rsib = bto.right_sibling(begin_pos)
                    left_max = max(left_max, self.tree[rsib].MAX)

                if bto.is_right_son(end_pos):
                    lsib = bto.left_sibling(end_pos)
                    right_max = max(right_max, self.tree[lsib].MAX)

            begin_pos = bto.parent(begin_pos)
            end_pos = bto.parent(end_pos)

        return max(left_max, right_max)
