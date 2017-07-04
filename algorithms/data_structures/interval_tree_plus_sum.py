import algorithms.data_structures.array_binary_tree_operation as bto


class IntervalTreePlusSum:

    class TreeNode:
        def __init__(self, load=0, sub=0):
            self.LOAD = load
            self.SUB = sub

    def set_size(self, size):
        tmp = 1
        while tmp < size:
            tmp <<= 1

        self.size = tmp
        treeSize = self.size * 2
        self.tree = [IntervalTreePlusSum.TreeNode() for i in range(treeSize)]

    def update_sub(self, nb, length):
        if nb >= self.size:
            self.tree[nb].SUB = self.tree[nb].LOAD
        else:
            sumOfChilds = self.tree[bto.left_son(nb)].SUB 
            sumOfChilds += self.tree[bto.right_son(nb)].SUB
            self.tree[nb].SUB = sumOfChilds + self.tree[nb].LOAD * length

    def insert(self, value, begin_pos, end_pos):
        assert(begin_pos >= 0)
        assert(end_pos < self.size)
        assert(begin_pos <= end_pos)

        begin_pos += self.size
        end_pos += self.size

        length = 1
        self.tree[begin_pos].LOAD += value
        self.tree[end_pos].LOAD += value if begin_pos != end_pos else 0

        while bto.is_in_tree(begin_pos):
            if bto.parent(begin_pos) != bto.parent(end_pos):
                    
                    if bto.is_left_son(begin_pos):
                        self.tree[bto.right_sibling(begin_pos)].LOAD += value
                        self.update_sub(bto.right_sibling(begin_pos), length)

                    if bto.is_right_son(end_pos):
                        self.tree[bto.left_sibling(end_pos)].LOAD += value
                        self.update_sub(bto.left_sibling(end_pos), length)

            self.update_sub(begin_pos, length)
            self.update_sub(end_pos, length)

            length <<= 1
            begin_pos = bto.parent(begin_pos)
            end_pos = bto.parent(end_pos)

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

        while bto.is_in_tree(begin_pos):
            result += left_path * self.tree[begin_pos].LOAD
            result += right_path * self.tree[end_pos].LOAD

            if bto.parent(begin_pos) != bto.parent(end_pos):

                if bto.is_left_son(begin_pos):
                    result += self.tree[bto.right_sibling(begin_pos)].SUB
                    left_path += length

                if bto.is_right_son(end_pos):
                    result += self.tree[bto.left_sibling(end_pos)].SUB
                    right_path += length

            begin_pos = bto.parent(begin_pos)
            end_pos = bto.parent(end_pos)
            length <<= 1

        return result
