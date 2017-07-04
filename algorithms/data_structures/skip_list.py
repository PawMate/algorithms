from random import random, randint, sample
class SkipListNode:
    def __init__(self, value = 0):
        self.value = value
        self.nexts = []

class SkipList:

    def __init__(self, MAX_LVL = 20):
        self.MAX_LVL = MAX_LVL

        self.head = SkipListNode(-1)
        self.head.nexts = [None] * self.MAX_LVL

    def get_random_lvl(self):
        lvl = 1
        while lvl < self.MAX_LVL and random() >= 0.5:
            lvl += 1
        return lvl

    def contain(self, value):

        item = self.head

        for act_lvl in range(self.MAX_LVL - 1, -1, -1):
            while item.nexts[act_lvl] != None and item.nexts[act_lvl].value < value:
                item = item.nexts[act_lvl]

            if item.nexts[act_lvl] != None and item.nexts[act_lvl].value == value:
                return True

        return False

    def add(self, value):
        if self.contain(value):
            return

        new_node = SkipListNode(value)
        random_lvl = self.get_random_lvl()
        new_node.nexts = [None] * random_lvl

        item = self.head

        for act_lvl in range(self.MAX_LVL - 1, -1, -1):

            while item.nexts[act_lvl] != None and item.nexts[act_lvl].value < value:
                item = item.nexts[act_lvl]

            if act_lvl < random_lvl:
                new_node.nexts[act_lvl] = item.nexts[act_lvl]
                item.nexts[act_lvl] = new_node

    def remove(self, value):
        if not self.contain(value):
            return

        item = self.head

        for act_lvl in range(self.MAX_LVL - 1, -1, -1):
            while item.nexts[act_lvl] != None and item.nexts[act_lvl].value < value:
                item = item.nexts[act_lvl]

            if item.nexts[act_lvl] != None and item.nexts[act_lvl].value == value:
                item.nexts[act_lvl] = item.nexts[act_lvl].nexts[act_lvl]
