import random
# import heapq
import numpy as np
import torch as th


# https://github.com/rlcode/per/blob/master/SumTree.py
# https://github.com/rlcode/per/blob/master/prioritized_memory.py


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    # SumTree를 초기화하는 함수 
    # capacity로 트리 노드의 최대 크기를 입력 받는다. 
    # 또한 초깃값에 따라 트리와 데이터 배열을 생성
    def __init__(self, capacity): 
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        # self.heap = []
        # heapq.heapify(self.heap)

    # update to the root node
    # 현재 인덱스(idx)에서 변화(change)를 적용하여 부모 노드를 업데이트한다.
    # 이 업데이트는 루트 노드까지 이어진다.
    def _propagate(self, idx, change): 
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    # 특정 가중치를 기준으로 잎 노드에서 자료를 찾는다. 
    # 여기서 s는 랜덤 가중치로 설정
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # 트리의 루트 노드 값을 반환한다.
    def total(self):
        return self.tree[0]

    # store priority and sample
    # 데이터를 저장하고 우선순위(p) 및 데이터를 저장한다. 
    # 거의 제거된 우선순위를 검색하기 위해 힙(heap)이 사용되며, 입력된 우선순위(p)와 함께 데이터를 저장하고 업데이트한다.
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority 
    # 입력된 인덱스(idx)와 우선순위(p)를 기반으로 SumTree를 업데이트한다.
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    # 검색된 값을 사용하여 인덱스(idx), 우선순위, 데이터를 반환한다.
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
