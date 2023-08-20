import numpy
import time


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        # 우선 순위를 저장
        self.tree = numpy.zeros(2 * capacity - 1)
        # transition을 저장
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # 우선 순위가 변경될 때 그 변경을 트리의 상위 노드로 전파 (하위 노드의 우선 순위 변경 시 그 변경을 반영하여 상위 노드 업데이트)
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # 주어진 값 s에 해당하는 리프 노드를 찾는 메서드 (이를 통해 우선 순위에 비례하여 transition 샘플링 가능)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # 모둔 transition의 우선 순위 합 return
    def total(self):
        return self.tree[0]

    # p, data는 각각 우선 순위와 transition
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # 주어진 idx의 우선 순위를 p로 업데이트
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # 주어진 값 s에 해당하는 transition을 가져오는 메서드
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        if isinstance(self.data[dataIdx], int):
            print(f"idx : {idx}")
            print(f"dataIdx : {dataIdx}")
            print(f"s : {s}")
            print(f"self.tree[idx] : {self.tree[idx]}")
            print(f"self.data[dataIdx] : {self.data[dataIdx]}")
            time.sleep(10000)
        """
            ==== tree ====
            idx : 1114110
            self.tree[idx] : 0.0
            self.data[dataIdx] : 0
            ============
            원인 : 각 list는 numpy.zeros로 인해 0으로 초기화된 상태이므로 추후 다른 값으로 초기화되지 않은 idx에 접근해서 0 출력하는거 
        """

        return (idx, self.tree[idx], self.data[dataIdx])
