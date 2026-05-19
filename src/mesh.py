import numpy as np

class Mesh:
    def __init__(self, a=0.0, b=1.0, N0=4):
        self.nodes = np.linspace(a, b, N0 + 1) 
        self.update_elements()

    def update_elements(self):
        N = len(self.nodes) - 1
        self.h = np.diff(self.nodes)
        self.centers = (self.nodes[:-1] + self.nodes[1:]) / 2
        self.N = N

    def refine_element(self, elem_idx):
        if elem_idx < 0 or elem_idx >= self.N:
            raise ValueError(f"Неправильний індекс елемента: {elem_idx}")
        new_node = self.centers[elem_idx]
        insert_pos = elem_idx + 1
        self.nodes = np.insert(self.nodes, insert_pos, new_node)
        self.update_elements()

    def refine_elements(self, elem_indices):
        for idx in sorted(elem_indices, reverse=True):
            self.refine_element(idx)

    def get_element_nodes(self, elem_idx):
        return self.nodes[elem_idx], self.nodes[elem_idx + 1]