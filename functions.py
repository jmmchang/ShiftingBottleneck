from collections import defaultdict, deque
from ortools.sat.python import cp_model
import math

class DAG:
    def __init__(self, num_jobs, num_machines):
        self.n = num_jobs
        self.m = num_machines
        self.makespan = 0
        self.neighbors = defaultdict(list)
        self.disjunctive_edges = defaultdict(list)

    @staticmethod
    def solve_subproblem(p, r, d):
        n = len(p)
        mapping = {}
        for i, (x, y) in enumerate(p):
            mapping[i] = (x, y)

        horizon = sum(list(p.values())) + max(list(r.values()))
        model = cp_model.CpModel()
        start = {}
        end = {}
        interval = {}

        for i in range(n):
            start[i] = model.new_int_var(0, horizon, f'start_{i}')
            end[i] = model.new_int_var(0, horizon, f'end_{i}')
            interval[i] = model.new_interval_var(start[i], p[mapping[i]], end[i], f'interval_{i}')

        for i in range(n):
            model.add(start[i] >= r[mapping[i]])

        model.add_no_overlap([interval[i] for i in range(n)])
        lateness = model.new_int_var(0, horizon, 'lateness')
        model.add_max_equality(lateness, [end[i] - d[mapping[i]] for i in range(n)])
        model.minimize(lateness)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60
        solver.Solve(model)
        order = sorted(range(n), key = lambda x: solver.Value(start[x]))

        return solver.Value(lateness), [mapping[o] for o in order]

    def longest_distance(self, u, v):
        dist = defaultdict(lambda: -math.inf)
        dist[u] = 0
        queue = deque([u])

        while queue:
            node = queue.popleft()
            for neighbor, weight in self.neighbors[node]:
                if dist[neighbor] < dist[node] + weight:
                    dist[neighbor] = dist[node] + weight
                    queue.append(neighbor)

        return dist[v]

    def insert_edges(self, edges, processing_times):
        for i in range(len(edges) - 1):
            self.neighbors[edges[i]].append((edges[i + 1], processing_times[edges[i]]))

    def delete_edges(self, machine, processing_times):
        for i in range(len(self.disjunctive_edges[machine]) - 1):
            self.neighbors[self.disjunctive_edges[machine][i]].remove((self.disjunctive_edges[machine][i+1], processing_times[self.disjunctive_edges[machine][i]]))

    def build_graph(self, edges):
        for (u,v,w) in edges:
            self.neighbors[u].append((v, w))

        self.makespan = self.longest_distance("source", "sink")

    def prepare_data(self, machine, processing_times):
        jobs = [u for u in self.neighbors.keys() if u[0] == machine]
        release_date = defaultdict(float)
        due_date = defaultdict(float)

        for i, j in jobs:
            release_date[(i, j)] = self.longest_distance("source", (i, j))
            due_date[(i, j)] = self.makespan - self.longest_distance((i, j), "sink") + processing_times[(i, j)]

        return jobs, release_date, due_date

    def backtracking(self, machine_scheduled, processing_times):
        for m in machine_scheduled:
            self.delete_edges(m, processing_times)
            jobs, release_date, due_date = self.prepare_data(m, processing_times)
            filtered = {k: v for k, v in processing_times.items() if k in jobs}
            lateness, solution = self.solve_subproblem(filtered, release_date, due_date)
            self.disjunctive_edges[m] = solution
            self.insert_edges(solution, processing_times)

    def run(self, edges, processing_times):
        self.build_graph(edges)
        machine_not_scheduled = set(range(1, self.m + 1))
        machine_scheduled = set()
        solution = []

        while machine_not_scheduled:
            lateness = -math.inf
            machine_to_be_scheduled = -1
            for i in machine_not_scheduled:
                jobs, release_date, due_date = self.prepare_data(i, processing_times)
                filtered = {k: v for k, v in processing_times.items() if k in jobs}
                curr_lateness, curr_solution = self.solve_subproblem(filtered, release_date, due_date)
                if curr_lateness > lateness:
                    machine_to_be_scheduled = i
                    lateness = curr_lateness
                    solution = curr_solution

            self.disjunctive_edges[machine_to_be_scheduled] = solution
            self.insert_edges(solution, processing_times)
            self.backtracking(machine_scheduled, processing_times)
            machine_scheduled.add(machine_to_be_scheduled)
            machine_not_scheduled.remove(machine_to_be_scheduled)
            self.makespan = self.longest_distance("source", "sink")

        return self.makespan, self.disjunctive_edges