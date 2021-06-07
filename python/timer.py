import time
import pandas as pd
import numpy as np


class Timer(object):
    def __init__(self, name="the", ordered=False):
        self.name = name
        self.time_dict = {}
        self.last_checkpoint = time.time()
        self.ordered = ordered
        self.naction = 1

    def update(self):
        import time
        self.last_checkpoint = time.time()

    def add_checkpoint(self, comment):
        now = time.time()
        dt = now - self.last_checkpoint
        if self.ordered:
            comment = f'{self.naction} {comment}'
            self.naction += 1
        if comment in self.time_dict:
            self.time_dict[comment] += dt
        else:
            self.time_dict[comment] = dt
        self.last_checkpoint = now

    def summary(self):
        columns = ["Action", "Time (s)", "Time (%)"]
        action_groups = []
        muons = [
            "Filled muon variables",
            "Mu1 and Mu2",
            "Selected events and muons"
        ]
        jets = [
            "Applied JEC/JER, if enabled"
            "Filled jet variables",
            "Completed jet loop",
            "Prepared jets",
            "Selected jets"
        ]
        corrections = [
            "Applied QGL and B-tag weights",
            "Computed event weights",
            "FSR recovery",
            "GeoFit correction"
        ]
        other = [
            "Applied HLT and lumimask",
            "Calculated SA variables",
            "Filled outputs",
            "Initialization"
        ]

        for action in list(self.time_dict.keys()):
            action_groups.append(action)
            """
            if action in muons:
                action_groups.append("Muons")
            elif action in jets:
                action_groups.append("Jets")
            elif action in corrections:
                action_groups.append("Corrections")
            elif action in other:
                action_groups.append("Other")
            else:
                action_groups.append(action)
            """
        summary = pd.DataFrame(columns=columns)
        total_time = round(sum(list(self.time_dict.values())), 5)
        summary[columns[0]] = np.array(action_groups)
        summary[columns[1]] = np.round(
            np.array(list(self.time_dict.values())), 2)
        summary[columns[2]] = np.round(
            100 * np.array(list(self.time_dict.values())) / total_time, 3)
        print('-' * 50)
        print(f'Summary of {self.name} timer:')
        print('-' * 50)
        print(summary.groupby(columns[0]).sum())
        print('-' * 50)
        print(f'Total time: {total_time} s')
        print('=' * 50)
        print()
