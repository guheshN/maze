# -------------------------------------------------
# File for Tasks A and B
# Class for knapsack
# PLEASE UPDATE THIS FILE
#
# __author__ = 'Edward Small'
# __copyright__ = 'Copyright 2025, RMIT University'
# -------------------------------------------------

import csv
from maze.maze import Maze


class Knapsack:
    """
    Base class for the knapsack.
    """

    def __init__(self, capacity: int, knapsackSolver: str):
        """
        Constructor.

        @param capacity: the maximum weight the knapsack can hold
        @param knapsackSolver: the method we wish to use to find optimal knapsack items (recur or dynamic)
        """
        # initialise variables
        self.capacity = capacity
        self.optimalValue = 0
        self.optimalWeight = 0
        self.optimalCells = []
        self.knapsackSolver = knapsackSolver

    def solveKnapsack(self, maze: Maze, filename: str):
        """
        Calls the method to calculate the optimal knapsack solution
        @param maze: The maze we are considering
        """
        map = []
        # Sort by row (i) first, then column (j)
        sorted_items = sorted(maze.m_items.items(), key=lambda item: (item[0][0], item[0][1]))

        for cell, (weight, value) in sorted_items:
            map.append([cell, weight, value])

        if self.knapsackSolver == "recur":
            self.optimalCells, self.optimalWeight, self.optimalValue = self.recursiveKnapsack(map,
                                                                                              self.capacity,
                                                                                              len(map),
                                                                                              filename)
        elif self.knapsackSolver == "dynamic":
            self.optimalCells, self.optimalWeight, self.optimalValue = self.dynamicKnapsack(map,
                                                                                            self.capacity,
                                                                                            len(map),
                                                                                            filename)

        else:
            raise Exception("Incorrect Knapsack Solver Used.")

    def recursiveKnapsack(self, items: list, capacity: int, num_items: int, filename: str = None,
                          stats={'count': 0, 'logged': False}):
        """
        Recursive 0/1 Knapsack that logs how many times it's been called
        when the base case is first hit.

        @param items: list of (name, weight, value)
        @param capacity: current remaining knapsack capacity
        @param num_items: number of items still being considered
        @param filename: where to save call count on first base case (used for testing)
        @param stats: dict tracking call count and log status (used for testing)
        """
        # Increment count on every call 
        stats['count'] += 1

        # delete the below 3 lines if function implemented
        # with open(filename + '.txt', "w") as f:
        #     f.write(str(stats['count']))
        # stats['logged'] = True

        # Base case
        if capacity == 0 or num_items == 0:
            if not stats['logged'] and filename:
                with open(filename+'.txt', "w") as f:
                    f.write(str(stats['count']))
                stats['logged'] = True  # Make sure we only log once
            return [], 0, 0

        # Get the k-th item 
        item = items[num_items - 1]
        location, weight, value = item[0], item[1], item[2]

        # If the item's weight is more than the remaining capacity, skip it
        if weight > capacity:
            return self.recursiveKnapsack(items, capacity, num_items - 1, filename, stats)
        else:
            # Include the item
            inc_locs, inc_weight, inc_value = self.recursiveKnapsack(items, capacity - weight, num_items - 1, filename, stats)
            inc_locs = inc_locs + [location]
            inc_weight = inc_weight + weight
            inc_value = inc_value + value

            # Exclude the item
            exc_locs, exc_weight, exc_value = self.recursiveKnapsack(items, capacity, num_items - 1, filename, stats)

            # Choose the better option
            if inc_value > exc_value:
                return inc_locs, inc_weight, inc_value
            else:
                return exc_locs, exc_weight, exc_value

    def dynamicKnapsack(self, items: list, capacity: int, num_items: int, filename: str):
        """
        Dynamic 0/1 Knapsack that saves the dynamic programming table as a csv.

        @param items: list of (name, weight, value)
        @param capacity: current remaining knapsack capacity
        @param num_items: number of items still being considered
        @param filename: save name for csv of table (used for testing)

        """

        # Initialize DP table with None
        dp = [[None] * (capacity + 1) for _ in range(num_items + 1)]

        # first row is all 0s
        dp[0] = [0] * (capacity + 1)


        """
        IMPLEMENT ME FOR TASK B
        """
        def MFKnapsack(i :int,j:int):
            # item = items[num_items - 1]
            # # _, weight, value = item[0], item[1], item[2]
            # # Base case
            if i == 0 or j == 0:
                dp[i][j] = 0  
                return 0
            
            
            if dp[i][j] is not None:
                return dp[i][j]
            
            location, weight, value = items[i-1][0], items[i-1][1], items[i-1][2]
            
            if weight > j:
                x=MFKnapsack(i-1,j)
                
            else:
                x=max(MFKnapsack(i-1,j),(MFKnapsack(i-1,j-weight)+value))

            dp[i][j] = x

            return dp[i][j]


        max_value = MFKnapsack(num_items, capacity)
        selected_weight = 0
        selected_items = []
        i,j = num_items,capacity
            
        while i>0 and j>0:
            item = items[i-1]
            location, weight, value = item[0], item[1], item[2]
                    
            if dp[i][j] != dp[i-1][j]:
                selected_items.append(location)
                selected_weight += weight
                j -= weight
            i -= 1

                    # # === Save DP Table to CSV ===
                    
        self.saveCSV(dp, items, capacity, filename)
            
        return selected_items, selected_weight, max_value


    def saveCSV(self, dp: list, items: list, capacity: int, filename: str):
        with open(filename+".csv", 'w', newline='') as f:
            writer = csv.writer(f)

            # Header: capacities from 0 to capacity
            header = [''] + [str(j) for j in range(capacity + 1)]
            writer.writerow(header)

            # First row: dp[0], meaning "no items considered"
            first_row = [''] + [(val if val is not None else '#') for val in dp[0]]
            writer.writerow(first_row)

            # Following rows: each item
            for i in range(1, len(dp)):
                row_label = f"({items[i - 1][1]}, {items[i - 1][2]})"
                row = [row_label] + [(val if val is not None else '#') for val in dp[i]]
                writer.writerow(row)

