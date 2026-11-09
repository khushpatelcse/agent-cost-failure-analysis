# cost analysis

import time


class CostAnalyzer:

    def __init__(self):
        self.start_time = None

    def start_timer(self):
        """Start measuring how long the agent takes."""
        self.start_time = time.perf_counter()

    def count_tool_calls(self, tool_calls):
        """Count how many tools the model requested."""
        return len(tool_calls)

    def get_elapsed_seconds(self):
        """Calculate the total time taken by the agent."""
        if self.start_time is None:
            return 0.0

        elapsed_time = time.perf_counter() - self.start_time

        return round(elapsed_time, 4)