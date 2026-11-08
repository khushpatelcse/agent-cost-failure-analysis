# failure analysis

class FailureAnalyzer:
    """Tracks how many tool failures occur during an agent run."""

    def __init__(self):
        self.failures = 0

    def reset(self):
        """Reset the failure count for a new agent run."""
        self.failures = 0

    def record_failure(self):
        """Record one tool failure."""
        self.failures += 1

    def get_failure_count(self):
        """Return the total number of failures."""
        return self.failures