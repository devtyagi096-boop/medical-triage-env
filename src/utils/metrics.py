"""
Performance tracking utilities
"""

from typing import Dict, List, Any


class EpisodeMetrics:
    """Track and summarise episode performance"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rewards: List[float] = []
        self.correct: int = 0
        self.total: int = 0
        self.deteriorations: int = 0
        self.critical_waits: List[float] = []

    def record_step(self, reward: float, info: Dict[str, Any]):
        self.rewards.append(reward)
        self.total += 1
        if info.get('correct_triage'):
            self.correct += 1
        self.deteriorations += info.get('deteriorations', 0)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, len(self.rewards))

    def summary(self) -> Dict[str, Any]:
        return {
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
            'accuracy': self.accuracy,
            'steps': len(self.rewards),
            'deteriorations': self.deteriorations,
        }
