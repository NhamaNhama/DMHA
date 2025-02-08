import numpy as np
from typing import Optional
from .memory_manager import MemoryManager
from .online_learning import OnlineLearner

class MetaCognitionModule:
    """
    整合性スコア/フィードバックから矛盾を検知し、自己修正＆(任意)オンライン学習まで行う。
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check_and_repair_context(self,
                                 consistency_score: float,
                                 memory_manager: MemoryManager,
                                 session_id: str,
                                 user_feedback: Optional[bool] = None,
                                 online_learner: Optional[OnlineLearner] = None) -> bool:
        if (consistency_score > self.threshold) or (user_feedback is True):
            success = self._attempt_context_repair(memory_manager, session_id)
            if success and online_learner is not None:
                # オンライン学習を実行 → 再TorchScript化
                self._trigger_online_learning(online_learner, session_id)
            return success
        return False

    def _attempt_context_repair(self, memory_manager: MemoryManager, session_id: str) -> bool:
        try:
            old_context = memory_manager.fetch_working_memory(session_id)
            if old_context is not None and len(old_context) > 0:
                summary_embedding = np.mean(old_context, axis=0)
                memory_manager._update_long_term_memory(session_id, np.array([summary_embedding]))
                memory_manager.clear_working_memory(session_id)
                memory_manager.trigger_cluster_refit()
            return True
        except Exception:
            return False

    def _trigger_online_learning(self, online_learner: OnlineLearner, session_id: str):
        # 簡易的サンプル
        training_samples = [
            ("User: Hello.\nAI: Contradiction happened before.",
             "AI: Correction: I've updated my knowledge to be consistent now.")
        ]
        online_learner.apply_online_learning(training_samples)
        online_learner.rescript_model()