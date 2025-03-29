import numpy as np
from typing import Optional
from .memory_manager import MemoryManager
from .online_learning import OnlineLearner
import psycopg2  # 例: PostgreSQL 接続用ライブラリ

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
        """
        - 外部DBやファイルシステムからトレーニングデータを動的に取得
        - ロギング＆例外処理
        - 学習完了後にTorchScriptなどへの再変換
        """
        try:
            training_samples = self._fetch_training_data_for_session(session_id)
            if not training_samples:
                print(f"[MetaCognition] No training data found for session: {session_id}")
                return
            online_learner.apply_online_learning(training_samples)
            online_learner.rescript_model()
            print(f"[MetaCognition] Online learning completed for session: {session_id}")
        except Exception as e:
            print(f"[MetaCognition] Online learning failed for session {session_id}: {str(e)}")

    def _fetch_training_data_for_session(self, session_id: str):
        """
        DBやストレージから session_id に紐づく学習データを取得。
        必要に応じてフィルタや型変換を行い、[(input, target), ...] の形式で返す。
        """
        connection = None
        training_data = []
        try:
            # 実際の接続情報は設定ファイルや環境変数から読み込むなどしてください
            connection = psycopg2.connect(
                dbname="mydatabase",
                user="myuser",
                password="mypassword",
                host="db.example.com",
                port="5432"
            )
            cursor = connection.cursor()
            query = """
            SELECT user_input, ai_response
            FROM conversation_logs
            WHERE session_id = %s
            AND is_labeled = True
            """
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()

            for row in rows:  # row = (user_input, ai_response)
                user_input, ai_response = row
                training_data.append((user_input, ai_response))

            cursor.close()
        except (Exception, psycopg2.DatabaseError) as e:
            print(f"[MetaCognition] Failed to fetch training data from DB: {str(e)}")
        finally:
            if connection:
                connection.close()

        return training_data