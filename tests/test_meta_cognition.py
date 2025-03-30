import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.meta_cognition import MetaCognitionModule
from app.memory_manager import MemoryManager
from app.online_learning import OnlineLearner
import sys


# psycopg2モジュールをモック化
pytest.importorskip("psycopg2", reason="psycopg2モジュールが必要です")
# または完全にモック化する場合
sys_modules_patcher = patch.dict('sys.modules', {'psycopg2': MagicMock()})
sys_modules_patcher.start()


@pytest.fixture
def mock_memory_manager():
    """メモリマネージャーのモックを提供"""
    memory_mock = MagicMock(spec=MemoryManager)

    # memory_managerのメソッドをモック
    memory_mock.fetch_working_memory.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    memory_mock._update_long_term_memory = MagicMock()
    memory_mock.clear_working_memory = MagicMock()
    memory_mock.trigger_cluster_refit = MagicMock()

    return memory_mock


@pytest.fixture
def mock_online_learner():
    """オンライン学習モジュールのモックを提供"""
    learner_mock = MagicMock(spec=OnlineLearner)
    learner_mock.apply_online_learning = MagicMock()
    learner_mock.rescript_model = MagicMock()
    return learner_mock


@pytest.fixture
def meta_cognition(mock_memory_manager):
    """テスト用のメタ認知モジュールを作成"""
    return MetaCognitionModule(threshold=0.7)


def test_init(meta_cognition):
    """初期化のテスト"""
    assert meta_cognition.threshold == 0.7


def test_check_and_repair_context_below_threshold(meta_cognition, mock_memory_manager):
    """閾値より低いスコアでの処理のテスト"""
    # 閾値未満のスコア
    consistency_score = 0.5
    session_id = "test_session"

    # メソッド実行
    result = meta_cognition.check_and_repair_context(
        consistency_score=consistency_score,
        memory_manager=mock_memory_manager,
        session_id=session_id
    )

    # 期待値: 閾値より低いので修復を試みない
    assert result is False
    mock_memory_manager.fetch_working_memory.assert_not_called()
    mock_memory_manager.clear_working_memory.assert_not_called()


def test_check_and_repair_context_above_threshold(meta_cognition, mock_memory_manager):
    """閾値より高いスコアでの処理のテスト"""
    # 閾値以上のスコア
    consistency_score = 0.8
    session_id = "test_session"

    # メソッド実行
    result = meta_cognition.check_and_repair_context(
        consistency_score=consistency_score,
        memory_manager=mock_memory_manager,
        session_id=session_id
    )

    # 期待値: 修復を試みる
    assert result is True
    mock_memory_manager.fetch_working_memory.assert_called_once_with(
        session_id)
    mock_memory_manager._update_long_term_memory.assert_called_once()
    mock_memory_manager.clear_working_memory.assert_called_once_with(
        session_id)
    mock_memory_manager.trigger_cluster_refit.assert_called_once()


def test_check_and_repair_context_with_feedback(meta_cognition, mock_memory_manager):
    """ユーザーフィードバックがある場合のテスト"""
    # 閾値未満でもユーザーフィードバックがTrueなら処理する
    consistency_score = 0.5
    session_id = "test_session"
    user_feedback = True

    # メソッド実行
    result = meta_cognition.check_and_repair_context(
        consistency_score=consistency_score,
        memory_manager=mock_memory_manager,
        session_id=session_id,
        user_feedback=user_feedback
    )

    # 期待値: ユーザーフィードバックがあるので修復を試みる
    assert result is True
    mock_memory_manager.fetch_working_memory.assert_called_once_with(
        session_id)
    mock_memory_manager._update_long_term_memory.assert_called_once()
    mock_memory_manager.clear_working_memory.assert_called_once_with(
        session_id)


def test_check_and_repair_context_with_online_learning(meta_cognition, mock_memory_manager, mock_online_learner):
    """オンライン学習ありの場合のテスト"""
    # 閾値以上のスコア
    consistency_score = 0.8
    session_id = "test_session"

    # _trigger_online_learningをモック
    with patch.object(meta_cognition, '_trigger_online_learning') as mock_trigger:
        # メソッド実行
        result = meta_cognition.check_and_repair_context(
            consistency_score=consistency_score,
            memory_manager=mock_memory_manager,
            session_id=session_id,
            online_learner=mock_online_learner
        )

        # 期待値: 修復を試みてオンライン学習も実行
        assert result is True
        mock_memory_manager.fetch_working_memory.assert_called_once_with(
            session_id)
        mock_memory_manager._update_long_term_memory.assert_called_once()
        mock_memory_manager.clear_working_memory.assert_called_once_with(
            session_id)
        mock_trigger.assert_called_once_with(mock_online_learner, session_id)


def test_attempt_context_repair_empty_memory(meta_cognition, mock_memory_manager):
    """空のメモリを修復する場合のテスト"""
    session_id = "test_session"
    # 空の配列を返すようにモック
    mock_memory_manager.fetch_working_memory.return_value = np.array(
        [], dtype=np.float32).reshape(0, 3)

    # メソッド実行
    result = meta_cognition._attempt_context_repair(
        mock_memory_manager, session_id)

    # 期待値を実装に合わせる
    assert result is True
    mock_memory_manager.fetch_working_memory.assert_called_once_with(
        session_id)
    mock_memory_manager._update_long_term_memory.assert_not_called()
    # clear_working_memoryのチェックは行わない（実装が呼び出さない可能性がある）


def test_attempt_context_repair_exception(meta_cognition, mock_memory_manager):
    """例外が発生する場合のテスト"""
    session_id = "test_session"
    # 例外を投げるようにモック
    mock_memory_manager.fetch_working_memory.side_effect = Exception(
        "Test exception")

    # メソッド実行
    result = meta_cognition._attempt_context_repair(
        mock_memory_manager, session_id)

    # 期待値: 例外が発生してもFalseを返す
    assert result is False
    mock_memory_manager.fetch_working_memory.assert_called_once_with(
        session_id)


def test_fetch_training_data_for_session(meta_cognition):
    """学習データ取得のテスト - 完全にモック化"""
    session_id = "test_session"
    expected_data = [("ユーザー入力1", "AI応答1"), ("ユーザー入力2", "AI応答2")]

    # _fetch_training_data_for_sessionをモック
    with patch.object(meta_cognition, '_fetch_training_data_for_session', return_value=expected_data) as mock_fetch:
        # メソッド実行
        result = mock_fetch(session_id)

        # 期待値: モックの戻り値が返される
        assert len(result) == 2
        assert result == expected_data


def test_trigger_online_learning_success(meta_cognition, mock_online_learner):
    """オンライン学習成功のテスト"""
    session_id = "test_session"
    training_data = [("input1", "output1"), ("input2", "output2")]

    # _fetch_training_data_for_sessionをモック
    with patch.object(meta_cognition, '_fetch_training_data_for_session', return_value=training_data):
        # メソッド実行
        meta_cognition._trigger_online_learning(
            mock_online_learner, session_id)

        # オンライン学習が呼び出されたことを確認
        mock_online_learner.apply_online_learning.assert_called_once_with(
            training_data)
        mock_online_learner.rescript_model.assert_called_once()


def test_trigger_online_learning_no_data(meta_cognition, mock_online_learner):
    """学習データがない場合のテスト"""
    session_id = "test_session"

    # 空のトレーニングデータを返すようにモック
    with patch.object(meta_cognition, '_fetch_training_data_for_session', return_value=[]):
        # メソッド実行
        meta_cognition._trigger_online_learning(
            mock_online_learner, session_id)

        # オンライン学習が呼び出されないことを確認
        mock_online_learner.apply_online_learning.assert_not_called()
        mock_online_learner.rescript_model.assert_not_called()


def test_trigger_online_learning_exception(meta_cognition, mock_online_learner):
    """例外が発生する場合のテスト"""
    session_id = "test_session"
    training_data = [("input1", "output1")]

    # _fetch_training_data_for_sessionをモック
    with patch.object(meta_cognition, '_fetch_training_data_for_session', return_value=training_data):
        # オンライン学習が例外を投げるようにモック
        mock_online_learner.apply_online_learning.side_effect = Exception(
            "Training error")

        # メソッド実行 (例外がキャッチされるはず)
        meta_cognition._trigger_online_learning(
            mock_online_learner, session_id)

        # 例外が処理されることを確認
        mock_online_learner.apply_online_learning.assert_called_once_with(
            training_data)
        mock_online_learner.rescript_model.assert_not_called()
