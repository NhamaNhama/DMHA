import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.memory_manager import MemoryManager
import json


@pytest.fixture
def mock_redis():
    """Redisのモックを提供"""
    redis_mock = MagicMock()
    redis_mock.pipeline.return_value.__enter__.return_value = redis_mock
    return redis_mock


@pytest.fixture
def mock_logger():
    """Loggerのモックを提供"""
    return MagicMock()


@pytest.fixture
def memory_manager(mock_redis, mock_logger):
    """テスト用メモリマネージャーを作成"""
    with patch('pymilvus.connections.connect'):
        # テストモードでMilvusをスキップする設定
        return MemoryManager(
            redis=mock_redis,
            logger=mock_logger,
            test_mode=True
        )


def test_init(memory_manager, mock_logger):
    """初期化のテスト"""
    assert memory_manager.redis is not None
    assert memory_manager.logger is mock_logger
    assert memory_manager.test_mode is True
    assert memory_manager.milvus_connection_active is False
    assert memory_manager.faiss_index is not None
    mock_logger.info.assert_called_with("Milvus接続をスキップします")


def test_hierarchical_caching(memory_manager, mock_redis):
    """階層的キャッシュのテスト"""
    # テストデータ作成
    session_id = "test_session"
    embeddings = np.random.rand(10, 4096).astype(np.float32)

    # メソッド実行
    memory_manager.hierarchical_caching(session_id, embeddings)

    # Redisへの保存が呼ばれたことを確認
    assert mock_redis.multi.called
    assert mock_redis.lpush.call_count == 10
    assert mock_redis.ltrim.called
    assert mock_redis.expire.called
    assert mock_redis.execute.called


def test_fetch_working_memory(memory_manager, mock_redis):
    """ワーキングメモリ取得のテスト"""
    # テストデータ設定
    session_id = "test_session"
    test_vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    # モックの戻り値を設定
    mock_redis.lrange.return_value = [
        json.dumps(test_vectors[0]),
        json.dumps(test_vectors[1])
    ]

    # メソッド実行
    result = memory_manager.fetch_working_memory(session_id)

    # 期待値と一致することを確認
    assert mock_redis.lrange.called
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    np.testing.assert_almost_equal(result[0], test_vectors[0])
    np.testing.assert_almost_equal(result[1], test_vectors[1])


def test_clear_working_memory(memory_manager, mock_redis):
    """ワーキングメモリのクリアテスト"""
    session_id = "test_session"

    memory_manager.clear_working_memory(session_id)

    # Redisのdeleteが呼ばれたことを確認
    mock_redis.delete.assert_called_once_with(f"{session_id}:working_memory")


@pytest.mark.skip(reason="FAISSテストでセグメンテーションフォルトが発生するため一時的にスキップ")
def test_cluster_with_faiss(memory_manager):
    """FAISSクラスタリングのテスト"""
    # パッチを適用して_init_faiss_indexを上書き
    with patch.object(memory_manager, '_init_faiss_index') as mock_init:
        # クラスタ数を少なくしたFAISSインデックスを作成
        import faiss
        nlist = 5  # クラスタ数を少なく設定
        quantizer = faiss.IndexFlatL2(memory_manager.dimension)
        memory_manager.faiss_index = faiss.IndexIVFFlat(
            quantizer, memory_manager.dimension, nlist, faiss.METRIC_L2)

        # テストデータ作成
        embedding = np.random.rand(memory_manager.dimension).astype(np.float32)

        # 事前にインデックスにデータを追加
        # クラスタ数以上のデータポイントを用意
        train_data = np.random.rand(
            10, memory_manager.dimension).astype(np.float32)
        memory_manager.faiss_index.train(train_data)
        memory_manager.faiss_index.add(train_data)

        # クラスタリングを実行
        cluster_id = memory_manager._cluster_with_faiss(embedding)

        # クラスタIDが返されることを確認
        assert isinstance(cluster_id, int)


def test_empty_embeddings(memory_manager):
    """空の埋め込みデータの処理テスト"""
    # 空のnumpy配列
    empty_embeddings = np.array([], dtype=np.float32).reshape(0, 4096)

    # 例外が発生しないことを確認
    memory_manager.hierarchical_caching("test_session", empty_embeddings)

    # Noneの場合も例外が発生しないことを確認
    memory_manager.hierarchical_caching("test_session", None)


def test_extract_entities(memory_manager, mock_redis):
    """エンティティ抽出のテスト"""
    session_id = "test_session"
    test_entities = {"person": ["Alice", "Bob"], "location": ["Tokyo"]}

    # モックの戻り値を設定
    mock_redis.get.return_value = json.dumps(test_entities)

    # メソッド実行
    result = memory_manager._extract_entities(session_id)

    # 期待値と一致することを確認
    mock_redis.get.assert_called_once_with(f"{session_id}:entities")
    assert result == test_entities
