import json
import time
import numpy as np
import logging
from typing import Optional
from redis import Redis
from pymilvus import connections, Collection
import faiss
import hdbscan
import os


class MemoryManager:
    """
    Redis (短期メモリ) + Milvus (長期メモリ) + Faiss/HDBSCANクラスタリング
    """

    def __init__(self, redis: Redis, logger: logging.Logger, cluster_method: str = "faiss",
                 milvus_host: str = "localhost", test_mode: bool = False):
        self.redis = redis
        self.logger = logger
        self.cache_ttl = 3600
        self.cluster_method = cluster_method
        self.dimension = 4096
        self.test_mode = test_mode

        # Milvus接続
        self.milvus_connection_active = False
        if not test_mode and os.environ.get("SKIP_MILVUS", "false").lower() != "true":
            try:
                # pymilvus の接続設定例
                connections.connect(
                    alias="default",
                    host=milvus_host,
                    port="19530"
                )
                self.milvus_connection_active = True
                logger.info(f"Milvusに接続しました: {milvus_host}")
            except Exception as e:
                logger.warning(f"Milvus接続エラー: {str(e)}")
        else:
            logger.info("Milvus接続をスキップします")

        # クラスタリング用のインデックス
        self.faiss_index = None
        self.cluster_model = None
        self.hdbscan_embeddings_key = "hdbscan_temp_embeddings"

        # Milvusコレクション初期化
        self.long_term_memory_collection = None
        if self.milvus_connection_active:
            try:
                self.long_term_memory_collection = Collection(
                    "long_term_memory", using="default")
                logger.info("Milvusコレクションを初期化しました")
            except Exception as e:
                logger.warning(f"Milvusコレクション取得エラー: {str(e)}")

        # Faissインデックス初期化
        if self.cluster_method == "faiss":
            self._init_faiss_index()

    def _init_faiss_index(self):
        nlist = 100
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.faiss_index = faiss.IndexIVFFlat(
            quantizer, self.dimension, nlist, faiss.METRIC_L2)

    def hierarchical_caching(self, session_id: str, embeddings: np.ndarray):
        if embeddings is None or embeddings.shape[0] == 0:
            return
        try:
            with self.redis.pipeline() as pipe:
                pipe.multi()
                for emb in embeddings:
                    pipe.lpush(f"{session_id}:working_memory",
                               json.dumps(emb.tolist()))
                pipe.ltrim(f"{session_id}:working_memory", 0, 511)
                pipe.expire(f"{session_id}:working_memory", self.cache_ttl)
                pipe.execute()

            if embeddings.shape[0] > 512:
                self._update_long_term_memory(session_id, embeddings[512:])
        except Exception as e:
            self.logger.error(f"Error in hierarchical_caching: {str(e)}")

    def fetch_working_memory(self, session_id: str) -> Optional[np.ndarray]:
        try:
            cached = self.redis.lrange(f"{session_id}:working_memory", 0, -1)
            if cached:
                vectors = [json.loads(v) for v in cached]
                return np.array(vectors, dtype=np.float32)
            return None
        except Exception as e:
            self.logger.error(f"Error in fetch_working_memory: {str(e)}")
            return None

    def clear_working_memory(self, session_id: str):
        try:
            self.redis.delete(f"{session_id}:working_memory")
        except Exception as e:
            self.logger.error(f"Error in clear_working_memory: {str(e)}")

    def _update_long_term_memory(self, session_id: str, embeddings: np.ndarray):
        """長期記憶をベクトルデータベースに保存"""
        try:
            entities = self._extract_entities(session_id)
            records = []
            for emb in embeddings:
                cluster_id = self._cluster_embeddings(emb)
                record = {
                    "id": int(time.time() * 1000),
                    "embedding": emb.tolist(),
                    "timestamp": int(time.time()),
                    "entity_tags": entities,
                    "semantic_cluster": cluster_id
                }
                records.append(record)

            # Milvusコレクションが有効な場合のみ保存
            if self.milvus_connection_active and self.long_term_memory_collection:
                self.long_term_memory_collection.insert(records)
                self.long_term_memory_collection.flush()
                self.logger.info(f"{len(records)}件のベクトルを長期記憶に保存しました")
            else:
                self.logger.info("Milvus接続がないため、長期記憶への保存をスキップします")

        except Exception as e:
            self.logger.error(f"Error in _update_long_term_memory: {str(e)}")

    def _cluster_embeddings(self, embedding: np.ndarray) -> int:
        if self.cluster_method == "faiss":
            return self._cluster_with_faiss(embedding)
        elif self.cluster_method == "hdbscan":
            return self._cluster_with_hdbscan(embedding)
        else:
            return int(np.argmax(embedding[:10]))

    def _cluster_with_faiss(self, embedding: np.ndarray) -> int:
        if self.faiss_index is None:
            self._init_faiss_index()

        emb_f32 = embedding.reshape(1, -1).astype(np.float32)
        if not self.faiss_index.is_trained:
            train_data = np.repeat(emb_f32, 10, axis=0)
            self.faiss_index.train(train_data)

        if self.faiss_index.ntotal == 0:
            self.faiss_index.add(emb_f32)
            return 0

        D, I = self.faiss_index.search(emb_f32, 1)
        nearest_id = I[0][0]
        dist = D[0][0]
        return nearest_id

    def _cluster_with_hdbscan(self, embedding: np.ndarray) -> int:
        stored = self.redis.get(self.hdbscan_embeddings_key)
        if stored:
            arr_list = json.loads(stored)
        else:
            arr_list = []

        arr_list.append(embedding.tolist())
        self.redis.set(self.hdbscan_embeddings_key, json.dumps(arr_list))

        if len(arr_list) >= 50:
            data = np.array(arr_list, dtype=np.float32)
            self.cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=5, metric='euclidean')
            self.cluster_model.fit(data)
            labels = self.cluster_model.labels_
            return labels[-1]
        else:
            return -1

    def _extract_entities(self, session_id: str) -> dict:
        try:
            raw = self.redis.get(f"{session_id}:entities")
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def trigger_cluster_refit(self):
        if self.cluster_method == "faiss":
            pass
        elif self.cluster_method == "hdbscan":
            pass
