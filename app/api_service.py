import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import jwt

from .context_system import ContextAwareSystem
from .memory_manager import MemoryManager


class APIService:
    class ContextRequest(BaseModel):
        session_id: str
        text: str
        metadata: Optional[Dict[str, Any]] = None
        feedback: Optional[bool] = None

    class ContextResponse(BaseModel):
        processed_text: str
        context_score: float
        entities: List[str]
        consistency_check: bool
        meta_cognition_repair: bool

    def __init__(self, system: ContextAwareSystem):
        self.app = FastAPI()
        self.system = system
        self.inference_engine = self.system.get_inference_engine()
        self.memory_manager = MemoryManager(
            redis=self.system.redis,
            milvus_host=self.system.config.milvus_host,
            logger=self.system.logger,
            cluster_method=self.system.config.cluster_method,
            test_mode=self.system.config.test_mode
        )
        self.jwt_secret = self.system.config.jwt_secret
        self._setup_routes()

    def _token_auth_scheme(self, request: Request) -> str:
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(
                status_code=401, detail="Missing Authorization header.")
        if not token.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Invalid token format.")
        return token.split(" ")[1]

    def _verify_jwt_token(self, token: str):
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired.")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return payload

    def _setup_routes(self):
        @self.app.post("/v1/contextualize", response_model=self.ContextResponse)
        async def process_request(
            request: self.ContextRequest,
            raw_token: str = Depends(self._token_auth_scheme)
        ):
            self._verify_jwt_token(raw_token)
            return await self._handle_request(request)

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.post("/auth/token")
        def issue_token(username: str):
            payload = {
                "sub": username,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600
            }
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            return {"access_token": token}

    async def _handle_request(self, request: ContextRequest) -> ContextResponse:
        start_time = datetime.now()
        try:
            self.system.metrics['inference_count'].inc()

            # 1) テキストをエンコード
            inputs = self._encode_text(request.text)
            # 2) 短期メモリに埋め込み
            self.memory_manager.hierarchical_caching(
                request.session_id, self._to_embedding(inputs))

            # 3) 推論
            inference_start = time.time()
            with torch.inference_mode():
                outputs = self.inference_engine(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    memory_context=self._retrieve_context(request.session_id)
                )
            inf_latency = time.time() - inference_start
            self.system.metrics['inference_latency'].observe(inf_latency)

            # 4) 整合性&シンボリックチェック
            consistency_score = outputs['consistency_score'].item()
            # symbolic_check: 実運用レベル。テキスト内に fact/deny/rule があればKBに格納して矛盾判定
            symbolic_valid = self.system.ns_interface.symbolic_check(
                request.text, scope=request.session_id)
            final_consistency = (
                consistency_score < self.system.config.consistency_threshold) and symbolic_valid
            self.system.metrics['conflict_score'].set(consistency_score)

            # 5) メタ認知制御
            meta_cognition_invoked = False
            if not final_consistency:
                meta_cognition_invoked = self.system.meta_cognition.check_and_repair_context(
                    consistency_score=consistency_score,
                    memory_manager=self.memory_manager,
                    session_id=request.session_id,
                    user_feedback=request.feedback,
                    online_learner=self.system.online_learner
                )

            # 6) 応答
            return self.ContextResponse(
                processed_text=self._postprocess(outputs['logits']),
                context_score=consistency_score,
                entities=self._extract_entities(outputs),
                consistency_check=final_consistency,
                meta_cognition_repair=meta_cognition_invoked
            )
        except Exception as e:
            self.system.logger.error(f"Processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Processing error")
        finally:
            latency = (datetime.now() - start_time).total_seconds()
            self.system.metrics['request_latency'].observe(latency)

    def _encode_text(self, text: str) -> dict:
        return self.inference_engine.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

    def _to_embedding(self, inputs: dict):
        device = self.inference_engine.device
        with torch.no_grad():
            outputs = self.inference_engine.base_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                return None
            vec = last_hidden_state[:, 0, :].cpu().numpy()
        return vec

    def _retrieve_context(self, session_id: str):
        cached = self.memory_manager.fetch_working_memory(session_id)
        if cached is not None and len(cached) > 0:
            return torch.tensor(cached, dtype=torch.float32)
        return None

    def _postprocess(self, logits: torch.Tensor) -> str:
        if logits is None:
            return "No logits available."
        ids = torch.argmax(logits, dim=-1)[0]
        return self.inference_engine.tokenizer.decode(ids, skip_special_tokens=True)

    def _extract_entities(self, outputs: dict):
        return ["Entity1", "Entity2"]
