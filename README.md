# DMHA (Dynamic Memory Hierarchy Architecture) の提案

本リポジトリは、大規模言語モデル (LLM) を用いた長期文脈管理と論理的一貫性の自己修正を目的とする、新しいアーキテクチャ DMHA (Dynamic Memory Hierarchy Architecture) の実装を集約している。Redis・Milvus・Faiss/HDBSCAN・PyTorch・FastAPI・LoRA (PEFT) などを組み合わせることで、長期記憶に相当する大規模履歴の管理とメタ認知制御、さらには神経シンボリック推論を融合し、高度な知的対話を実現することを狙っている。

---

## 目次

- [プロジェクトの目的](#プロジェクトの目的)
- [新規性](#新規性)
- [理論的背景](#理論的背景)
  - [1. 短期/長期メモリの統合](#1-短期長期メモリの統合)
  - [2. メタ認知制御と自己修正](#2-メタ認知制御と自己修正)
  - [3. 神経シンボリック推論](#3-神経シンボリック推論)
  - [4. オンライン学習とモデル最適化](#4-オンライン学習とモデル最適化)
- [ディレクトリ構成](#ディレクトリ構成)
- [インストールとセットアップ](#インストールとセットアップ)
- [動作の概要](#動作の概要)
  - [APIフロー](#apiフロー)
  - [メタ認知制御フロー](#メタ認知制御フロー)
- [今後の展望](#今後の展望)
- [ライセンスと注意事項](#ライセンスと注意事項)


---

## プロジェクトの目的
1.	長期的な文脈保持
数千〜数万トークンを超える会話履歴を安全かつ効率的に保持し、必要に応じて再利用することで、従来のLLMに見られる短いコンテキスト制限を突破する。
2.	矛盾検出と自己修正
大規模言語モデルがしばしば陥る事実矛盾や文脈不整合を、メタ認知制御モジュールと神経シンボリック推論によって自動検出し、自己修正を行う。
3.	継続学習 (オンライン学習)
LoRA (Low-Rank Adaptation) を用いたパラメータ効率の高い微調整により、ユーザーからのフィードバックや新情報を取り込み、リアルタイムでモデルをアップデートする。

---

## 新規性
•短期メモリ (Redis) + 長期メモリ (Milvus)
短期/長期メモリを階層的に管理し、Redisで最新会話を、高速かつ容量制限の少ないMilvusでベクトル検索型の大規模長期記憶を保持する。

•神経シンボリック推論の融合
自然言語をFact/Rule形式にパースし、矛盾をチェックする実運用レベルの仕組みを導入することで、単なるニューラル生成では扱いにくい論理的不整合の排除を目指す。

•メタ認知制御
Consistency Scoreなどを観測して文脈矛盾を検出し、自己修正ループを発動する実装例を備える。修正後には再度クラスタリングや要約を行い、さらにオンライン学習を挟むことでモデル自体を更新する。

•LoRAによるオンライン学習 + TorchScript再コンパイル
オンライン学習の後、再度TorchScript最適化を行うことで高速推論とモデルの自己更新を両立している。

---

理論的背景

1. 短期/長期メモリの統合

人間の脳科学では、短期記憶 (Working Memory) と長期記憶 (Long-Term Memory) が分化しているとされる。短期記憶は少量の情報を即座に扱う作業用メモリであり、長期記憶は大容量かつ永続性の高い記憶を保持する。

DMHAでは、短期記憶としてRedisを用いて最新の会話コンテキストを管理し、古くなった情報はMilvusに保存する。MilvusにはEmbeddingを通じてベクトル化された会話履歴を長期保存し、FaissやHDBSCANを活用してセマンティックなクラスタリングや類似検索を可能にしている。

2. メタ認知制御と自己修正

脳科学・認知科学的には、メタ認知は「自分自身の認知活動を監視し、必要ならば修正する」能力と定義される。本プロジェクトでは、この概念をメタ認知制御モジュールとして実装し、以下を実行する:
	•	整合性スコアの監視
推論時にConsistency Scoreを計算し、しきい値を超える不整合があれば矛盾と判断。
	•	自己修正ループ
過去のWorking Memoryを要約してLong-Term Memoryに格納し、短期領域をクリアするなど、文脈の再編を行う。
	•	クラスタリング再学習やオンライン学習へのトリガー
ユーザーからのフィードバックがあった場合や自己矛盾が顕在化した場合に、モデルのパラメータを更新する。

3. 神経シンボリック推論

LLMによるニューラル生成だけでは、論理的一貫性や矛盾検出を十分に行いにくい。そこで、NeuralSymbolicInterfaceがテキスト中の fact: / deny: / rule: を解析し、(entity, attribute, value, negation, confidence, scope) といった構造化Factやルールを得る。内部のKnowledgeBaseがFactやRuleを管理し、複雑な矛盾を検出する:
	•	Fact同士の衝突
同じエンティティ属性における肯定・否定の重複、あるいは異なる値の衝突など
	•	Rule違反
「もし条件がすべて成り立つならば…」というルールを満たすはずの結論が否定されている

こうした明示的論理構造を扱うことで、誤った回答や一貫性のない振る舞いをメタ認知制御と連携して是正する。

4. オンライン学習とモデル最適化

LoRA (Low-Rank Adaptation) を用いることで、大規模LLMの全パラメータを動かさずに少数の追加パラメータのみを学習し、新情報をリアルタイムで反映する。本プロジェクトでは、
	1.	微調整 (apply_online_learning)
ユーザーからの学習サンプル (input_text, target_text) を用いて、LoRA層のみを数ステップ学習
	2.	TorchScript再コンパイル (rescript_model)
学習済みモデルを再度 torch.jit.script し、optimize_for_inference → freeze で高速推論に対応

これにより、LLMの柔軟なオンライン学習と推論最適化を両立する。

---

## ディレクトリ構成
```text
my-dmha-project/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── context_system.py
│   ├── cluster_manager.py
│   ├── inference_engine.py
│   ├── memory_manager.py
│   ├── meta_cognition.py
│   ├── symbolic_interface.py
│   ├── api_service.py
│   ├── online_learning.py
│   └── utils/
│       ├── __init__.py
│       └── torchscript_utils.py
├── requirements.txt
├── Dockerfile
└── README.md
```
	•	main.py: アプリのエントリポイント (FastAPI + uvicorn)
	•	config.py: 環境変数やハイパーパラメータの管理
	•	context_system.py: DMHA中枢。メタ認知やシンボリック、オンライン学習モジュールを初期化
	•	memory_manager.py: Redisで短期、Milvusに長期メモリを格納し、Faiss/HDBSCANでクラスタリング
	•	meta_cognition.py: 一貫性スコアを監視し、矛盾時に自己修正・オンライン学習トリガー
	•	symbolic_interface.py: Fact/Rule形式で論理矛盾を検出するシンボリック推論 (実運用レベル)
	•	online_learning.py: LoRAによるオンライン学習を実装し、学習後はTorchScript再コンパイルを行う
	•	utils/torchscript_utils.py: モデル読み込み → TorchScript化のユーティリティ

    ---

## インストールとセットアップ
1.	依存パッケージのインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

2.	Dockerビルド (任意)
```bash
docker build -t my-dmha-project .
```
GPUを使う場合はnvidia/cudaベースのイメージやfaiss-gpu対応環境が必要

3.	実行
```bash
docker run -p 8080:8080 my-dmha-project
```

---

## 動作の概要

APIフロー
	•	POST /auth/token
ユーザー名からJWTを発行
	•	POST /v1/contextualize
JWTを付与しつつテキストを送ると、
	1.	テキストを埋め込み (短期メモリに保存)
	2.	LLM推論 (TorchScript or Raw)
	3.	整合性スコア + シンボリック推論
	4.	矛盾あればメタ認知制御 → 自己修正
	5.	オンライン学習 (LoRA) → TorchScript再コンパイル

メタ認知制御フロー
	1.	Consistency Scoreとシンボリック判断を合わせ、閾値を超えたら自己修正を発動
	2.	過去のWorking Memoryを要約してLong-Term Memoryに格納、短期をクリア
	3.	(任意) クラスタリング再学習・LoRA学習を実行
	4.	学習後はTorchScript再適用し、最適化済みモデルで引き続き推論

---

## 今後の展望
	•	高度なルールエンジンとの連携
PrologやZ3等の外部シンボリックエンジンを統合し、より洗練された論理推論を可能にしたい。
	•	拡張的オンライン学習
LoRAだけでなく、p-tuning v2やAdapterなど他のPEFT手法を取り入れ、短い学習時間で効果を最大化するアプローチを検討。
	•	分散クラスタリング
Faiss/HDBSCANを水平スケールして、超大規模ベクトルを扱いつつリアルタイムに近い更新を行う仕組みを強化する。
	•	メタ認知モデルの拡張
単なる閾値監視でなく、より高次のメタ推論(計画・監視など)を導入し、多段階に再構築するメカニズムを模索。

---

## ライセンスと注意事項
	•	Llama 2などの大規模モデルを使う場合、Meta社の利用規約およびライセンスを順守する必要がある。
	•	本実装は研究・学習目的で公開しており、商用利用や本番導入には追加の検討が必要。
	•	オンライン学習機能はLoRAに限定して実装しており、大規模環境でのメモリ・リソース使用量を十分考慮すること。

---

以上が、DMHAプロジェクトの全体像と理論的背景を含むREADMEである。脳科学的な短期/長期記憶の階層モデル、メタ認知的自己修正、神経シンボリック推論、オンライン学習の各要素を結合し、長期文脈保持と論理的一貫性を両立する次世代LLMの可能性を探究している。興味があればコードを参照し、自身のユースケースに合わせて拡張や運用テストを進めることを推奨する。