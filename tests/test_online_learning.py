from app.online_learning import OnlineLearner
import pytest
import torch
from unittest.mock import MagicMock, patch
import sys

# peftモジュールをモック化
peft_mock = MagicMock()
lora_config_mock = MagicMock()
get_peft_model_mock = MagicMock()
peft_mock.LoraConfig = lora_config_mock
peft_mock.get_peft_model = get_peft_model_mock

# sys.modulesにモックを追加してインポートエラーを回避
sys.modules['peft'] = peft_mock

# online_learningをインポート


@pytest.fixture
def mock_base_model():
    """ベースモデルのモックを提供"""
    model_mock = MagicMock()
    model_mock.train = MagicMock()
    model_mock.eval = MagicMock()
    model_mock.to = MagicMock(return_value=model_mock)
    return model_mock


@pytest.fixture
def mock_tokenizer():
    """トークナイザーのモックを提供"""
    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = "[PAD]"
    tokenizer_mock.eos_token = "[EOS]"
    return tokenizer_mock


@pytest.fixture
def online_learner(mock_base_model, mock_tokenizer):
    """テスト用のオンライン学習モジュールを作成"""
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        # OnlineLearnerのインスタンス作成
        learner = OnlineLearner(
            base_model=mock_base_model,
            device=torch.device("cpu"),
            tokenizer_name="test-model",
            lr=1e-4,
            num_steps=5
        )
        # テスト用に簡易化
        learner.peft_model = mock_base_model
        return learner


def test_init(online_learner, mock_base_model, mock_tokenizer):
    """初期化のテスト"""
    assert online_learner.device == torch.device("cpu")
    assert online_learner.lr == 1e-4
    assert online_learner.num_steps == 5
    assert online_learner.tokenizer == mock_tokenizer


def test_apply_online_learning_no_samples(online_learner):
    """サンプルがない場合のオンライン学習テスト"""
    samples = []
    online_learner.apply_online_learning(samples)
    # 何も起きないこと


@patch('torch.optim.AdamW')
def test_apply_online_learning_success(mock_adamw, online_learner):
    """オンライン学習成功のテスト"""
    # テスト準備
    mock_optimizer = MagicMock()
    mock_adamw.return_value = mock_optimizer
    output = MagicMock()
    output.loss = torch.tensor(0.5)
    online_learner.peft_model.return_value = output

    # テストデータ
    samples = [("入力テキスト", "出力テキスト")]

    # テスト実行
    online_learner.apply_online_learning(samples)

    # 検証
    assert mock_adamw.called


@patch('torch.jit.script')
def test_rescript_model(mock_script, online_learner):
    """モデルの再スクリプト化テスト"""
    # テスト実行
    online_learner.rescript_model()

    # 検証
    assert mock_script.called
