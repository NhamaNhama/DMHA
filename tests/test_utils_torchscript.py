import pytest
import torch
from unittest.mock import patch, MagicMock
from app.utils.torchscript_utils import load_and_torchscript_model


@pytest.fixture
def mock_auto_model():
    """AutoModelのモックを提供"""
    model_mock = MagicMock()
    model_mock.config.hidden_size = 768
    model_mock.to = MagicMock(return_value=model_mock)
    model_mock.eval = MagicMock(return_value=model_mock)
    return model_mock


@pytest.fixture
def mock_auto_tokenizer():
    """AutoTokenizerのモックを提供"""
    tokenizer_mock = MagicMock()
    return tokenizer_mock


@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
@patch('torch.jit.script')
def test_load_and_torchscript_model_from_hf(mock_script, mock_tokenizer, mock_model, mock_auto_model):
    """HuggingFaceからのモデル読み込みテスト"""
    # モックの設定
    mock_model.return_value = mock_auto_model
    mock_script.return_value = MagicMock()

    # テスト対象の関数を実行
    model = load_and_torchscript_model(
        model_name="test-model",
        device=torch.device("cpu"),
        script_path=None
    )

    # モデルが読み込まれたことを確認
    mock_model.assert_called_once_with("test-model")

    # モデルがGPUに移動され、評価モードに設定されたことを確認
    mock_auto_model.to.assert_called_once_with(torch.device("cpu"))
    mock_auto_model.eval.assert_called_once()

    # TorchScriptでコンパイルされたことを確認
    mock_script.assert_called_once_with(mock_auto_model)

    # 正しいモデルが返されることを確認
    assert model == mock_script.return_value


@patch('torch.jit.load')
def test_load_and_torchscript_model_from_path(mock_jit_load, mock_auto_model):
    """事前にコンパイルされたモデルの読み込みテスト"""
    # モックの設定
    mock_jit_load.return_value = mock_auto_model

    # テスト対象の関数を実行
    model = load_and_torchscript_model(
        model_name="unused-model-name",
        device=torch.device("cpu"),
        script_path="/path/to/model.pt"
    )

    # 保存されたモデルが読み込まれたことを確認
    mock_jit_load.assert_called_once_with(
        "/path/to/model.pt", map_location=torch.device("cpu"))

    # モデルがGPUに移動されたことを確認
    mock_auto_model.to.assert_called_once_with(torch.device("cpu"))

    # 正しいモデルが返されることを確認
    assert model == mock_auto_model


@patch('transformers.AutoModel.from_pretrained')
@patch('torch.jit.script')
def test_load_and_torchscript_model_exception(mock_script, mock_model, mock_auto_model):
    """TorchScriptコンパイル失敗時のテスト"""
    # モックの設定
    mock_model.return_value = mock_auto_model
    mock_script.side_effect = Exception("TorchScript compilation error")

    # テスト対象の関数を実行
    model = load_and_torchscript_model(
        model_name="test-model",
        device=torch.device("cpu"),
        script_path=None
    )

    # モデルが読み込まれたことを確認
    mock_model.assert_called_once_with("test-model")

    # TorchScriptが例外を投げても、元のモデルが返されることを確認
    assert model == mock_auto_model
