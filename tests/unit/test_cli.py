"""Unit tests for CLI module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner
from model_zoo.cli import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls."""
    with patch("model_zoo.cli.subprocess") as mock:
        mock.run.return_value.returncode = 0
        mock.run.return_value.stdout = ""
        mock.run.return_value.stderr = ""
        yield mock


def test_init_command(runner, tmp_path):
    """Test model family initialization."""
    with patch("model_zoo.cli.Path") as mock_path:
        mock_path.return_value = tmp_path / "dist" / "clip"
        mock_path.return_value.mkdir = Mock()
        
        result = runner.invoke(app, ["init", "clip", "--target", "k3d"])
        
        assert result.exit_code == 0
        assert "Initializing clip" in result.stdout


def test_deploy_command_k3d(runner, mock_subprocess, tmp_path):
    """Test deployment to k3d."""
    # Mock file operations
    with patch("model_zoo.cli.Path") as mock_path_class:
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "template: {{MODEL_NAME}}"
        mock_path.write_text = Mock()
        mock_path.glob.return_value = [Mock(name="head.yaml")]
        mock_path_class.return_value = mock_path
        
        # Mock kubectl get service response
        mock_subprocess.run.return_value.stdout = '{"status": {"loadBalancer": {}}}'
        
        # Mock Ray
        with patch("model_zoo.cli.ray") as mock_ray:
            mock_actor = Mock()
            mock_ray.get_actor.return_value = mock_actor
            mock_ray.get.return_value = {"similarity": 0.8}
            
            result = runner.invoke(app, [
                "deploy", "test-model",
                "--weights", "gs://bucket/weights",
                "--gpu", "t4",
                "--target", "k3d"
            ])
            
            assert result.exit_code == 0
            assert "deployed successfully" in result.stdout


def test_logs_command(runner, mock_subprocess):
    """Test logs command."""
    result = runner.invoke(app, ["logs", "test-model"])
    
    assert result.exit_code == 0
    mock_subprocess.run.assert_called()


def test_list_command(runner, mock_subprocess):
    """Test list command."""
    # Mock kubectl response
    mock_subprocess.run.return_value.stdout = '''
    {
        "items": [
            {
                "metadata": {
                    "labels": {"model": "test-model"}
                },
                "status": {"readyReplicas": 1},
                "spec": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [{"image": "test:latest"}]
                        }
                    }
                }
            }
        ]
    }
    '''
    
    result = runner.invoke(app, ["list"])
    
    assert result.exit_code == 0
    assert "test-model" in result.stdout


def test_delete_command(runner, mock_subprocess, tmp_path):
    """Test delete command."""
    with patch("model_zoo.cli.Path") as mock_path_class:
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path
        
        result = runner.invoke(app, ["delete", "test-model", "--yes"])
        
        assert result.exit_code == 0
        mock_subprocess.run.assert_called()


@patch("model_zoo.cli.ray")
def test_infer_clip(mock_ray, runner, mock_subprocess, tmp_path):
    """Test inference with CLIP model."""
    # Create test image file
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake image data")
    
    # Mock service response
    mock_subprocess.run.return_value.stdout = '''
    {
        "spec": {"type": "ClusterIP"},
        "status": {"loadBalancer": {}}
    }
    '''
    
    # Mock Ray inference
    mock_actor = Mock()
    mock_ray.get_actor.return_value = mock_actor
    mock_ray.get.return_value = {"similarity": 0.85}
    
    result = runner.invoke(app, [
        "infer", "clip-model",
        "--file", str(test_image),
        "--text", "a test image"
    ])
    
    assert result.exit_code == 0
    assert "similarity" in result.stdout


def test_infer_missing_params(runner):
    """Test inference with missing required parameters."""
    result = runner.invoke(app, [
        "infer", "clip-model",
        "--file", "nonexistent.jpg"
    ])
    
    assert result.exit_code == 1
    assert "not found" in result.stdout