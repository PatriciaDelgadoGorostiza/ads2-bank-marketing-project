import pytest
from typer.testing import CliRunner

from src.cli import app


@pytest.mark.integration
def test_pipeline_cli_exposes_train_and_predict_commands():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "predict" in result.output
