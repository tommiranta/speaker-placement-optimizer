"""Tests for room_mode_optimizer.cli."""
from unittest.mock import patch

from click.testing import CliRunner

from room_mode_optimizer.cli import main

# A simple rectangular room URL for fast tests
SIMPLE_URL = (
    "https://www.vesalaasanen.com/tools/room-mode-calculator"
    "#poly,2.60,0.00,0.00,4.00,0.00,4.00,3.00,0.00,3.00"
    "|s,3.50,0.50,0.00,0.0,0.0,1,1"
    "|s,3.50,2.50,0.00,0.0,0.0,1,1"
    "|l,2.00,1.50,1.11"
    "|t21|a0.30"
)


class TestCliOptions:
    def test_missing_url_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Missing option '--url'" in result.output

    def test_help_shows_all_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for opt in ["--url", "--fix-listener", "--absorption",
                     "--move-fraction", "--top", "--reorigin", "--open-browser"]:
            assert opt in result.output

    def test_no_reorigin_keeps_original_coords(self):
        # URL with origin at (1,1)
        offset_url = (
            "https://www.vesalaasanen.com/tools/room-mode-calculator"
            "#poly,2.60,1.00,1.00,5.00,1.00,5.00,4.00,1.00,4.00"
            "|s,4.50,1.50,0.00,0.0,0.0,1,1"
            "|s,4.50,3.50,0.00,0.0,0.0,1,1"
            "|l,3.00,2.50,1.11"
            "|t21|a0.30"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["--url", offset_url, "--no-reorigin", "--top", "1"])
        assert result.exit_code == 0
        # With --no-reorigin, coords in output should reflect original (1-based)
        assert "1.00,1.00" in result.output  # polygon starts at (1,1)

    def test_reorigin_is_default(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        # With default reorigin, polygon starts at (0,0)
        assert "0.00,0.00" in result.output

    @patch("room_mode_optimizer.cli.click.launch")
    def test_open_browser_calls_launch(self, mock_launch):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1", "--open-browser"])
        assert result.exit_code == 0
        assert "Opening best result in browser" in result.output
        mock_launch.assert_called_once()
        call_url = mock_launch.call_args[0][0]
        assert call_url.startswith("https://www.vesalaasanen.com/")
        # URL should not be percent-encoded (#, | should be literal)
        assert "#poly," in call_url
        assert "|s," in call_url

    @patch("room_mode_optimizer.cli.click.launch")
    def test_no_open_browser_by_default(self, mock_launch):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        mock_launch.assert_not_called()

    def test_freq_max_option(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1", "--freq-max", "150"])
        assert result.exit_code == 0
        assert "20–150 Hz" in result.output
