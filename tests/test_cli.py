"""Tests for speaker_placement_optimizer.cli."""
import re
from unittest.mock import patch

from click.testing import CliRunner

from speaker_placement_optimizer.cli import main

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
                     "--move-fraction", "--max-speaker-depth",
                     "--top", "--freq-max", "--reorigin", "--open-browser"]:
            assert opt in result.output

    def test_default_top_is_3(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL])
        assert result.exit_code == 0
        assert "#3" in result.output
        assert "#4" not in result.output

    def test_no_reorigin_keeps_original_coords(self):
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
        assert "1.00,1.00" in result.output

    def test_reorigin_is_default(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        assert "0.00,0.00" in result.output

    @patch("speaker_placement_optimizer.cli._open_url")
    def test_open_browser_calls_open_url(self, mock_open):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1", "--open-browser"])
        assert result.exit_code == 0
        assert "Opening best result in browser" in result.output
        mock_open.assert_called_once()
        call_url = mock_open.call_args[0][0]
        assert "#poly," in call_url
        assert "|s," in call_url

    @patch("speaker_placement_optimizer.cli._open_url")
    def test_no_open_browser_by_default(self, mock_open):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        mock_open.assert_not_called()

    def test_freq_max_option(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1", "--freq-max", "150"])
        assert result.exit_code == 0
        assert "20–150 Hz" in result.output

    def test_output_shows_wall_distances_in_cm(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        assert "cm from front wall" in result.output
        assert "cm from side wall" in result.output

    def test_output_shows_listener_side_walls(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        # Listener line should show distances to both side walls
        assert "cm from side walls" in result.output

    def test_output_no_grid_coordinates(self):
        """Output should not show raw (x, y) grid coordinates."""
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        # The results section should not have "(x.xx, y.yy)" coordinate pairs
        # (only the URL contains raw coords, which is fine)
        lines = result.output.split("\n")
        result_started = False
        for line in lines:
            if line.strip().startswith("#1"):
                result_started = True
            if result_started and line.strip().startswith("Speaker"):
                assert "(" not in line, f"Grid coords found in result: {line}"
