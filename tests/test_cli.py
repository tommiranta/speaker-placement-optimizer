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


def _extract_redirect_url(output):
    """Extract the URL from the HTML redirect file referenced in OSC 8 link."""
    m = re.search(r"file://(/[^\033]+\.html)", output)
    if not m:
        return None
    with open(m.group(1)) as f:
        html = f.read()
    m2 = re.search(r'url=([^"]+)"', html)
    return m2.group(1) if m2 else None


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
        url = _extract_redirect_url(result.output)
        assert url is not None, "No redirect file in output"
        assert "1.00,1.00" in url

    def test_reorigin_is_default(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        url = _extract_redirect_url(result.output)
        assert url is not None, "No redirect file in output"
        assert "0.00,0.00" in url

    @patch("speaker_placement_optimizer.cli._open_url")
    def test_open_browser_calls_open_url(self, mock_open):
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1", "--open-browser"])
        assert result.exit_code == 0
        assert "Opening best result in browser" in result.output
        mock_open.assert_called_once()

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
        assert "cm from side walls" in result.output

    def test_redirect_url_has_literal_pipe(self):
        """The URL inside the HTML redirect must have literal |, not %7C."""
        runner = CliRunner()
        result = runner.invoke(main, ["--url", SIMPLE_URL, "--top", "1"])
        assert result.exit_code == 0
        url = _extract_redirect_url(result.output)
        assert url is not None
        assert "|s," in url, f"URL has encoded pipes: {url}"
        assert "%7C" not in url
