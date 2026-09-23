# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for CLI module."""

import pytest
from click.testing import CliRunner


class TestCLI:
    """Tests for command-line interface."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_version_command(self, runner):
        """Test version command."""
        from unbihexium.cli import cli
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
    
    def test_help_command(self, runner):
        """Test help command."""
        from unbihexium.cli import cli
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Unbihexium" in result.output
