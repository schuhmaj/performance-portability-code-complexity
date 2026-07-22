"""Tests for the unified ppbcc command dispatcher."""

from ppbcc.__main__ import build_parser, main


def test_parser_lists_all_workflows():
    parser = build_parser()
    for command in ("benchmark", "code-complexity", "p3analysis"):
        args = parser.parse_args([command, "--help"])
        assert args.command == command
        assert args.arguments == ["--help"]


def test_no_subcommand_prints_help(capsys):
    assert main([]) == 0
    assert "code-complexity" in capsys.readouterr().out
