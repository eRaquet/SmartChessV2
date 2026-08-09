# ruff: noqa: S101
"""Tests for the game module."""

from smartchess.agent import RandomAgentConfig, create_agent
from smartchess.board import BoardConfig, create_board
from smartchess.game import StandardGameConfig, create_game
from smartchess.types import BoardStatus


def test_game_formed_elements() -> None:
    """Test playing a game with preformed elements."""
    board_config = BoardConfig()
    board = create_board(board_config)

    agent_config = RandomAgentConfig(seed=0)
    random_agent = create_agent(agent_config)

    game_config = StandardGameConfig(
        white=random_agent, black=random_agent, board=board, renderer=None, collect=False
    )
    game = create_game(game_config)
    game.play_game()
    assert board.terminated


def test_game_unformed_elements() -> None:
    """Test playing a game with only configs for elements."""
    board_config = BoardConfig()

    agent_config = RandomAgentConfig(seed=0)

    game_config = StandardGameConfig(
        white=agent_config, black=agent_config, board=board_config, renderer=None, collect=False
    )
    game = create_game(game_config)
    game.play_game()


def test_game_collect() -> None:
    """Test the collection functionality of a game."""
    board_config = BoardConfig()
    board = create_board(board_config)

    agent_config = RandomAgentConfig(seed=0)
    random_agent = create_agent(agent_config)

    game_config = StandardGameConfig(
        white=random_agent, black=random_agent, board=board, renderer=None, collect=True
    )
    game = create_game(game_config)
    log = game.play_game()

    assert log is not None
    assert log.game.result is not None
    assert log.game.result in BoardStatus.TERMINATED

    assert log.game.ply_number is not None
    assert log.game.ply_number > 1
