"""Class for displaying chess boards that works nominally in conjunction with a board environment."""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # disable the pygame welcome message

import chess
import pygame as pg

from modules.config import PROJECT_PATH


class Display:
    """Class to display boards associated with the Board environment."""

    def __init__(self) -> None:
        """Initiate the display."""
        pg.init()

        # dictionary of images associated with each piece
        self.images = {
            (chess.PAWN, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whitePawn.png"),
            (chess.KNIGHT, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whiteKnight.png"),
            (chess.BISHOP, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whiteBishop.png"),
            (chess.ROOK, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whiteRook.png"),
            (chess.QUEEN, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whiteQueen.png"),
            (chess.KING, chess.WHITE): pg.image.load(PROJECT_PATH / "images" / "whiteKing.png"),
            (chess.PAWN, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackPawn.png"),
            (chess.KNIGHT, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackKnight.png"),
            (chess.BISHOP, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackBishop.png"),
            (chess.ROOK, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackRook.png"),
            (chess.QUEEN, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackQueen.png"),
            (chess.KING, chess.BLACK): pg.image.load(PROJECT_PATH / "images" / "blackKing.png"),
        }

        width_hight = 520
        self.surf = pg.display.set_mode((width_hight, width_hight))

        self.selected_square = None
        self.highlight_mask: list[chess.Square] = []

        self.display_board(chess.Board())

    # display board
    def display_board(
        self, board: chess.Board, board_map: dict[chess.Square, chess.Piece] | None = None
    ) -> None:
        """

        Display a board object.

        Parameters
        ----------
        board : chess.Board
            Board to display
        board_map : dict[chess.Piece, chess.Square] | None, optional
            Map of pieces, optional for efficiency, by default None
        """
        pg.event.pump()  # temporary fix for flushing pygame events

        if board_map is None:
            board_map = board.piece_map()

        self.highlight_mask = []

        # create a background color
        self.surf.fill(pg.Color(100, 75, 25))

        # render each square
        for i in range(64):
            column = i % 8
            row = 7 - int((i - i % 8) / 8)
            square = i
            light_dark = (column + row + 1) % 2

            # highlight if the current square is a possible move for the selected piece
            if self.selected_square is not None:
                highlight = board.find_move(self.selected_square, square) in board.legal_moves
                self.highlight_mask.append(square)
            else:
                highlight = False

            selected = square == self.selected_square

            # if our king is on this square and check is placed on the board
            if (
                square in board_map
                and board_map[square].color == board.turn
                and board_map[square].piece_type == chess.KING
                and board.is_check() is True
            ):
                color = (100, 0, 0)

            elif highlight:
                color = (15 + 130 * light_dark, 70 + 160 * light_dark, 40 * light_dark)

            else:
                color = (
                    (0, 0, 100)
                    if selected
                    else (60 + 160 * light_dark, 35 + 130 * light_dark, 15 + 40 * light_dark)
                )

            # draw the square color
            pg.draw.rect(self.surf, color, pg.Rect(20 + 60 * column, 20 + 60 * row, 60, 60))

            piece = board_map.get(i)

            # if there is a piece at this square, render it onto the board
            if piece is not None:
                # place a piece if one exist on that square
                self.surf.blit(
                    self.images[(piece.piece_type, piece.color)],
                    pg.Rect(20 + 60 * column, 20 + 60 * row, 60, 60),
                )

        pg.display.update()

    def end_display(self) -> None:
        """Close display."""
        pg.quit()
