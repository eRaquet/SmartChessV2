"""Class for displaying chess boards that works nominally in conjunction with a board environment."""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # disable the pygame welcome message

import chess
import pygame as pg

from modules.config import FPS, PROJECT_PATH

BOARD_RIM_THICKNESS = 20
BOARD_WIDTH = 480
SQUARE_WIDTH = BOARD_WIDTH / 8


class Display:
    """Class to display boards associated with the Board environment."""

    def __init__(self) -> None:
        """Initiate the display."""
        pg.init()

        self.clock = pg.time.Clock()

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
                highlight = chess.Move(self.selected_square, square) in board.legal_moves
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
            pg.draw.rect(
                self.surf,
                color,
                pg.Rect(
                    BOARD_RIM_THICKNESS + SQUARE_WIDTH * column,
                    BOARD_RIM_THICKNESS + SQUARE_WIDTH * row,
                    SQUARE_WIDTH,
                    SQUARE_WIDTH,
                ),
            )

            piece = board_map.get(i)

            # if there is a piece at this square, render it onto the board
            if piece is not None:
                # place a piece if one exist on that square
                self.surf.blit(
                    self.images[(piece.piece_type, piece.color)],
                    pg.Rect(
                        BOARD_RIM_THICKNESS + SQUARE_WIDTH * column,
                        BOARD_RIM_THICKNESS + SQUARE_WIDTH * row,
                        SQUARE_WIDTH,
                        SQUARE_WIDTH,
                    ),
                )

        pg.display.update()

    def get_user_input(
        self, board: chess.Board, board_map: dict[chess.Square, chess.Piece] | None = None
    ) -> chess.Move | None:
        """

        Check if there is a user input.

        Parameters
        ----------
        board : chess.Board
            chess.board object which is being displayed
        board_map : dict[chess.Square, chess.Piece] | None, optional
            map of pieces on board, by default None
        """
        if board_map is None:
            board_map = board.piece_map()

        self.clock.tick(FPS)

        # get events
        events = pg.event.get()

        user_input = None

        for event in events:
            # if button pressed...
            if event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()

                # if mouse is over the board
                if (
                    pos[0] >= BOARD_RIM_THICKNESS and pos[0] <= BOARD_WIDTH + BOARD_RIM_THICKNESS
                ) and (
                    pos[1] >= BOARD_RIM_THICKNESS and pos[1] <= BOARD_WIDTH + BOARD_RIM_THICKNESS
                ):
                    # shift position
                    pos = (pos[0] - BOARD_RIM_THICKNESS, pos[1] - BOARD_RIM_THICKNESS)

                    column = int(pos[0] / SQUARE_WIDTH)
                    row = 7 - int(pos[1] / SQUARE_WIDTH)
                    square = column + 8 * row

                    # ...no square selected->select square
                    if self.selected_square is None:
                        if square in board_map and board_map[square].color == board.turn:
                            self.selected_square = square
                            self.display_board(board, board_map=board_map)

                    # ...square is selected
                    else:
                        if square in board_map and board_map[square].color == board.turn:
                            self.selected_square = (
                                square if self.selected_square != square else None
                            )
                            self.display_board(board, board_map=board_map)

                        if square in self.highlight_mask:
                            # if the move is a pawn promotion
                            if board_map[self.selected_square].piece_type == chess.PAWN and (
                                chess.square_rank(square) == 7 or chess.square_rank(square) == 0  # noqa: PLR2004
                            ):
                                user_input = chess.Move(self.selected_square, square, chess.QUEEN)
                            user_input = chess.Move(self.selected_square, square)

                        self.selected_square = None
                        self.display_board(board, board_map=board_map)

                        if user_input:
                            return user_input
        return None
