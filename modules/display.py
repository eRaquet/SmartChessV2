"""Module for displaying chess boards."""

import os

from modules.utils import get_action

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # disable the pygame welcome message

import chess
import pygame as pg

from modules.chess_types import RESIGN, Action, MoveVector
from modules.config import BOARD_RIM_THICKNESS, BOARD_WIDTH, FPS, PROJECT_PATH, SQUARE_WIDTH


class Display:
    """Class to display boards associated with the Board environment."""

    def __init__(self) -> None:
        """Initiate the display."""
        pg.init()

        self._clock = pg.time.Clock()

        # dictionary of images associated with each piece
        self._images = {
            (chess.PAWN, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whitePawn.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.KNIGHT, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whiteKnight.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.BISHOP, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whiteBishop.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.ROOK, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whiteRook.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.QUEEN, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whiteQueen.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.KING, chess.WHITE): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "whiteKing.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.PAWN, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackPawn.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.KNIGHT, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackKnight.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.BISHOP, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackBishop.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.ROOK, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackRook.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.QUEEN, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackQueen.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
            (chess.KING, chess.BLACK): pg.transform.smoothscale(
                pg.image.load(PROJECT_PATH / "images" / "blackKing.png"),
                (SQUARE_WIDTH, SQUARE_WIDTH),
            ),
        }

        width_hight = BOARD_WIDTH + 2 * BOARD_RIM_THICKNESS
        self._surf = pg.display.set_mode((width_hight, width_hight))

        self._selected_square = None
        self._highlight_mask: list[chess.Square] = []

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

        self._highlight_mask = []

        # create a background color
        self._surf.fill(pg.Color(100, 75, 25))

        # render each square
        for i in range(64):
            column = i % 8
            row = 7 - int((i - i % 8) / 8)
            square = i
            light_dark = (column + row + 1) % 2

            # highlight if the current square is a possible move for the selected piece
            if self._selected_square is not None:
                highlight = chess.Move(self._selected_square, square) in board.legal_moves

                if highlight:
                    self._highlight_mask.append(square)
            else:
                highlight = False

            selected = square == self._selected_square

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
                self._surf,
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
                self._surf.blit(
                    self._images[(piece.piece_type, piece.color)],
                    pg.Rect(
                        BOARD_RIM_THICKNESS + SQUARE_WIDTH * column,
                        BOARD_RIM_THICKNESS + SQUARE_WIDTH * row,
                        SQUARE_WIDTH,
                        SQUARE_WIDTH,
                    ),
                )

        pg.display.update()

    def get_user_input(
        self,
        board: chess.Board,
        moves: MoveVector,
        board_map: dict[chess.Square, chess.Piece] | None = None,
    ) -> Action | None:
        """

        Check if there is a user input.

        This function enforces a frames per second limit, but does not implement an internal loop
        currently.  That feature could be added at some point, if needed, but this way we can avoid
        wierd infinite loops.

        Parameters
        ----------
        board : chess.Board
            chess.board object which is being displayed
        moves : MoveVector
            legal moves from the current position
        board_map : dict[chess.Square, chess.Piece] | None, optional
            map of pieces on board, by default None

        Returns
        -------
        Action | None
            action chosen by user, or None if no action yet selected
        """
        if board_map is None:
            board_map = board.piece_map()

        self._clock.tick(FPS)

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
                    if self._selected_square is None:
                        if square in board_map and board_map[square].color == board.turn:
                            self._selected_square = square
                            self.display_board(board, board_map=board_map)

                    # ...square is selected
                    else:
                        # square is a valid square to move to
                        if square in self._highlight_mask:
                            # if the move is a pawn promotion
                            if board_map[self._selected_square].piece_type == chess.PAWN and (
                                chess.square_rank(square) == 7 or chess.square_rank(square) == 0  # noqa: PLR2004
                            ):
                                user_input = chess.Move(self._selected_square, square, chess.QUEEN)
                            user_input = chess.Move(self._selected_square, square)
                            self._selected_square = None

                        # square is not a vlid square to move to, but is a square with a piece of
                        # our color
                        elif square in board_map and board_map[square].color == board.turn:
                            self._selected_square = (
                                square if self._selected_square != square else None
                            )
                            self.display_board(board, board_map=board_map)

                        # square is an "unselectable" square (empty or opponent)
                        else:
                            self._selected_square = None
                            self.display_board(board, board_map=board_map)

                        # if valid user input was created, return that
                        if user_input:
                            return get_action(user_input, moves)

            # return a resignation event if the window was exited
            if event.type == pg.QUIT:
                self.exit()
                return RESIGN
        return None

    def exit(self) -> None:
        """Close the display window."""
        pg.quit()
