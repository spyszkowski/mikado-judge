"""Game state management for Mikado Judge."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from mikado.detect import Stick
from mikado.judge import JudgmentResult

logger = logging.getLogger(__name__)


class TurnPhase(Enum):
    """Phases within a single player turn."""
    WAITING_FOR_BEFORE = auto()   # Waiting for the player to signal they're ready
    WAITING_FOR_AFTER = auto()    # Before captured; waiting for after the move
    JUDGING = auto()              # Running detection and judgment
    COMPLETE = auto()             # Turn result is available


@dataclass
class TurnRecord:
    """Record of a single player turn."""
    player_id: int
    turn_number: int
    result: JudgmentResult | None = None
    removed_stick_class: str | None = None
    points_scored: int = 0


@dataclass
class Player:
    """A player in the game."""
    player_id: int
    name: str
    score: int = 0
    turns: list[TurnRecord] = field(default_factory=list)


@dataclass
class GameState:
    """Full game state including players, scores, and turn history."""
    players: list[Player]
    current_player_index: int = 0
    turn_number: int = 0
    phase: TurnPhase = TurnPhase.WAITING_FOR_BEFORE
    current_turn: TurnRecord | None = None
    game_over: bool = False

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]


class GameManager:
    """Manages the flow of a Mikado game.

    Tracks players, turns, and scoring. Coordinates the
    capture → judge → score → next_turn cycle.

    Args:
        player_names: Names of players (2 or more).
        points_config: Mapping from class name to point value.
    """

    def __init__(
        self,
        player_names: list[str],
        points_config: dict[str, int] | None = None,
    ) -> None:
        if len(player_names) < 2:
            raise ValueError("At least 2 players required")
        self.points_config: dict[str, int] = points_config or {}
        self.state = GameState(
            players=[Player(player_id=i, name=name) for i, name in enumerate(player_names)]
        )
        logger.info("Game started with players: %s", player_names)

    @classmethod
    def from_config(cls, player_names: list[str], config: dict[str, Any]) -> "GameManager":
        """Create a GameManager loading point values from classes.yaml data."""
        return cls(player_names=player_names, points_config=config.get("points", {}))

    # ------------------------------------------------------------------
    # Turn flow
    # ------------------------------------------------------------------

    def start_turn(self) -> None:
        """Begin a new turn for the current player."""
        if self.state.game_over:
            raise RuntimeError("Game is over")
        self.state.turn_number += 1
        self.state.current_turn = TurnRecord(
            player_id=self.state.current_player.player_id,
            turn_number=self.state.turn_number,
        )
        self.state.phase = TurnPhase.WAITING_FOR_AFTER
        logger.info(
            "Turn %d started: %s", self.state.turn_number, self.state.current_player.name
        )

    def record_judgment(self, result: JudgmentResult) -> None:
        """Record the judgment for the current turn and update scores."""
        if self.state.current_turn is None:
            raise RuntimeError("No turn in progress — call start_turn() first")

        self.state.current_turn.result = result
        self.state.phase = TurnPhase.JUDGING

        if not result.fault and result.removed_sticks:
            removed = result.removed_sticks[0]
            class_name = removed.class_name
            points = self.points_config.get(class_name, 0)
            self.state.current_turn.removed_stick_class = class_name
            self.state.current_turn.points_scored = points
            self.state.current_player.score += points
            logger.info(
                "%s scored %d points (removed %s)",
                self.state.current_player.name, points, class_name,
            )
        elif result.fault:
            logger.info("%s: FAULT — turn ends, no points", self.state.current_player.name)

        self.state.current_player.turns.append(self.state.current_turn)
        self.state.phase = TurnPhase.COMPLETE

    def next_turn(self) -> None:
        """Advance to the next player's turn."""
        self.state.current_player_index = (
            self.state.current_player_index + 1
        ) % len(self.state.players)
        self.state.current_turn = None
        self.state.phase = TurnPhase.WAITING_FOR_BEFORE
        logger.info("Next player: %s", self.state.current_player.name)

    def end_game(self) -> list[Player]:
        """End the game and return players sorted by score (descending)."""
        self.state.game_over = True
        ranked = sorted(self.state.players, key=lambda p: p.score, reverse=True)
        logger.info(
            "Game over. Winner: %s with %d points",
            ranked[0].name, ranked[0].score,
        )
        return ranked

    def scoreboard(self) -> str:
        """Return a formatted scoreboard string."""
        lines = ["=== Scoreboard ==="]
        ranked = sorted(self.state.players, key=lambda p: p.score, reverse=True)
        for i, player in enumerate(ranked, 1):
            lines.append(f"  {i}. {player.name}: {player.score} pts")
        return "\n".join(lines)
