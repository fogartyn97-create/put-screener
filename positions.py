"""Position tracker — CRUD operations backed by a local JSON file."""
from __future__ import annotations
import json
import uuid
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent / "positions.json"


def _load() -> dict:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text())
    return {"positions": []}


def _save(data: dict):
    DB_PATH.write_text(json.dumps(data, indent=2, default=str))


def all_positions() -> list[dict]:
    return _load()["positions"]


def add_position(ticker: str, strike: float, credit: float, expiration: str,
                 open_date: str, contracts: int = 1, notes: str = "",
                 rolled_from_id: str | None = None) -> dict:
    data = _load()
    # Inherit roll_chain_id from parent if this is a roll
    chain_id = None
    if rolled_from_id:
        parent = next((p for p in data["positions"] if p["id"] == rolled_from_id), None)
        chain_id = parent["roll_chain_id"] if parent else str(uuid.uuid4())
    else:
        chain_id = str(uuid.uuid4())

    pos = {
        "id": str(uuid.uuid4()),
        "ticker": ticker.upper(),
        "strike": float(strike),
        "credit": float(credit),
        "contracts": int(contracts),
        "expiration": expiration,
        "open_date": open_date,
        "status": "active",
        "close_date": None,
        "close_debit": None,
        "outcome": None,
        "notes": notes,
        "roll_chain_id": chain_id,
        "rolled_from_id": rolled_from_id,
    }
    data["positions"].append(pos)
    _save(data)
    return pos


def update_position(pos_id: str, outcome: str, close_date: str,
                    close_debit: float = 0.0, notes: str = "") -> dict | None:
    data = _load()
    for pos in data["positions"]:
        if pos["id"] == pos_id:
            pos["outcome"] = outcome
            pos["close_date"] = close_date
            pos["close_debit"] = float(close_debit)
            pos["status"] = "closed"
            if notes:
                pos["notes"] = notes
            _save(data)
            return pos
    return None


def delete_position(pos_id: str) -> bool:
    data = _load()
    before = len(data["positions"])
    data["positions"] = [p for p in data["positions"] if p["id"] != pos_id]
    _save(data)
    return len(data["positions"]) < before


def compute_pnl(pos: dict) -> float | None:
    """Return P&L in dollars. None if position still active."""
    if pos["status"] == "active":
        return None
    credit = pos["credit"] * pos["contracts"] * 100
    debit = (pos.get("close_debit") or 0) * pos["contracts"] * 100
    return round(credit - debit, 2)


def stats() -> dict:
    positions = all_positions()
    closed = [p for p in positions if p["status"] == "closed"]
    active = [p for p in positions if p["status"] == "active"]

    pnls = [compute_pnl(p) for p in closed if compute_pnl(p) is not None]
    total_pnl = round(sum(pnls), 2) if pnls else 0
    wins = [p for p in closed if p.get("outcome") == "expired"]
    win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_credit = round(sum(p["credit"] for p in positions) / len(positions), 2) if positions else 0

    # Per-ticker stats
    ticker_map: dict[str, dict] = {}
    for p in positions:
        t = p["ticker"]
        if t not in ticker_map:
            ticker_map[t] = {"ticker": t, "trades": 0, "pnl": 0, "wins": 0, "closed": 0}
        ticker_map[t]["trades"] += 1
        pnl = compute_pnl(p)
        if pnl is not None:
            ticker_map[t]["pnl"] += pnl
            ticker_map[t]["closed"] += 1
            if p.get("outcome") == "expired":
                ticker_map[t]["wins"] += 1

    ticker_stats = sorted(ticker_map.values(), key=lambda x: x["pnl"], reverse=True)

    return {
        "total_positions": len(positions),
        "active": len(active),
        "closed": len(closed),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_credit": avg_credit,
        "ticker_stats": ticker_stats,
    }
