"""Flask app — serverless-compatible for Vercel + full-featured for local."""
import os
import datetime
import threading
from flask import Flask, render_template, request, jsonify
from screener import run_screen
import positions as pos_db

app = Flask(__name__)

# ── In-memory scan state (local only; Vercel uses synchronous scan) ──────────
_scan_state = {"running": False, "progress": 0, "total": 0, "results": [], "error": None}
_scan_lock = threading.Lock()

IS_VERCEL = os.environ.get("VERCEL") == "1"


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", is_vercel=IS_VERCEL)


# ── Screener — synchronous on Vercel, background thread locally ───────────────
@app.route("/api/scan", methods=["POST"])
def start_scan():
    data = request.get_json(silent=True) or {}
    custom_tickers = data.get("tickers")
    rsi      = float(data.get("rsi", 40))
    pct_b    = float(data.get("pct_b", 35)) / 100
    iv       = float(data.get("iv", 20))
    ivr      = float(data.get("ivr", 40))
    price_min = float(data.get("price_min", 0))
    price_max = float(data.get("price_max", 0))

    tickers = [t.strip().upper() for t in custom_tickers if t.strip()] if custom_tickers else None

    if IS_VERCEL:
        # Synchronous — required for serverless (max ~15 tickers before timeout)
        if not tickers:
            return jsonify({"error": "On the hosted version please enter custom tickers (max 15). For full S&P 500 scanning run the app locally."}), 400
        if len(tickers) > 15:
            return jsonify({"error": f"Max 15 tickers on the hosted version ({len(tickers)} entered). Run locally for more."}), 400
        try:
            results = run_screen(
                tickers=tickers, rsi_threshold=rsi, pct_b_threshold=pct_b,
                iv_threshold=iv, ivr_threshold=ivr,
                price_min=price_min, price_max=price_max, max_workers=3,
            )
            return jsonify({"done": True, "results": results, "total": len(tickers)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Local: background thread with polling
        with _scan_lock:
            if _scan_state["running"]:
                return jsonify({"error": "Scan already running"}), 409
            _scan_state.update({"running": True, "progress": 0, "total": 0, "results": [], "error": None})

        def on_progress(done, total):
            with _scan_lock:
                _scan_state["progress"] = done
                _scan_state["total"] = total

        def do_scan():
            try:
                results = run_screen(
                    tickers=tickers, rsi_threshold=rsi, pct_b_threshold=pct_b,
                    iv_threshold=iv, ivr_threshold=ivr,
                    price_min=price_min, price_max=price_max,
                    max_workers=3, progress_callback=on_progress,
                )
                with _scan_lock:
                    _scan_state["results"] = results
                    _scan_state["running"] = False
            except Exception as e:
                with _scan_lock:
                    _scan_state["error"] = str(e)
                    _scan_state["running"] = False

        threading.Thread(target=do_scan, daemon=True).start()
        return jsonify({"status": "started"})


@app.route("/api/status")
def scan_status():
    if IS_VERCEL:
        return jsonify({"error": "Use /api/scan directly on hosted version"}), 400
    with _scan_lock:
        return jsonify(dict(_scan_state))


@app.route("/api/stop", methods=["POST"])
def stop_scan():
    with _scan_lock:
        _scan_state["running"] = False
    return jsonify({"status": "stopped"})


# ── Positions ─────────────────────────────────────────────────────────────────
@app.route("/api/positions", methods=["GET"])
def get_positions():
    all_pos = pos_db.all_positions()
    enriched = [dict(p, pnl=pos_db.compute_pnl(p)) for p in all_pos]
    return jsonify(enriched)


@app.route("/api/positions", methods=["POST"])
def create_position():
    data = request.get_json(silent=True) or {}
    try:
        pos = pos_db.add_position(
            ticker=data["ticker"],
            strike=data["strike"],
            credit=data["credit"],
            expiration=data["expiration"],
            open_date=data.get("open_date", str(datetime.date.today())),
            contracts=data.get("contracts", 1),
            notes=data.get("notes", ""),
            rolled_from_id=data.get("rolled_from_id"),
        )
        return jsonify(pos), 201
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400


@app.route("/api/positions/<pos_id>", methods=["PUT"])
def update_position(pos_id):
    data = request.get_json(silent=True) or {}
    pos = pos_db.update_position(
        pos_id=pos_id,
        outcome=data.get("outcome", "expired"),
        close_date=data.get("close_date", str(datetime.date.today())),
        close_debit=data.get("close_debit", 0.0),
        notes=data.get("notes", ""),
    )
    if not pos:
        return jsonify({"error": "Position not found"}), 404
    return jsonify(dict(pos, pnl=pos_db.compute_pnl(pos)))


@app.route("/api/positions/<pos_id>", methods=["DELETE"])
def delete_position(pos_id):
    if pos_db.delete_position(pos_id):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/positions/stats", methods=["GET"])
def position_stats():
    return jsonify(pos_db.stats())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port)
