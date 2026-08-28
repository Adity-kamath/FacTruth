"""Flask UI, JSON API, and optional Twilio WhatsApp webhook for facTruth."""

from __future__ import annotations

import os

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from core import run_fact_check


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

# ---------------------------------------------------------------------------
# Web pages
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/index_plain.html")
def home():
    return render_template(
        "index_plain.html"
    )





# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.post("/api/verify")
def api_verify():

    data = request.get_json(
        silent=True
    )

    if not isinstance(data, dict):

        return jsonify(
            error=(
                'Send a JSON object such as '
                '{"text": "your claim"}.'
            )
        ), 400

    text = str(
        data.get("text")
        or ""
    ).strip()

    if not text:

        return jsonify(
            error="No text provided."
        ), 400

    result = run_fact_check(text)

    return jsonify(result), (
        503
        if "error" in result
        else 200
    )

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_too_large(_error):

    return jsonify(
        error="Request is too large."
    ), 413

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=int(
            os.getenv(
                "PORT",
                "5000"
            )
        ),
        debug=(
            os.getenv(
                "FLASK_DEBUG"
            ) == "1"
        ),
    )