"""
Flask API for InvoiceGuard.

Endpoints:
  POST /analyze      — multipart upload of one invoice (PDF or image),
                       returns the full RiskReport as JSON.
  POST /analyze/batch — multiple files at once, returns a list of reports.
  GET  /healthz      — liveness probe.

The API is intentionally minimal — built for Postman testing, not for human
consumption (the UI was cut at the hackathon per the project notes). The
Mockups/ directory ships SVG visualizations of what the intended UI would
look like; the API output is structured so a future UI can render it without
any server changes.
"""

from __future__ import annotations

from flask import Flask, Response, jsonify, request

from app.detectors.vendor import InMemoryVendorDirectory
from app.orchestrator import Orchestrator


def create_app(orchestrator: Orchestrator | None = None) -> Flask:
    app = Flask(__name__)

    # Cap upload size at 20 MB — typical invoice PDFs are < 2 MB
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    # Default orchestrator: in-memory directory pre-seeded with a couple
    # of "known good" vendors so the demo flags unknowns correctly.
    default_directory = InMemoryVendorDirectory(known={
        "Acme Corporation": "12-3456780",
        "Globex LLC": "98-7654320",
        "Initech Inc.": "45-6789012",
    })
    orch = orchestrator or Orchestrator(vendor_directory=default_directory)

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"status": "ok"})

    @app.post("/analyze")
    def analyze() -> tuple[Response, int] | Response:
        if "file" not in request.files:
            return jsonify({"error": "missing 'file' in multipart form"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        try:
            data = f.read()
            report = orch.analyze(data, f.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"processing failed: {e}"}), 500

        return jsonify(report.to_dict())

    @app.post("/analyze/batch")
    def analyze_batch() -> tuple[Response, int] | Response:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "no files in 'files' field"}), 400

        out = []
        for f in files:
            if not f.filename:
                continue
            try:
                report = orch.analyze(f.read(), f.filename)
                out.append(report.to_dict())
            except Exception as e:  # noqa: BLE001
                out.append({"filename": f.filename, "error": str(e)})
        return jsonify({"reports": out})

    return app


def main() -> None:
    """Convenience entry point: `python -m app.api.server`."""
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
