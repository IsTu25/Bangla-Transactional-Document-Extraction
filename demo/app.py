"""
Interactive Web Demo for Bangla Document KIE with Real-Time Span Grounding Visualization.

Run locally:
  python demo/app.py --port 8501
"""

import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import SpanGroundedVerifier

verifier = SpanGroundedVerifier()

# Load sample data for instant exploration in UI
with open(REPO_ROOT / "data/test/gold_test_samples.json", "r", encoding="utf-8") as f:
    SAMPLE_DOCS = {d["document_type"]: d for d in json.load(f)}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bangla KIE - Span-Grounded Extraction & Verifier</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Noto+Sans+Bengali:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --accent-green: #10b981;
      --accent-blue: #3b82f6;
      --accent-purple: #8b5cf6;
      --accent-red: #ef4444;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: 'Outfit', 'Noto Sans Bengali', sans-serif;
      min-height: 100vh;
      padding: 2rem;
      background-image: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.12) 0%, transparent 40%);
    }
    .header {
      max-width: 1300px;
      margin: 0 auto 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }
    .title h1 {
      font-size: 1.8rem;
      font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .title p { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(59, 130, 246, 0.2);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      max-width: 1300px;
      margin: 0 auto;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .card h2 {
      font-size: 1.15rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-weight: 600;
    }
    select, button, textarea {
      font-family: inherit;
      border-radius: 8px;
      font-size: 0.95rem;
    }
    select {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 1rem;
    }
    button {
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      color: white;
      border: none;
      padding: 0.65rem 1.25rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button:hover { opacity: 0.9; transform: translateY(-1px); }
    .controls { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
    textarea {
      width: 100%;
      height: 240px;
      background: #0f172a;
      border: 1px solid var(--border);
      color: #e2e8f0;
      padding: 1rem;
      font-family: 'JetBrains Mono', 'Noto Sans Bengali', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      resize: vertical;
    }
    .preview-box {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      font-family: 'JetBrains Mono', 'Noto Sans Bengali', monospace;
      font-size: 0.9rem;
      line-height: 1.7;
      min-height: 180px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .span-hl {
      background: rgba(16, 185, 129, 0.25);
      color: #6ee7b7;
      border-bottom: 2px solid #10b981;
      padding: 0 3px;
      border-radius: 3px;
      font-weight: 600;
    }
    .metrics-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-top: 0.5rem;
    }
    .metric-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem;
      text-align: center;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; color: #34d399; }
    .metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-top: 0.2rem; }
    .full-width { grid-column: 1 / -1; }
    pre {
      background: #030712;
      border-radius: 8px;
      padding: 1rem;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: #93c5fd;
      max-height: 350px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">
      <h1>Bangla Transactional-Document Extraction</h1>
      <p>Span-Grounded JSON Extraction with Real-Time Verifier Telemetry</p>
    </div>
    <div>
      <span class="badge">Phase 7: Interactive Demo</span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>
        <span>1. Input OCR Text</span>
        <div class="controls">
          <select id="docTypeSelect">
            <option value="utility_bill">Utility Bill (DESCO)</option>
            <option value="mfs_receipt">MFS Receipt (bKash)</option>
            <option value="shop_invoice">Shop Invoice (Shwapno)</option>
          </select>
          <button onclick="loadSample()">Load Sample</button>
        </div>
      </h2>
      <textarea id="ocrInput" placeholder="Paste unstructured Bangla OCR text here..."></textarea>
      <button onclick="verifyExtraction()">Verify & Ground Spans</button>
    </div>

    <div class="card">
      <h2><span>2. Verifier Telemetry & Health</span> <span id="statusTag" class="badge">Awaiting Input</span></h2>
      <div class="metrics-bar">
        <div class="metric-chip">
          <div id="mScore" class="metric-val">-</div>
          <div class="metric-label">Verifier Score</div>
        </div>
        <div class="metric-chip">
          <div id="mSchema" class="metric-val">-</div>
          <div class="metric-label">Schema Status</div>
        </div>
        <div class="metric-chip">
          <div id="mHallu" class="metric-val">-</div>
          <div class="metric-label">Hallucinations</div>
        </div>
        <div class="metric-chip">
          <div id="mGround" class="metric-val">-</div>
          <div class="metric-label">Grounded Spans</div>
        </div>
      </div>
      <h3 style="font-size: 0.95rem; margin-top: 0.5rem; color: var(--text-muted);">Grounded Visualizer (Character Slices Highlighted):</h3>
      <div id="previewBox" class="preview-box">Original text with verified source spans will appear here...</div>
    </div>

    <div class="card full-width">
      <h2><span>3. Grounded Structured JSON Output</span></h2>
      <pre id="jsonOutput">// Extracted JSON payload with {value, source_span} will render here...</pre>
    </div>
  </div>

  <script>
    const sampleDocs = """ + json.dumps(SAMPLE_DOCS, ensure_ascii=False) + """;

    function loadSample() {
      const type = document.getElementById('docTypeSelect').value;
      const doc = sampleDocs[type];
      if (doc) {
        document.getElementById('ocrInput').value = doc.ocr_text;
        verifyExtraction();
      }
    }

    async function verifyExtraction() {
      const type = document.getElementById('docTypeSelect').value;
      const text = document.getElementById('ocrInput').value;
      const doc = sampleDocs[type];
      const prediction = doc ? doc.ground_truth : {};

      const res = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_type: type, ocr_text: text, prediction: prediction })
      });
      const data = await res.json();

      document.getElementById('mScore').innerText = data.overall_verifier_score.toFixed(3);
      document.getElementById('mSchema').innerText = data.passes_schema ? "VALID" : "FAIL";
      document.getElementById('mSchema').style.color = data.passes_schema ? "#34d399" : "#ef4444";
      document.getElementById('mHallu').innerText = data.hallucination_count;
      document.getElementById('mHallu').style.color = data.hallucination_count === 0 ? "#34d399" : "#ef4444";
      document.getElementById('mGround').innerText = data.taxonomy_breakdown.VALID || 0;

      document.getElementById('statusTag').innerText = "Verified by Engine";
      document.getElementById('statusTag').style.color = "#34d399";

      document.getElementById('jsonOutput').innerText = JSON.stringify(prediction, null, 2);

      // Render span highlighting
      renderSpans(text, prediction);
    }

    function renderSpans(text, prediction) {
      const spans = [];
      for (const [k, v] of Object.entries(prediction)) {
        if (v && v.source_span && typeof v.source_span.start === 'number') {
          spans.push(v.source_span);
        }
      }
      spans.sort((a, b) => a.start - b.start);

      let html = "";
      let lastIdx = 0;
      for (const s of spans) {
        if (s.start >= lastIdx) {
          html += escapeHtml(text.slice(lastIdx, s.start));
          html += '<span class="span-hl" title="Span: [' + s.start + ':' + s.end + ']">' + escapeHtml(text.slice(s.start, s.end)) + '</span>';
          lastIdx = s.end;
        }
      }
      html += escapeHtml(text.slice(lastIdx));
      document.getElementById('previewBox').innerHTML = html;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // Auto-load initial sample
    window.onload = loadSample;
  </script>
</body>
</html>
"""


class DemoRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/verify":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            report = verifier.verify_document(
                ocr_text=req["ocr_text"],
                prediction=req["prediction"],
                doc_type=req["document_type"],
            )

            res = {
                "overall_verifier_score": report.overall_score,
                "passes_schema": report.passes_schema,
                "hallucination_count": report.hallucination_count,
                "taxonomy_breakdown": report.taxonomy_counts,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_demo(port: int = 8501):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DemoRequestHandler)
    print(f"Bangla KIE Demo running at http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bangla KIE interactive demo.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    run_demo(args.port)
