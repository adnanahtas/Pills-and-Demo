"""
QicScan AI Platform — RF-DETR nano ONNX + Each Sorting (GS1 DataMatrix)
FastAPI + WebSocket: pill detection & each sorting with GS1 parsing.

Usage:
    python server.py
    Open http://localhost:8000 in your browser.
"""

import asyncio
import os
import json
import re
import time
import traceback
import uuid
import base64
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ─────────────────────────────── CONFIG ─────────────────────────────────────
MODEL_PATH      = "rfdetr-nano (2).onnx"   # pill detection model
BOX_MODEL_PATH  = "rfdetr-nano (3).onnx"   # box counting model
DEFAULT_CONF    = 0.75
CLASS_NAMES     = ["pill"]
FORCE_GPU       = True   # Set False to allow CPU fallback
DATA_FILE       = "sessions.json"        # pill detection sessions
EACH_DATA_FILE  = "each_sessions.json"   # each sorting sessions
BOX_DATA_FILE   = "box_sessions.json"    # box counting sessions

IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─────────────────────────────── LOAD MODEL ─────────────────────────────────
print("=" * 60)
print("  Pill Detection AI — Loading RF-DETR nano ONNX model …")
print("=" * 60)

_available = ort.get_available_providers()
print(f"  Available providers : {_available}")

_sess_opts = ort.SessionOptions()
_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_sess_opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
_sess_opts.intra_op_num_threads     = 1

_cuda_opts = {
    "device_id":                  0,
    "arena_extend_strategy":      "kNextPowerOfTwo",
    "gpu_mem_limit":               4 * 1024 ** 3,
    "cudnn_conv_algo_search":      "EXHAUSTIVE",
    "do_copy_in_default_stream":   True,
}

# Provider priority: CUDA (if installed) → DirectML (Windows GPU, no CUDA needed) → CPU
_providers = [
    ("CUDAExecutionProvider", _cuda_opts),
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]
session    = ort.InferenceSession(MODEL_PATH, sess_options=_sess_opts, providers=_providers)
_active    = session.get_providers()
print(f"  Active providers    : {_active}")

_GPU_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider"]
_on_gpu = any(p in _active for p in _GPU_PROVIDERS)

if FORCE_GPU and not _on_gpu:
    raise RuntimeError(
        "\n[ERROR] No GPU provider is active!\n"
        "  Option A (DirectML, no CUDA needed): pip install onnxruntime-directml\n"
        "  Option B (CUDA): install CUDA 12.x + cuDNN 9.x, then pip install onnxruntime-gpu\n"
        "  Set FORCE_GPU=False in server.py to run on CPU instead."
    )

_active_gpu = next((p for p in _GPU_PROVIDERS if p in _active), "CPU")
print(f"  ✓ Running on : {_active_gpu}")

_inputs      = session.get_inputs()
_outputs     = session.get_outputs()
INPUT_NAME   = _inputs[0].name
INPUT_SHAPE  = _inputs[0].shape
OUTPUT_NAMES = [o.name for o in _outputs]

print(f"\n  Input  : '{INPUT_NAME}'  shape = {INPUT_SHAPE}")
for o in _outputs:
    print(f"  Output : '{o.name}'  shape = {o.shape}")


def _static_dim(d: object, fallback: int) -> int:
    return d if (isinstance(d, int) and d > 0) else fallback


MODEL_H = _static_dim(INPUT_SHAPE[2], 640)
MODEL_W = _static_dim(INPUT_SHAPE[3], 640)
print(f"\n  Inference size : {MODEL_W} × {MODEL_H}")
print("  Model loaded ✓")
print("=" * 60)

# ── Box counting model (loaded after _static_dim is defined) ─────────────────
print("=" * 60)
print("  Box Counting AI — Loading model …")
print("=" * 60)
box_session      = ort.InferenceSession(BOX_MODEL_PATH, sess_options=_sess_opts, providers=_providers)
_box_inputs      = box_session.get_inputs()
_box_outputs     = box_session.get_outputs()
BOX_INPUT_NAME   = _box_inputs[0].name
BOX_INPUT_SHAPE  = _box_inputs[0].shape
BOX_OUTPUT_NAMES = [o.name for o in _box_outputs]
BOX_H = _static_dim(BOX_INPUT_SHAPE[2], 640) if len(BOX_INPUT_SHAPE) > 2 else 640
BOX_W = _static_dim(BOX_INPUT_SHAPE[3], 640) if len(BOX_INPUT_SHAPE) > 3 else 640
_box_probe = box_session.run(None, {BOX_INPUT_NAME: np.zeros((1,3,BOX_H,BOX_W), dtype=np.float32)})
_box_logits = next((o for o in _box_probe if o.ndim==3 and o.shape[-1]!=4), _box_probe[0])
_box_num_classes = _box_logits.shape[-1]
_BOX_LABEL_MAP = {0: "Medicine_box", 1: "Medicine_bottle"}
BOX_CLASS_NAMES = [_BOX_LABEL_MAP.get(i, f"item_{i}") for i in range(_box_num_classes)]
print(f"  Box model loaded ✓  input={BOX_INPUT_SHAPE}  classes={_box_num_classes}")
print("=" * 60)


# ─────────────────────────────── PRE / POST ──────────────────────────────────
def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Resize + ImageNet-normalise → NCHW float32 tensor."""
    img = cv2.resize(bgr, (MODEL_W, MODEL_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def postprocess(raw: list, orig_h: int, orig_w: int, conf: float) -> list:
    """
    RF-DETR ONNX typical outputs (Roboflow export):
        output[?]  logits    [1, N, num_classes]  ← last dim != 4
        output[?]  pred_boxes [1, N, 4]           ← last dim == 4
    Boxes are (cx, cy, w, h) normalised [0-1].
    """
    logits = pred_boxes = None

    for name, out in zip(OUTPUT_NAMES, raw):
        if out.ndim == 3:
            if out.shape[-1] == 4:
                pred_boxes = out[0]     # (N, 4)
            else:
                logits = out[0]         # (N, C)

    if logits is None or pred_boxes is None:
        print(f"[WARN] Unexpected model outputs: "
              f"{[(n, r.shape) for n, r in zip(OUTPUT_NAMES, raw)]}")
        return []

    # Sigmoid → confidence per class
    probs  = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))   # (N, C)
    scores = probs.max(axis=-1)                                   # (N,)
    cls_id = probs.argmax(axis=-1)                                # (N,)

    mask = scores >= conf
    if not mask.any():
        return []

    detections = []
    for s, c, (cx, cy, bw, bh) in zip(scores[mask], cls_id[mask], pred_boxes[mask]):
        x1 = max(0,       int((cx - bw / 2) * orig_w))
        y1 = max(0,       int((cy - bh / 2) * orig_h))
        x2 = min(orig_w,  int((cx + bw / 2) * orig_w))
        y2 = min(orig_h,  int((cy + bh / 2) * orig_h))
        label = CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"cls_{c}"
        detections.append({
            "box":   [x1, y1, x2, y2],
            "score": round(float(s), 4),
            "label": label,
        })

    return detections


_executor = ThreadPoolExecutor(max_workers=4)


async def run_inference(inp: np.ndarray):
    """Run pill ONNX session in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: session.run(None, {INPUT_NAME: inp}))


def preprocess_box(bgr: np.ndarray) -> np.ndarray:
    """Resize + ImageNet-normalise for box model → NCHW float32 tensor."""
    img = cv2.resize(bgr, (BOX_W, BOX_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def postprocess_box(raw: list, orig_h: int, orig_w: int, conf: float) -> list:
    """Same RF-DETR postprocess logic for the box model."""
    logits = pred_boxes = None
    for name, out in zip(BOX_OUTPUT_NAMES, raw):
        if out.ndim == 3:
            if out.shape[-1] == 4:
                pred_boxes = out[0]
            else:
                logits = out[0]
    if logits is None or pred_boxes is None:
        return []
    probs  = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    scores = probs.max(axis=-1)
    cls_id = probs.argmax(axis=-1)
    mask   = scores >= conf
    if not mask.any():
        return []
    detections = []
    for s, c, (cx, cy, bw, bh) in zip(scores[mask], cls_id[mask], pred_boxes[mask]):
        x1 = max(0,       int((cx - bw/2) * orig_w))
        y1 = max(0,       int((cy - bh/2) * orig_h))
        x2 = min(orig_w,  int((cx + bw/2) * orig_w))
        y2 = min(orig_h,  int((cy + bh/2) * orig_h))
        label = BOX_CLASS_NAMES[int(c)] if int(c) < len(BOX_CLASS_NAMES) else f"item_{c}"
        detections.append({"box": [x1,y1,x2,y2], "score": round(float(s),4), "label": label})
    return detections


def annotate_detections(img_bytes: bytes, dets: list, conf: float) -> str:
    """Draw bounding boxes on image, return base64 JPEG."""
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return ""
    h, w = frame.shape[:2]
    # group by label for consistent colours
    label_colors: dict = {}
    palette = [
        (230,57,70),(33,150,243),(76,175,80),(255,152,0),
        (156,39,176),(0,188,212),(255,87,34),(63,81,181),
    ]
    for det in dets:
        lbl = det["label"]
        if lbl not in label_colors:
            label_colors[lbl] = palette[len(label_colors) % len(palette)]
        r,g,b = label_colors[lbl]
        bgr   = (b,g,r)
        x1,y1,x2,y2 = det["box"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, max(2, w//300))
        # corner accents
        cl = max(6, int(min(x2-x1,y2-y1)*0.15))
        ct = max(2, w//400)
        for (px,py),(dx,dy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),((x2,y2),(-1,-1)),((x1,y2),(1,-1))]:
            cv2.line(frame,(px,py),(px+dx*cl,py),(255,255,255),ct,cv2.LINE_AA)
            cv2.line(frame,(px,py),(px,py+dy*cl),(255,255,255),ct,cv2.LINE_AA)
        # label chip
        pct  = int(det["score"]*100)
        text = f"{lbl} {pct}%"
        fs   = max(0.4, min(w,h)/1200)
        tk   = max(1, int(fs*1.5))
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, tk)
        pad = 5
        lx, ly = x1, y1-th-pad*2 if y1 > th+pad*2 else y1+2
        cv2.rectangle(frame,(lx,ly),(lx+tw+pad*2,ly+th+pad*2),bgr,-1)
        cv2.putText(frame,text,(lx+pad,ly+th+pad),cv2.FONT_HERSHEY_DUPLEX,fs,(255,255,255),tk,cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────── GS1 PARSER ─────────────────────────────────
GS1_SEPARATOR = chr(0x1D)   # ASCII GS (standard)
# FNC1 variants seen in real DataMatrix / pylibdmtx output
ALT_SEPARATORS = ["\x1D", "\x1e", "\x04", "\x00", "~1", "\\F", "<GS>"]

EXPIRY_COLORS = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D",
    "#6A4C93", "#F72585", "#4361EE", "#3A86FF",
    "#8338EC", "#06D6A0", "#FFB703", "#FB5607",
]


def _normalize_raw(raw: str) -> str:
    """Normalize various GS / FNC1 separator encodings to ASCII GS 0x1D."""
    # pylibdmtx sometimes emits raw bytes for FNC1; cover common representations
    for sep in ALT_SEPARATORS:
        raw = raw.replace(sep, GS1_SEPARATOR)
    # Also strip any leading FNC1 / AIM symbology identifier e.g. ]d2 ]C1 ]Q3
    return raw


# Variable-length AIs that MUST be terminated by GS separator when not last
VAR_LENGTH_AIS = {"10", "21", "30", "310", "00"}

# Fixed-length AI table: AI -> (field_name, exact_length)
GS1_FIXED = {
    "00": ("sscc",         18),
    "01": ("gtin",         14),
    "02": ("content_gtin", 14),
    "11": ("prod_date",     6),
    "17": ("expiry",        6),
}
# Variable-length AI table: AI -> (field_name, max_length)
GS1_VARIABLE = {
    "10": ("lot",    20),
    "21": ("serial", 20),
    "30": ("qty",     8),
}


def parse_gs1(raw: str) -> dict:
    """
    Parse a GS1-128 / GS1 DataMatrix barcode string.
    Format example: 01<GTIN14>21<SERIAL><GS>17<YYMMDD>10<LOT>
    <GS> = ASCII 0x1D (or FNC1 variants normalised to it).
    Variable-length AIs are terminated by <GS> or end-of-string.
    """
    original_raw = raw
    raw = _normalize_raw(raw)

    result = {
        "gtin": "", "lot": "", "serial": "", "expiry": "",
        "expiry_display": "", "expiry_raw": "",
        "extra": {}, "raw": original_raw, "valid": False
    }

    # Strip AIM symbology identifier e.g. ]d2  ]C1  ]Q3  ]e0
    s = raw
    if s.startswith("]") and len(s) >= 3:
        s = s[3:]       # ']' + type-char + version-char
    elif s.startswith("]"):
        s = s[1:]

    # Strip leading GS if any
    s = s.lstrip(GS1_SEPARATOR)

    pos = 0
    n   = len(s)

    while pos < n:
        # Skip any GS separators between AIs
        while pos < n and s[pos] == GS1_SEPARATOR:
            pos += 1
        if pos >= n:
            break

        matched = False
        # Try AI lengths: 4, 3, 2  (longest first to avoid false matches)
        for ai_len in (4, 3, 2):
            if pos + ai_len > n:
                continue
            ai = s[pos:pos + ai_len]

            if ai in GS1_FIXED:
                name, length = GS1_FIXED[ai]
                pos += ai_len
                val = s[pos:pos + length]
                pos += length
                # Store
                if name == "gtin":       result["gtin"] = val
                elif name == "expiry":
                    result["expiry_raw"]     = val
                    result["expiry"]         = _format_expiry(val)
                    result["expiry_display"] = _expiry_display(val)
                else:
                    result["extra"][ai] = val
                matched = True
                break

            if ai in GS1_VARIABLE:
                name, _max = GS1_VARIABLE[ai]
                pos += ai_len
                # Read until GS separator or end-of-string
                gs_end = s.find(GS1_SEPARATOR, pos)
                candidate = s[pos:gs_end] if gs_end != -1 else s[pos:]
                # Even without a GS separator, stop when a known AI appears
                # e.g. serial "05747877...17262312 10AB12" — 17 and 10 are AIs
                cut = len(candidate)
                for look in range(1, len(candidate)):
                    found_cut = False
                    for al in (4, 3, 2):
                        if look + al > len(candidate):
                            continue
                        maybe_ai = candidate[look:look + al]
                        # Only cut on fixed-length AIs — variable AIs (10,21,30)
                        # cannot be validated and cause false cuts inside serial digits
                        if maybe_ai not in GS1_FIXED:
                            continue
                        _, flen = GS1_FIXED[maybe_ai]
                        after = candidate[look + al:]
                        if len(after) < flen:
                            continue   # not enough data → not a real AI here
                        # Validate date AIs: month must be 01-12
                        if maybe_ai in ("17", "11"):
                            mm = after[2:4]
                            if not mm.isdigit() or not (1 <= int(mm) <= 12):
                                continue
                        # Validate year is digits
                        yy = after[:2]
                        if not yy.isdigit():
                            continue
                        cut = look
                        found_cut = True
                        break
                    if found_cut:
                        break
                val = candidate[:cut]
                # advance pos: if GS separator was found and cut == full candidate, consume the GS too
                if gs_end != -1 and cut == len(candidate):
                    pos = gs_end + 1
                else:
                    pos = pos + cut
                # Store
                if name == "lot":      result["lot"]    = val
                elif name == "serial": result["serial"] = val
                else:                  result["extra"][ai] = val
                matched = True
                break

        if not matched:
            pos += 1   # Unknown AI — advance one char

    result["valid"] = bool(result["gtin"] or result["serial"] or result["lot"])
    return result


def _format_expiry(yymmdd: str) -> str:
    """Convert YYMMDD → YYYY-MM-DD ISO string."""
    if len(yymmdd) < 6:
        return yymmdd
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    year = int(yy)
    year += 2000 if year <= 49 else 1900
    if dd == "00":
        dd = "01"
    return f"{year:04d}-{mm}-{dd}"


def _expiry_display(yymmdd: str) -> str:
    """Convert YYMMDD → human readable MM/YYYY."""
    if len(yymmdd) < 6:
        return yymmdd
    return f"{yymmdd[2:4]}/{('20' if int(yymmdd[:2])<=49 else '19')+yymmdd[:2]}"


def _dmtx_scan_gray(gray, scale=1.0, timeout=800):
    """Run pylibdmtx on a grayscale image, optionally upscaled. Returns raw results."""
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    if scale != 1.0:
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return dmtx_decode(gray, timeout=timeout), scale


def scan_barcodes_from_image(img_bytes: bytes) -> list:
    """
    Decode DataMatrix / QR / other barcodes from image bytes.
    Primary: pylibdmtx (DataMatrix) + pyzbar (QR/Code128).
    Fallback: OpenCV QR detector.
    Returns list of dicts: {data, polygon, rect, type}
    """
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    img_h = frame.shape[0]
    results = []
    seen_data = set()   # deduplicate by raw data string

    def _add(data, polygon, rect, sym_type):
        if data and data not in seen_data:
            seen_data.add(data)
            results.append({"data": data, "polygon": polygon, "rect": rect, "type": sym_type})

    # ── pylibdmtx — DataMatrix (primary for GS1 DataMatrix barcodes) ────
    try:
        from pylibdmtx.pylibdmtx import decode as dmtx_decode  # noqa: F401
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing variants to maximize detection rate
        candidates = [gray]
        # Upscale small images for better detection
        h, w = gray.shape[:2]
        if max(h, w) < 1200:
            candidates.append(cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        candidates.append(clahe.apply(gray))
        # Sharpened
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        candidates.append(cv2.filter2D(gray, -1, kernel))

        for i, img_variant in enumerate(candidates):
            scale = img_variant.shape[0] / img_h   # ratio to map coords back
            try:
                dm_results = dmtx_decode(img_variant, timeout=800)
            except Exception:
                continue
            for sym in dm_results:
                try:
                    data = sym.data.decode("utf-8", errors="replace")
                except Exception:
                    data = str(sym.data)
                if data in seen_data:
                    continue
                # pylibdmtx rect origin is bottom-left; convert to top-left image coords
                x  = int(sym.rect.left  / scale)
                w2 = int(sym.rect.width  / scale)
                h2 = int(sym.rect.height / scale)
                y_bl = int(sym.rect.top / scale)
                top  = img_h - y_bl - h2
                polygon = [(x, top), (x+w2, top), (x+w2, top+h2), (x, top+h2)]
                _add(data, polygon,
                     {"left": x, "top": top, "width": w2, "height": h2},
                     "DATAMATRIX")
            if results:
                break   # stop trying variants once we have detections
    except Exception as _dmtx_err:
        print(f"[WARN] pylibdmtx error — DataMatrix scanning unavailable: {_dmtx_err}")

    # ── pyzbar (QR, Code128, EAN13 — does NOT support DataMatrix) ───────
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from pyzbar.pyzbar import ZBarSymbol
        decoded = pyzbar_decode(frame, symbols=[
            ZBarSymbol.QRCODE, ZBarSymbol.CODE128, ZBarSymbol.EAN13,
        ])
        for sym in decoded:
            try:
                data = sym.data.decode("utf-8", errors="replace")
            except Exception:
                data = str(sym.data)
            polygon = [(p.x, p.y) for p in sym.polygon]
            rect    = sym.rect
            _add(data, polygon,
                 {"left": rect.left, "top": rect.top,
                  "width": rect.width, "height": rect.height},
                 sym.type.name)
    except ImportError:
        pass

    # ── OpenCV QR fallback (only if nothing found above) ────────────────
    if not results:
        try:
            qr   = cv2.QRCodeDetector()
            _res = qr.detectAndDecodeMulti(frame)
            if len(_res) == 4:
                _, data_list, points, *_ = _res
            else:
                data_list, points, *_ = _res
            if data_list and points is not None:
                for d, pts in zip(data_list, points):
                    if d:
                        pts_int = pts.astype(int).tolist()
                        xs = [p[0] for p in pts_int]
                        ys = [p[1] for p in pts_int]
                        _add(d, [(p[0], p[1]) for p in pts_int],
                             {"left": min(xs), "top": min(ys),
                              "width": max(xs)-min(xs), "height": max(ys)-min(ys)},
                             "QR")
        except Exception:
            pass

    return results


def annotate_image(img_bytes: bytes, items: list, color_map: dict) -> str:
    """
    Draw colour-coded polygon annotations on image.
    Shows only expiry date + ✓ tick inside each barcode box.
    Returns base64-encoded JPEG.
    """
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return ""

    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    # Scale text relative to image size so it's readable on any resolution
    base_scale = max(0.5, min(w, h) / 1000)

    for item in items:
        expiry  = item.get("expiry", "")
        color_h = color_map.get(expiry, "#888888")
        r = int(color_h[1:3], 16)
        g = int(color_h[3:5], 16)
        b = int(color_h[5:7], 16)
        bgr = (b, g, r)

        poly = item.get("polygon", [])
        if not poly:
            rect = item.get("rect", {})
            lft  = rect.get("left", 0)
            top  = rect.get("top",  0)
            rgt  = lft + rect.get("width",  50)
            bot  = top + rect.get("height", 50)
            poly = [(lft, top), (rgt, top), (rgt, bot), (lft, bot)]

        pts = np.array(poly, dtype=np.int32)
        xs  = [p[0] for p in poly]
        ys  = [p[1] for p in poly]
        box_w = max(xs) - min(xs)
        box_h = max(ys) - min(ys)

        # ── Filled semi-transparent overlay ──
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # ── Thick border ──
        cv2.polylines(frame, [pts], True, bgr, max(2, w // 300))

        # ── Corner tick marks (white L-shapes at each corner) ──
        corner_len = max(8, int(min(box_w, box_h) * 0.18))
        cthick     = max(2, w // 400)
        corners = list(zip(
            [min(xs), max(xs), max(xs), min(xs)],
            [min(ys), min(ys), max(ys), max(ys)]
        ))
        dirs = [(1,1), (-1,1), (-1,-1), (1,-1)]
        for (cx2, cy2), (dx, dy) in zip(corners, dirs):
            cv2.line(frame, (cx2, cy2), (cx2 + dx*corner_len, cy2), (255,255,255), cthick, cv2.LINE_AA)
            cv2.line(frame, (cx2, cy2), (cx2, cy2 + dy*corner_len), (255,255,255), cthick, cv2.LINE_AA)

        # ── Centre: expiry date label ──
        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))

        disp   = item.get("expiry_display", "") or expiry or "No Exp"
        tick   = "✓ " + disp
        tscale = max(0.4, min(base_scale, box_w / max(cv2.getTextSize(tick, font, base_scale, 1)[0][0], 1) * 0.85))
        thick  = max(1, int(tscale * 1.6))

        (tw, th), baseline = cv2.getTextSize(tick, font, tscale, thick)
        tx = cx - tw // 2
        ty = cy + th // 2

        # Pill-shaped background behind text
        pad = 6
        cv2.rectangle(frame,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + baseline + pad),
                      bgr, -1)
        cv2.rectangle(frame,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + baseline + pad),
                      (255,255,255), max(1, cthick-1))

        # White text
        cv2.putText(frame, tick, (tx+1, ty+1), font, tscale, (0,0,0),   thick+1, cv2.LINE_AA)
        cv2.putText(frame, tick, (tx,   ty),   font, tscale, (255,255,255), thick, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────── BOX SESSION STORE ──────────────────────────
def _load_box_sessions() -> list:
    if os.path.exists(BOX_DATA_FILE):
        try:
            with open(BOX_DATA_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_box_sessions(sessions: list) -> None:
    with open(BOX_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


_box_sessions: list = _load_box_sessions()
print(f"  Box sessions     : {len(_box_sessions)} record(s) from {BOX_DATA_FILE}")


# ─────────────────────────────── SESSION STORE ───────────────────────────────
def _load_sessions() -> list:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_sessions(sessions: list) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


_sessions: list = _load_sessions()
print(f"  Sessions loaded  : {len(_sessions)} record(s) from {DATA_FILE}")


# ─────────────────────────── EACH SESSION STORE ──────────────────────────────
def _load_each_sessions() -> list:
    if os.path.exists(EACH_DATA_FILE):
        try:
            with open(EACH_DATA_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_each_sessions(sessions: list) -> None:
    with open(EACH_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


_each_sessions: list = _load_each_sessions()
print(f"  Each sessions    : {len(_each_sessions)} record(s) from {EACH_DATA_FILE}")


# ─────────────────────────────── FASTAPI ────────────────────────────────────
class SaveRequest(BaseModel):
    count:         int
    note:          str   = ""
    conf:          float
    fps:           float
    peak:          int   = 0
    annotated_img: str   = ""   # base64 JPEG, optional


class EachScanRequest(BaseModel):
    image_b64:  str          # base64-encoded JPEG
    note:       str  = ""


class EachSaveRequest(BaseModel):
    scan_id:    str
    note:       str  = ""


class BoxScanRequest(BaseModel):
    image_b64:  str
    conf:       float = 0.40


class BoxSaveRequest(BaseModel):
    scan_id:    str
    note:       str  = ""


class PillScanRequest(BaseModel):
    image_b64:  str
    conf:       float = 0.40


app = FastAPI(title="Pill Detection Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    with open("static/index.html", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/logo.png")
async def logo():
    """Serve the QicScan wordmark from the project root."""
    for candidate in ["QicScan_BarcodeQSWordmark-1024x258.png",
                      "static/QicScan_BarcodeQSWordmark-1024x258.png"]:
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type="image/png")
    return JSONResponse({"error": "logo not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH,
            "input_size": [MODEL_W, MODEL_H], "classes": CLASS_NAMES}


@app.get("/api/sessions")
async def get_sessions():
    return JSONResponse(_sessions)


@app.post("/api/save")
async def save_session(req: SaveRequest):
    record = {
        "id":            str(uuid.uuid4()),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "count":         req.count,
        "peak":          req.peak,
        "note":          req.note.strip(),
        "conf":          round(req.conf, 2),
        "fps":           round(req.fps, 1),
        "annotated_img": req.annotated_img,   # empty string when from live feed
    }
    _sessions.insert(0, record)          # newest first
    if len(_sessions) > 500:             # cap at 500 records
        _sessions.pop()
    _save_sessions(_sessions)
    print(f"[SAVE] count={req.count} peak={req.peak} note='{req.note}' annotated={'yes' if req.annotated_img else 'no'}")
    return JSONResponse(record, status_code=201)


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    global _sessions
    before = len(_sessions)
    _sessions = [s for s in _sessions if s["id"] != sid]
    if len(_sessions) == before:
        return JSONResponse({"error": "not found"}, status_code=404)
    _save_sessions(_sessions)
    return JSONResponse({"deleted": sid})


@app.delete("/api/sessions")
async def clear_all_sessions():
    global _sessions
    _sessions = []
    _save_sessions(_sessions)
    return JSONResponse({"deleted": "all"})


# ─────────────────────────── PILL IMAGE SCAN ────────────────────────────────
@app.post("/api/pill/scan")
async def pill_scan(req: PillScanRequest):
    """Run pill detection on a single uploaded/captured image."""
    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)

    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    loop = asyncio.get_running_loop()
    h, w = frame.shape[:2]
    inp  = preprocess(frame)
    raw  = await loop.run_in_executor(_executor, lambda: session.run(None, {INPUT_NAME: inp}))
    dets = postprocess(raw, h, w, req.conf)

    # build per-label counts
    counts: dict = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    annotated_b64 = await loop.run_in_executor(
        _executor, annotate_detections, img_bytes, dets, req.conf
    )
    return JSONResponse({
        "total":        len(dets),
        "counts":       counts,
        "detections":   dets,
        "annotated_img": annotated_b64,
    })


# ──────────────────────── EACH SORTING ROUTES ────────────────────────────────

# In-memory pending scans (not yet saved) keyed by scan_id
_pending_each: dict = {}


@app.post("/api/each/scan")
async def each_scan(req: EachScanRequest):
    """
    Receive a base64 JPEG, run barcode detection, parse GS1,
    annotate the image, and return full structured results.
    """
    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)

    loop = asyncio.get_running_loop()
    raw_barcodes = await loop.run_in_executor(
        _executor, scan_barcodes_from_image, img_bytes
    )

    # Parse each barcode — print raw data for debugging
    parsed_items = []
    print(f"\n{'='*60}")
    print(f"  EACH SCAN — {len(raw_barcodes)} barcode(s) detected")
    print(f"{'='*60}")
    for i, bc in enumerate(raw_barcodes):
        raw_data = bc["data"]
        raw_hex  = raw_data.encode("utf-8", errors="replace").hex()
        gs1 = parse_gs1(raw_data)
        print(f"  [{i+1}] type    : {bc.get('type','')}")
        print(f"       raw     : {repr(raw_data)}")
        print(f"       hex     : {raw_hex}")
        print(f"       gtin    : {gs1['gtin']}")
        print(f"       serial  : {gs1['serial']}")
        print(f"       lot     : {gs1['lot']}")
        print(f"       expiry  : {gs1['expiry']} ({gs1['expiry_display']})")
        print(f"       valid   : {gs1['valid']}")
        print()
        item = {
            **gs1,
            "polygon": bc.get("polygon", []),
            "rect":    bc.get("rect",    {}),
            "type":    bc.get("type",   ""),
        }
        parsed_items.append(item)
    print(f"{'='*60}\n")

    # Assign colour per unique expiry group
    expiry_dates = list(dict.fromkeys(
        i.get("expiry", "") for i in parsed_items
    ))
    color_map = {
        exp: EXPIRY_COLORS[idx % len(EXPIRY_COLORS)]
        for idx, exp in enumerate(expiry_dates)
    }

    # Annotate image
    annotated_b64 = await loop.run_in_executor(
        _executor, annotate_image, img_bytes, parsed_items, color_map
    )

    # Build expiry groups summary
    groups: dict = {}
    for item in parsed_items:
        exp = item.get("expiry", "") or "Unknown"
        if exp not in groups:
            groups[exp] = {
                "expiry":         exp,
                "expiry_display": item.get("expiry_display", ""),
                "color":          color_map.get(item.get("expiry",""), "#888"),
                "count":          0,
                "items":          [],
            }
        groups[exp]["count"] += 1
        groups[exp]["items"].append({
            "gtin":   item.get("gtin",   ""),
            "serial": item.get("serial", ""),
            "lot":    item.get("lot",    ""),
            "expiry": exp,
            "expiry_display": item.get("expiry_display", ""),
            "raw":    item.get("raw",    ""),
        })

    scan_id = str(uuid.uuid4())
    result  = {
        "scan_id":       scan_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "total":         len(parsed_items),
        "groups":        list(groups.values()),
        "color_map":     color_map,
        "annotated_img": annotated_b64,
        "note":          req.note.strip(),
    }

    _pending_each[scan_id] = result
    # trim pending cache
    if len(_pending_each) > 50:
        oldest = next(iter(_pending_each))
        _pending_each.pop(oldest, None)

    return JSONResponse(result)


@app.post("/api/each/save")
async def each_save(req: EachSaveRequest):
    """Persist a pending scan to disk."""
    global _each_sessions
    data = _pending_each.get(req.scan_id)
    if not data:
        return JSONResponse({"error": "scan_id not found"}, status_code=404)

    record = dict(data)          # keep annotated_img for history thumbnails
    record["note"] = req.note.strip() or record.get("note", "")
    record["id"]   = data["scan_id"]

    _each_sessions.insert(0, record)
    if len(_each_sessions) > 200:
        _each_sessions.pop()
    _save_each_sessions(_each_sessions)
    print(f"[EACH SAVE] total={record['total']} note='{record['note']}'")
    return JSONResponse(record, status_code=201)


@app.get("/api/each/sessions")
async def get_each_sessions():
    return JSONResponse(_each_sessions)


@app.delete("/api/each/sessions/{sid}")
async def delete_each_session(sid: str):
    global _each_sessions
    before = len(_each_sessions)
    _each_sessions = [s for s in _each_sessions if s.get("id") != sid]
    if len(_each_sessions) == before:
        return JSONResponse({"error": "not found"}, status_code=404)
    _save_each_sessions(_each_sessions)
    return JSONResponse({"deleted": sid})


@app.delete("/api/each/sessions")
async def clear_each_sessions():
    global _each_sessions
    _each_sessions = []
    _save_each_sessions(_each_sessions)
    return JSONResponse({"deleted": "all"})


# ──────────────────────── BOX COUNTING ROUTES ────────────────────────────────
_pending_box: dict = {}


@app.post("/api/box/scan")
async def box_scan(req: BoxScanRequest):
    """Run box/item counting on a single image."""
    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)

    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    loop = asyncio.get_running_loop()
    h, w = frame.shape[:2]
    inp  = preprocess_box(frame)
    raw  = await loop.run_in_executor(_executor, lambda: box_session.run(None, {BOX_INPUT_NAME: inp}))
    dets = postprocess_box(raw, h, w, req.conf)

    counts: dict = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    annotated_b64 = await loop.run_in_executor(
        _executor, annotate_detections, img_bytes, dets, req.conf
    )

    scan_id = str(uuid.uuid4())
    result  = {
        "scan_id":        scan_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "total":          len(dets),
        "counts":         counts,
        "detections":     dets,
        "annotated_img":  annotated_b64,
    }
    _pending_box[scan_id] = result
    if len(_pending_box) > 50:
        _pending_box.pop(next(iter(_pending_box)), None)
    return JSONResponse(result)


@app.post("/api/box/save")
async def box_save(req: BoxSaveRequest):
    global _box_sessions
    data = _pending_box.get(req.scan_id)
    if not data:
        return JSONResponse({"error": "scan_id not found"}, status_code=404)
    record = dict(data)          # keep annotated_img for history thumbnails
    record["note"] = req.note.strip()
    record["id"]   = data["scan_id"]
    _box_sessions.insert(0, record)
    if len(_box_sessions) > 200:
        _box_sessions.pop()
    _save_box_sessions(_box_sessions)
    return JSONResponse(record, status_code=201)


@app.get("/api/box/sessions")
async def get_box_sessions():
    return JSONResponse(_box_sessions)


@app.delete("/api/box/sessions/{sid}")
async def delete_box_session(sid: str):
    global _box_sessions
    before = len(_box_sessions)
    _box_sessions = [s for s in _box_sessions if s.get("id") != sid]
    if len(_box_sessions) == before:
        return JSONResponse({"error": "not found"}, status_code=404)
    _save_box_sessions(_box_sessions)
    return JSONResponse({"deleted": sid})


@app.delete("/api/box/sessions")
async def clear_box_sessions():
    global _box_sessions
    _box_sessions = []
    _save_box_sessions(_box_sessions)
    return JSONResponse({"deleted": "all"})


@app.websocket("/ws")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")

    conf        = DEFAULT_CONF
    frame_queue = asyncio.Queue(maxsize=2)   # hold at most 2 frames; always process freshest
    frame_cnt   = 0
    peak_count  = 0
    t_start     = time.perf_counter()
    alive       = True

    # ── Receiver: pull messages off the wire as fast as they arrive ────────
    async def receiver():
        nonlocal conf, alive
        try:
            while alive:
                msg = await ws.receive()

                if "text" in msg and msg["text"]:
                    try:
                        cfg = json.loads(msg["text"])
                        if "conf" in cfg:
                            conf = max(0.01, min(0.99, float(cfg["conf"])))
                            print(f"[WS] conf → {conf:.2f}")
                    except Exception:
                        pass
                    continue

                data = msg.get("bytes")
                if data:
                    # drop oldest frame so queue never blocks on a slow GPU
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS][recv] {e}")
        finally:
            alive = False
            try:
                frame_queue.put_nowait(None)   # unblock processor
            except Exception:
                pass

    # ── Processor: infer on each frame and send result back ───────────────
    async def processor():
        nonlocal frame_cnt, peak_count, alive
        try:
            while True:
                data = await frame_queue.get()
                if data is None:
                    break

                arr   = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                inp  = preprocess(frame)
                raw  = await run_inference(inp)
                dets = postprocess(raw, h, w, conf)

                frame_cnt += 1
                cnt        = len(dets)
                peak_count = max(peak_count, cnt)
                elapsed    = time.perf_counter() - t_start
                fps        = round(frame_cnt / elapsed, 1)

                if not alive:
                    break

                await ws.send_text(json.dumps({
                    "detections": dets,
                    "fps":        fps,
                    "count":      cnt,
                    "peak":       peak_count,
                }))
        except Exception as e:
            print(f"[WS][proc] {e}")
            traceback.print_exc()
        finally:
            alive = False

    try:
        await asyncio.gather(receiver(), processor())
    except Exception as e:
        print(f"[WS] {e}")
    finally:
        print("[WS] Client disconnected")


# ─────────────────────────────── MAIN ───────────────────────────────────────
if __name__ == "__main__":
    import os as _os
    _ssl_key  = "key.pem"
    _ssl_cert = "cert.pem"
    _use_ssl  = _os.path.exists(_ssl_key) and _os.path.exists(_ssl_cert)
    if _use_ssl:
        print(f"  SSL enabled → https://10.251.33.217:8000")
    else:
        print(f"  Running HTTP (no SSL certs found) → http://10.251.33.217:8000")
        print(f"  NOTE: Camera will be blocked on phone without HTTPS.")
        print(f"  Run 'openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj /CN=10.251.33.217' to enable SSL.")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        ssl_keyfile  = _ssl_key  if _use_ssl else None,
        ssl_certfile = _ssl_cert if _use_ssl else None,
    )
