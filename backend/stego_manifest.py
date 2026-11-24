"""Track-5 CTF support: frame-embedded SHA-256 manifest hash + PCAP/CSV artifact extractor.

Embedding
---------
A bit-string  HEADER(16) | SHA-256(256) | CRC-16(16) = 288 bits  is written as luminance blocks (bit 1 -> 215, bit 0 -> 40) into a strip along the bottom of
a frame. Blocks are 16x16 px (one H.264 macroblock) and the whole payload is repeated over every slot of the strip (repetition code), so that
decoding is a per-bit majority vote over repeats AND over frames. This survives 4:2:0 chroma subsampling and lossy H.264 (verified in tests).
The block strip is visible by design (it is a CTF; the hash is a puzzle element, not a covert channel).

Artifacts
---------
`build_artifacts` produces a telemetry CSV and a libpcap capture (Ethernet/IPv4/TCP carrying RFC 6455 WebSocket text frames that hold OCPP-J
messages). `manifest_hash` is the SHA-256 of the canonical JSON of {file: sha256}. The hash embedded in the video is that manifest hash, so
a solver extracts it from the video, recomputes it over the artifact files and thereby proves the artifacts belong to the footage.
"""
from __future__ import annotations
import hashlib, io, json, struct, zlib
import numpy as np

HEADER = 0xA55A
BLOCK = 16
LO, HI = 40, 215


# ------------------------------------------------------------------ bit-level helpers
def crc16(b: bytes) -> int:
    c = 0xFFFF
    for x in b:
        c ^= x << 8
        for _ in range(8): c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def payload_bits(digest_hex: str) -> np.ndarray:
    d = bytes.fromhex(digest_hex); assert len(d) == 32
    raw = struct.pack(">H", HEADER) + d + struct.pack(">H", crc16(d))
    return np.unpackbits(np.frombuffer(raw, np.uint8))


def _layout(h: int, w: int, rows: int = 4):
    cols = w // BLOCK; return rows, cols, h - rows * BLOCK


# ------------------------------------------------------------------ embed / extract
def embed_hash(frame: np.ndarray, digest_hex: str, rows: int = 4) -> np.ndarray:
    """Return a copy of an HxWx3 uint8 frame with the hash strip written into its bottom `rows` block-rows."""
    f = frame.copy(); h, w = f.shape[:2]; rows, cols, y0 = _layout(h, w, rows); bits = payload_bits(digest_hex); n = len(bits)
    for r in range(rows):
        for c in range(cols):
            b = bits[(r * cols + c) % n]
            f[y0 + r * BLOCK: y0 + (r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK, :] = HI if b else LO
    return f


def _slot_values(frame: np.ndarray, rows: int = 4) -> np.ndarray:
    h, w = frame.shape[:2]; rows, cols, y0 = _layout(h, w, rows); lum = frame[..., :3].astype(float).mean(axis=2)
    inner = BLOCK // 4                                                  # sample block centres only (codec ringing sits on edges)
    v = np.zeros(rows * cols)
    for r in range(rows):
        for c in range(cols):
            blk = lum[y0 + r * BLOCK + inner: y0 + (r + 1) * BLOCK - inner, c * BLOCK + inner:(c + 1) * BLOCK - inner]
            v[r * cols + c] = blk.mean()
    return v


def extract_hash(frames, rows: int = 4) -> dict:
    """Majority-vote decode over all slots of all given frames. Returns {ok, digest_hex, crc_ok, header_ok, n_frames, bit_margin}."""
    n = 288; votes = np.zeros(n); cnt = np.zeros(n); mean_lum = np.zeros(n)
    for fr in frames:
        v = _slot_values(fr, rows)
        for i, x in enumerate(v):
            k = i % n; votes[k] += (x > 127.5); cnt[k] += 1; mean_lum[k] += x
    bits = (votes / np.maximum(cnt, 1) > 0.5).astype(np.uint8); raw = np.packbits(bits).tobytes()
    hdr = struct.unpack(">H", raw[:2])[0]; d = raw[2:34]; crc = struct.unpack(">H", raw[34:36])[0]
    ok = hdr == HEADER and crc == crc16(d)
    margin = float(np.mean(np.abs(votes / np.maximum(cnt, 1) - 0.5)) * 2)
    return {"ok": bool(ok), "header_ok": hdr == HEADER, "crc_ok": crc == crc16(d), "digest_hex": d.hex(), "n_frames": len(frames), "bit_margin": margin}


# ------------------------------------------------------------------ video helpers
def write_mp4(path, frames, fps=30, crf: int = 23):
    import imageio.v2 as imageio
    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, ffmpeg_params=["-crf", str(crf)], macro_block_size=16)
    for f in frames: w.append_data(f)
    w.close()


def read_mp4(path, every: int = 1, limit: int | None = None):
    import imageio.v2 as imageio
    r = imageio.get_reader(str(path)); out = []
    for i, f in enumerate(r):
        if i % every == 0: out.append(np.asarray(f))
        if limit and len(out) >= limit: break
    r.close(); return out


# ------------------------------------------------------------------ PCAP (libpcap, Ethernet/IPv4/TCP, WebSocket text frames)
def _ws_frame(payload: bytes, mask: bool) -> bytes:
    n = len(payload); b0 = 0x81
    hdr = bytes([b0]) + (bytes([n | (0x80 if mask else 0)]) if n < 126 else bytes([126 | (0x80 if mask else 0)]) + struct.pack(">H", n))
    if mask:
        key = hashlib.sha256(payload).digest()[:4]                      # deterministic masking key (reproducible artifact)
        return hdr + key + bytes(p ^ key[i % 4] for i, p in enumerate(payload))
    return hdr + payload


def _ip_sum(h: bytes) -> int:
    s = sum(struct.unpack(">%dH" % (len(h) // 2), h)); s = (s & 0xFFFF) + (s >> 16); return (~s) & 0xFFFF


def _tcp_packet(src, dst, sport, dport, seq, ack, payload: bytes, ident: int) -> bytes:
    tcp = struct.pack(">HHIIBBHHH", sport, dport, seq, ack, 5 << 4, 0x18, 65535, 0, 0) + payload
    pseudo = bytes(src) + bytes(dst) + struct.pack(">BBH", 0, 6, len(tcp)); s = sum(struct.unpack(">%dH" % ((len(pseudo + tcp) + 1) // 2), (pseudo + tcp + b"\x00")[:((len(pseudo + tcp) + 1) // 2) * 2]))
    s = (s & 0xFFFF) + (s >> 16); s = (s & 0xFFFF) + (s >> 16); tcp = tcp[:16] + struct.pack(">H", (~s) & 0xFFFF) + tcp[18:]
    ip = struct.pack(">BBHHHBBH", 0x45, 0, 20 + len(tcp), ident & 0xFFFF, 0x4000, 64, 6, 0) + bytes(src) + bytes(dst)
    ip = ip[:10] + struct.pack(">H", _ip_sum(ip)) + ip[12:]
    eth = bytes.fromhex("020000000002") + bytes.fromhex("020000000001") + b"\x08\x00"
    return eth + ip + tcp


CSMS, CP = (10, 20, 0, 5), (10, 20, 0, 77)


def write_pcap(frames: list[dict], forged_ids: set[str] | None = None) -> bytes:
    """frames: [{'t_ms', 'frame': OCPP-J list}], emitted CP->CSMS (masked, per RFC 6455) except those whose uniqueId is in `forged_ids`, which
    are sent from a DIFFERENT source address (10.20.0.99), the artefact a defender is expected to spot."""
    out = io.BytesIO(); out.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    seq = {CP: 1000, (10, 20, 0, 99): 5000}; base = 1_700_000_000
    for i, f in enumerate(frames):
        uid = f["frame"][1]; forged = forged_ids is not None and uid in forged_ids
        src = (10, 20, 0, 99) if forged else CP; dst = CSMS
        pl = _ws_frame(json.dumps(f["frame"], separators=(",", ":")).encode(), mask=True)
        pkt = _tcp_packet(src, dst, 40000 + (1 if forged else 0), 8080, seq[src], 1, pl, i); seq[src] += len(pl)
        t = base + f["t_ms"] / 1000.0
        out.write(struct.pack("<IIII", int(t), int((t % 1) * 1e6), len(pkt), len(pkt))); out.write(pkt)
    return out.getvalue()


def read_pcap(data: bytes) -> list[dict]:
    """Artifact extractor: parse the capture and recover the OCPP-J messages (unmasking WebSocket frames) with src address and timestamp."""
    assert struct.unpack("<I", data[:4])[0] == 0xA1B2C3D4, "not a little-endian libpcap file"
    p = 24; out = []
    while p + 16 <= len(data):
        sec, usec, cl, ol = struct.unpack("<IIII", data[p:p + 16]); p += 16; pkt = data[p:p + cl]; p += cl
        ip = pkt[14:]; ihl = (ip[0] & 15) * 4; src = tuple(ip[12:16]); tcp = ip[ihl:]; off = (tcp[12] >> 4) * 4; ws = tcp[off:]
        if len(ws) < 2: continue
        m = bool(ws[1] & 0x80); n = ws[1] & 0x7F; q = 2
        if n == 126: n = struct.unpack(">H", ws[2:4])[0]; q = 4
        key = ws[q:q + 4] if m else b""; q += 4 if m else 0; body = ws[q:q + n]
        if m: body = bytes(b ^ key[i % 4] for i, b in enumerate(body))
        out.append({"t_s": sec + usec / 1e6, "src": ".".join(map(str, src)), "frame": json.loads(body.decode())})
    return out


# ------------------------------------------------------------------ artifacts + manifest
def telemetry_csv(rows: list[dict]) -> bytes:
    cols = list(rows[0].keys()); lines = [",".join(cols)] + [",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def manifest_hash(files: dict[str, bytes]) -> tuple[str, dict]:
    m = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), m


def flag_for(forged_frame: list) -> str:
    return "TWIN{" + hashlib.sha256(json.dumps(forged_frame, separators=(",", ":")).encode()).hexdigest()[:16] + "}"


def verify(video_frames, files: dict[str, bytes]) -> dict:
    """Solver check: does the hash hidden in the video equal the manifest hash recomputed over the supplied artifact files?"""
    e = extract_hash(video_frames); h, m = manifest_hash(files)
    return {"embedded": e["digest_hex"], "recomputed": h, "match": bool(e["ok"] and e["digest_hex"] == h), "per_file": m, "decode": e}


if __name__ == "__main__":      # python -m backend.stego_manifest extract incident.mp4 | pcap capture.pcap
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "extract": print(json.dumps(extract_hash(read_mp4(sys.argv[2], every=10, limit=6)), indent=1))
    elif len(sys.argv) == 3 and sys.argv[1] == "pcap": [print(m["t_s"], m["src"], json.dumps(m["frame"])) for m in read_pcap(open(sys.argv[2], "rb").read())]
    else: print("usage: python -m backend.stego_manifest extract <video.mp4> | pcap <capture.pcap>")
