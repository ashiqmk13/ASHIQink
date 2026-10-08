"""Whiteboard drawing engine.

Pipeline for one scene:
  Layer(s)  ->  plan_layer()  ->  pen path  P (N,2) + K (N,)   (K = how the pen moves *to* point i)
            ->  render_scene() -> frames (numpy uint8 HxWx3) with hand sprite, enter/exit, zoom, eraser wipe

Pen-move kinds
  0 travel (pen lifted, nothing revealed)      1 ink stroke (follows the skeleton of the line art)
  2 colour brush (reveals the coloured picture) 3 hatch fill (thick ink areas)
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .hands import blit
from .lineart import Layer

SPEED = {0: 5.0, 1: 1.0, 2: 2.4, 3: 1.7}   # relative pen speed per kind (cost = distance / speed)


# ----------------------------------------------------------------------------- path helpers
def serpentine(region: np.ndarray, step: int, kind: int, join_gap: int, ox=0, oy=0, flip=False):
    """Zig-zag the pen over `region` (bool). Only visits rows/runs that contain region pixels."""
    H, W = region.shape
    step = max(2, int(step)); half = max(1, step // 2)
    pts, kinds = [], []
    for y in range(half, H + half, step):
        yy = min(y, H - 1)
        band = region[max(0, y - half): y + half + 1].any(axis=0)
        if not band.any():
            continue
        d = np.diff(np.concatenate(([0], band.astype(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        runs = []
        for s, e in zip(starts, ends):
            if runs and s - runs[-1][1] <= join_gap:
                runs[-1][1] = e
            else:
                runs.append([int(s), int(e)])
        if flip:
            runs = [[e, s] for s, e in reversed(runs)]
        for s, e in runs:
            pts.append((s + ox, yy + oy)); kinds.append(0)
            pts.append((e + ox, yy + oy)); kinds.append(kind)
        flip = not flip
    if not pts:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.uint8)
    return np.array(pts, np.float32), np.array(kinds, np.uint8)


def contour_paths(mask: np.ndarray, spacing: int):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(np.float32)[::max(1, spacing)] for c in cnts if len(c) >= 4]


def skeleton_paths(mask: np.ndarray, pen_xy, spacing: int):
    """Centre-line strokes of a binary mask, walked like a pen would (continues straight through junctions)."""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return contour_paths(mask, spacing)
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n == 0:
        return []
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if n < 4:
        return [pts]
    H, W = sk.shape
    idmap = np.full((H + 2, W + 2), -1, np.int32)
    idmap[ys + 1, xs + 1] = np.arange(n, dtype=np.int32)
    nb = [[] for _ in range(n)]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nid = idmap[ys + 1 + dy, xs + 1 + dx]
            idx = np.nonzero(nid >= 0)[0]
            for i, j in zip(idx.tolist(), nid[idx].tolist()):
                nb[i].append(j)
    visited = np.zeros(n, bool)
    tree = cKDTree(pts)

    def walk(s):
        path, cur, d = [s], s, None
        while True:
            cand = [j for j in nb[cur] if not visited[j]]
            if not cand:
                break
            if len(cand) == 1 or d is None:
                nxt = min(cand, key=lambda j: abs(pts[j][0] - pts[cur][0]) + abs(pts[j][1] - pts[cur][1]))
            else:
                def score(j):
                    v = pts[j] - pts[cur]
                    return float(np.dot(v / (np.hypot(*v) + 1e-6), d)) - 0.03 * float(np.hypot(*v))
                nxt = max(cand, key=score)
            for j in cand:                       # swallow staircase pixels next to the chosen one
                if j != nxt and j in nb[nxt]:
                    visited[j] = True
            v = pts[nxt] - pts[cur]
            nv = np.hypot(*v) + 1e-6
            d = v / nv if d is None else 0.6 * d + 0.4 * v / nv
            visited[nxt] = True
            path.append(nxt); cur = nxt
        return path

    def pick_start(p):
        k = 24
        while True:
            kk = min(k, n)
            _, idx = tree.query(p, k=kk)
            for j in np.atleast_1d(idx):
                if not visited[j]:
                    return int(j)
            if kk >= n:
                return None
            k *= 4

    paths, pen = [], np.asarray(pen_xy, np.float32)
    while not visited.all():
        s = pick_start(pen)
        if s is None:
            break
        visited[s] = True
        fwd = walk(s)
        back = walk(s)
        full = back[::-1] + fwd[1:]
        arr = pts[full]
        sp = max(1, spacing)
        sub = arr[::sp]
        if len(arr) > 1 and (len(arr) - 1) % sp != 0:
            sub = np.vstack([sub, arr[-1:]])
        paths.append(sub)
        pen = arr[-1]
    return paths


def _clusters(mask: np.ndarray, gap: int):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap | 1, gap | 1))
    d = cv2.dilate(mask.astype(np.uint8), k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        sub = (lab[y:y + h, x:x + w] == i) & mask[y:y + h, x:x + w]
        if sub.sum() >= 4:
            out.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), mask=sub, cx=x + w / 2, cy=y + h / 2))
    return out


def _order_clusters(cl, mode, start_xy):
    if mode == "reading":
        if not cl:
            return cl
        med_h = float(np.median([c["h"] for c in cl]))
        cl = sorted(cl, key=lambda c: c["cy"])
        lines, cur = [], [cl[0]]
        for c in cl[1:]:
            if abs(c["cy"] - np.mean([q["cy"] for q in cur])) < 0.6 * med_h:
                cur.append(c)
            else:
                lines.append(cur); cur = [c]
        lines.append(cur)
        return [c for ln in lines for c in sorted(ln, key=lambda c: c["cx"])]
    remaining, out, pen = list(cl), [], np.array(start_xy, np.float32)
    while remaining:
        j = min(range(len(remaining)),
                key=lambda i: np.hypot(remaining[i]["cx"] - pen[0], remaining[i]["cy"] - pen[1]) + 0.35 * remaining[i]["cy"])
        c = remaining.pop(j); out.append(c)
        pen = np.array([c["cx"], c["cy"]], np.float32)
    return out


# ----------------------------------------------------------------------------- planning
def plan_layer(layer: Layer, mode: str, W: int, H: int, pen_xy, seed=0):
    """Return dict(P, K, r_ink, r_col) for one layer."""
    r = max(2, int(round(0.0035 * W)))
    R = max(8, int(round(0.03 * W)))
    spacing = max(1, int(r * 0.9))
    foot = layer.footprint > 0
    Ps, Ks = [], []
    pen = np.array(pen_xy, np.float32)

    def push(P, K):
        nonlocal pen
        if len(P):
            Ps.append(P); Ks.append(K); pen = P[-1].copy()

    if layer.kind == "image" and mode in ("scanner", "chunks"):
        Rb = max(8, int(0.022 * W))
        r_col = Rb
        if mode == "scanner":
            push(*serpentine(foot, int(Rb * 1.5), 2, Rb * 2))
        else:
            ys, xs = np.nonzero(foot)
            if len(xs):
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                nx = max(2, min(5, int((x1 - x0) / (0.22 * W)) + 1))
                ny = max(2, min(5, int((y1 - y0) / (0.22 * W)) + 1))
                tiles = [(i, j) for i in range(nx) for j in range(ny)]
                np.random.RandomState(seed + 7).shuffle(tiles)
                for i, j in tiles:
                    tx0, tx1 = x0 + (x1 - x0) * i // nx, x0 + (x1 - x0) * (i + 1) // nx
                    ty0, ty1 = y0 + (y1 - y0) * j // ny, y0 + (y1 - y0) * (j + 1) // ny
                    sub = foot[ty0:ty1, tx0:tx1]
                    if sub.any():
                        push(*serpentine(sub, int(Rb * 1.5), 2, Rb * 2, ox=tx0, oy=ty0, flip=bool((i + j) % 2)))
        return _pack(Ps, Ks, r, r_col)

    # ---- ink strokes, object by object
    ink_mask = layer.alpha > 100
    gap = max(3, int(r * 2.2)) if layer.kind == "text" else max(5, int(0.012 * W))
    clusters = _order_clusters(_clusters(ink_mask, gap), layer.order, (0, 0) if len(Ps) == 0 else pen)
    for c in clusters:
        sub, ox, oy = c["mask"], c["x"], c["y"]
        paths = skeleton_paths(sub, pen - (ox, oy), spacing)
        cover = np.zeros(sub.shape, np.uint8)
        for p in paths:
            pp = np.round(p).astype(np.int32)
            if len(pp) == 1:
                cv2.circle(cover, tuple(pp[0]), r, 255, -1)
            else:
                cv2.polylines(cover, [pp.reshape(-1, 1, 2)], False, 255, 2 * r + 1)
            P = p + (ox, oy)
            K = np.ones(len(P), np.uint8); K[0] = 0
            push(P.astype(np.float32), K)
        rem = sub & (cover == 0)
        if rem.sum() > max(30, 0.03 * sub.sum()):
            push(*serpentine(rem, max(2, int(r * 1.6)), 3, int(r * 3), ox=ox, oy=oy))

    # ---- colour pass
    r_col = R
    if layer.use_color and foot.any():
        push(*serpentine(foot, int(R * 1.5), 2, R * 2))
    return _pack(Ps, Ks, r, r_col)


def _pack(Ps, Ks, r, R):
    if not Ps:
        return dict(P=np.zeros((0, 2), np.float32), K=np.zeros(0, np.uint8), r_ink=r, r_col=R)
    return dict(P=np.vstack(Ps), K=np.concatenate(Ks), r_ink=r, r_col=R)


# ----------------------------------------------------------------------------- rendering
def _ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def compose_final(layers, bg_rgb, W, H):
    out = np.empty((H, W, 3), np.uint8); out[:] = bg_rgb
    for l in layers:
        np.copyto(out, l.color_rgb, where=(l.footprint > 0)[..., None])
    return out


def render_scene(layers, bg_rgb, W, H, fps, n_frames, draw_start, draw_dur, exit_dur, hand=None,
                 anim_mode="draw", zoom=True, wipe_frames=0, overlay=None, seed=0, cancel=None):
    """Generator of RGB frames. Timeline: hand enters -> draws (draw_dur) -> hand leaves -> hold (+zoom) -> eraser wipe."""
    final = compose_final(layers, bg_rgb, W, H)
    plans, pen = [], (0.0, 0.0)
    for li, l in enumerate(layers):
        p = plan_layer(l, anim_mode, W, H, pen, seed + li)
        if len(p["P"]):
            pen = tuple(p["P"][-1])
        plans.append(p)
    P = np.vstack([p["P"] for p in plans]) if plans else np.zeros((0, 2), np.float32)
    K = np.concatenate([p["K"] for p in plans]) if plans else np.zeros(0, np.uint8)
    Lid = np.concatenate([np.full(len(p["P"]), i, np.int32) for i, p in enumerate(plans)]) if plans else np.zeros(0, np.int32)
    N = len(P)

    if N:
        seg = np.hypot(*(P[1:] - P[:-1]).T)
        cost = np.concatenate([[0.0], seg / np.array([SPEED[int(k)] for k in K[1:]], np.float32)])
        C = np.cumsum(cost); Ctot = float(C[-1]) + 1e-6
        draw_dur = max(0.3, min(draw_dur, Ctot / (0.10 * W)))     # never crawl: simple drawings finish early
        tpos = draw_start + C / Ctot * draw_dur
        Pint = np.round(P).astype(np.int32)
    else:
        tpos = np.zeros(0); draw_dur = 0.0
    draw_end = draw_start + draw_dur
    snap_end = draw_end + 0.30
    hold0 = draw_end + exit_dur
    wipe0 = n_frames - wipe_frames

    cur = np.empty((H, W, 3), np.uint8); cur[:] = bg_rgb
    Rink = [np.zeros((H, W), np.uint8) for _ in layers]
    Rcol = [np.zeros((H, W), np.uint8) for _ in layers]
    abin = [(l.alpha > 0) for l in layers]
    fbin = [(l.footprint > 0) for l in layers]
    consumed = 0
    off_start = (W * 0.86, H + (hand.h if hand else 0) + 10)

    def consume(a, b):
        s = max(a, 1)
        if b <= s:
            return
        ks, ls = K[s:b], Lid[s:b]
        brk = np.nonzero((ks[1:] != ks[:-1]) | (ls[1:] != ls[:-1]))[0] + 1
        bounds = [0] + brk.tolist() + [len(ks)]
        touched = {}
        for u, v in zip(bounds[:-1], bounds[1:]):
            kind, li = int(ks[u]), int(ls[u])
            if kind == 0:
                continue
            pts = Pint[s + u - 1: s + v]
            ink = kind in (1, 3)
            rad = plans[li]["r_ink"] if ink else plans[li]["r_col"]
            mask = Rink[li] if ink else Rcol[li]
            if len(pts) == 1:
                cv2.circle(mask, tuple(int(q) for q in pts[0]), rad, 255, -1)
            else:
                cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, 2 * rad + 1)
            x0, y0 = pts.min(axis=0) - rad - 2; x1, y1 = pts.max(axis=0) + rad + 3
            bb = touched.get(li)
            touched[li] = (min(x0, bb[0]), min(y0, bb[1]), max(x1, bb[2]), max(y1, bb[3])) if bb else (x0, y0, x1, y1)
        for li, (x0, y0, x1, y1) in touched.items():
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            sl = (slice(y0, y1), slice(x0, x1))
            m = (Rink[li][sl] > 0) & abin[li][sl]
            if m.any():
                np.copyto(cur[sl], layers[li].ink_rgb[sl], where=m[..., None])
            m = (Rcol[li][sl] > 0) & fbin[li][sl]
            if m.any():
                np.copyto(cur[sl], layers[li].color_rgb[sl], where=m[..., None])

    rng = np.random.RandomState(seed)
    ph = rng.rand(2) * 6.28
    for fi in range(n_frames):
        if cancel is not None and cancel.is_set():
            return
        t = fi / fps
        if N:
            idx = int(np.searchsorted(tpos, t, side="right"))
            if idx > consumed:
                consume(consumed, idx); consumed = idx
        # base image
        if t >= snap_end or N == 0:
            base = final
        elif t > draw_end:
            a = _ease((t - draw_end) / 0.30)
            base = cv2.addWeighted(cur, 1 - a, final, a, 0)
        else:
            base = cur
        # zoom while holding
        if zoom and t >= hold0 and wipe0 > 0:
            T0 = hold0 * fps
            u = (fi - T0) / max(1.0, (n_frames - T0))
            s = 1.0 + 0.045 * _ease(u)
            M = np.array([[s, 0, (1 - s) * W / 2], [0, s, (1 - s) * H / 2]], np.float32)
            frame = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=tuple(int(c) for c in bg_rgb))
        else:
            frame = base.copy()
        # hand
        if hand is not None and N:
            if t < draw_start:
                u = _ease(t / max(1e-3, draw_start))
                px = off_start[0] + (P[0][0] - off_start[0]) * u
                py = off_start[1] + (P[0][1] - off_start[1]) * u
                blit(frame, hand, px, py)
            elif t <= draw_end:
                px = float(np.interp(t, tpos, P[:, 0])); py = float(np.interp(t, tpos, P[:, 1]))
                px += 1.6 * np.sin(t * 23 + ph[0]); py += 1.6 * np.sin(t * 19 + ph[1])
                blit(frame, hand, px, py)
            elif t < hold0:
                u = _ease((t - draw_end) / max(1e-3, exit_dur))
                ex, ey = W * 1.02, H * 0.9
                px = P[-1][0] + (ex - P[-1][0]) * u
                py = P[-1][1] + (ey + hand.h * 0.2 - P[-1][1]) * u
                blit(frame, hand, px, py)
        # eraser wipe
        if wipe_frames and fi >= wipe0:
            q = (fi - wipe0 + 1) / wipe_frames
            x = int(_ease(q) * (W + 40))
            frame[:, :min(W, x)] = bg_rgb
            bw = max(8, int(W * 0.03))
            xs, xe = max(0, x), min(W, x + bw)
            if xe > xs:
                frame[:, xs:xe] = (frame[:, xs:xe] * 0.35 + np.array([150, 155, 165]) * 0.65).astype(np.uint8)
        if overlay is not None:
            frame = overlay(frame, t)
        yield frame
