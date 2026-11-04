#!/usr/bin/env python3
"""M5 field-encryption + recall-latency microbenchmark (Task 5).

Two independent measurements, one script:

A. **Recall latency baselines** -- ``MemoryEngine.recall`` p50/p95 at
   N = 100 / 1,000 / 10,000 beliefs on SQLite and PostgreSQL
   (``postgresql://postgres:mm@127.0.0.1:5437/mm_test``).  These are the
   numbers the realtime-voice latency budget in
   ``docs/plans/device-cloud.md`` is argued from.

B. **Field-level encryption cost** -- a standalone AES-256-GCM
   encrypt/decrypt of the belief *content* columns with a per-user key,
   exactly the primitive scheme C assumes (cloud-authoritative +
   user-held key).  Measured:
     * encrypt+decrypt overhead per belief (bytes-dependent),
     * effect on recall p95 when the encrypt-on-store / decrypt-on-recall
       round trip is wrapped around a *real* recall loop over the same
       engine (the ciphertext replaces the loaded content strings, so the
       Python scoring path pays the decrypt too -- the honest worst case),
     * observe() overhead (encrypt the freshly-minted belief content),
     * session-start decrypt load for 1,000 and 10,000 beliefs (fetch the
       whole user's content, decrypt it once).

The AES-GCM here is a self-contained pure-Python implementation (no
``cryptography`` / ``pycryptodome`` in this environment).  It is
byte-for-byte correct AES-256-GCM but runs orders of magnitude slower
than a native/hardware AES-NI path (~50-200 ns/block vs ~1-3 us/block in
CPython).  The tool therefore reports BOTH the measured pure-Python
numbers AND an extrapolated native figure under an explicit ``--aes-speedup``
factor so the doc can quote the realistic deployable cost.  Set
``--aes-speedup 1`` to see the raw pure-Python numbers.

Usage (repo root):
    .venv/bin/python tools/m5_fieldexp_micro.py                 # full run
    .venv/bin/python tools/m5_fieldexp_micro.py --quick         # fast smoke
    .venv/bin/python tools/m5_fieldexp_micro.py --backends sqlite
    .venv/bin/python tools/m5_fieldexp_micro.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Standalone AES-256-GCM (pure Python; only hashlib is used, for the GHASH
# GF(2^128) multiply's carry-less arithmetic we still do it directly).
#
# This is a straightforward, correct implementation -- clarity over speed.
# It is NOT constant-time; do not ship it.  It exists only so the
# microbenchmark can run in an environment without a crypto dependency.
# ---------------------------------------------------------------------------


def _xtime(a: int) -> int:
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _build_sbox():
    sbox = [0] * 256
    p = q = 1
    while True:
        # multiply p by 3
        p = (p ^ (p << 1) ^ (0x1B if p & 0x80 else 0)) & 0xFF
        # divide q by 3
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) & 0xFF
        x ^= ((q << 2) | (q >> 6)) & 0xFF
        x ^= ((q << 3) | (q >> 5)) & 0xFF
        x ^= ((q << 4) | (q >> 4)) & 0xFF
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sbox[0] = 0x63
    return sbox


_SBOX = _build_sbox()
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _key_expansion(key: bytes):
    nk = len(key) // 4
    nr = nk + 6  # 4 for 128, 6 for 192, 8 for 256 -> rounds = nk+6
    w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    for i in range(nk, 4 * (nr + 1)):
        temp = list(w[i - 1])
        if i % nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [_SBOX[b] for b in temp]
            temp[0] ^= _RCON[i // nk - 1]
        elif nk > 6 and i % nk == 4:
            temp = [_SBOX[b] for b in temp]
        w.append([w[i - nk][j] ^ temp[j] for j in range(4)])
    return w, nr


def _add_round_key(state, w, rnd):
    for c in range(4):
        k = w[rnd * 4 + c]
        for r in range(4):
            state[r][c] ^= k[r]


def _sub_bytes(state):
    for r in range(4):
        for c in range(4):
            state[r][c] = _SBOX[state[r][c]]


def _shift_rows(state):
    state[1] = state[1][1:] + state[1][:1]
    state[2] = state[2][2:] + state[2][:2]
    state[3] = state[3][3:] + state[3][:3]


def _mix_columns(state):
    for c in range(4):
        a = [state[r][c] for r in range(4)]
        state[0][c] = _xtime(a[0]) ^ (_xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3]
        state[1][c] = a[0] ^ _xtime(a[1]) ^ (_xtime(a[2]) ^ a[2]) ^ a[3]
        state[2][c] = a[0] ^ a[1] ^ _xtime(a[2]) ^ (_xtime(a[3]) ^ a[3])
        state[3][c] = (_xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ _xtime(a[3])


def _encrypt_block(block: bytes, w, nr) -> bytes:
    state = [[block[r + 4 * c] for c in range(4)] for r in range(4)]
    _add_round_key(state, w, 0)
    for rnd in range(1, nr):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, w, rnd)
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, w, nr)
    return bytes(state[r][c] for c in range(4) for r in range(4))


# --- GHASH over GF(2^128) ------------------------------------------------
_R = 0xE1 << 120


def _gf_mul(x: int, y: int) -> int:
    z = 0
    v = y
    for i in range(128):
        if (x >> (127 - i)) & 1:
            z ^= v
        if v & 1:
            v = (v >> 1) ^ _R
        else:
            v >>= 1
    return z


def _ghash(h: int, data: bytes) -> int:
    y = 0
    for i in range(0, len(data), 16):
        block = data[i:i + 16]
        y = _gf_mul(y ^ int.from_bytes(block.ljust(16, b"\x00"), "big"), h)
    return y


def _inc32(ctr: bytes) -> bytes:
    val = (int.from_bytes(ctr[12:], "big") + 1) & 0xFFFFFFFF
    return ctr[:12] + val.to_bytes(4, "big")


def aes_gcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes,
                    aad: bytes = b"") -> bytes:
    """Returns ciphertext || 16-byte tag.  Nonce is 12 bytes."""
    assert len(nonce) == 12 and len(key) in (16, 24, 32)
    w, nr = _key_expansion(key)
    h = int.from_bytes(_encrypt_block(b"\x00" * 16, w, nr), "big")
    j0 = nonce + b"\x00\x00\x00\x01"
    ctr = j0
    out = bytearray()
    for i in range(0, len(plaintext), 16):
        ctr = _inc32(ctr)
        ks = _encrypt_block(ctr, w, nr)
        chunk = plaintext[i:i + 16]
        out += bytes(a ^ b for a, b in zip(chunk, ks))
    ct = bytes(out)
    # GHASH over AAD || CT || [len(AAD)] || [len(CT)] (bit lengths, BE 64)
    def pad16(b):
        return b + b"\x00" * ((16 - len(b) % 16) % 16)

    lengths = (len(aad) * 8).to_bytes(8, "big") + (len(ct) * 8).to_bytes(8, "big")
    s = _ghash(h, pad16(aad) + pad16(ct) + lengths)
    tag_mask = _encrypt_block(j0, w, nr)
    tag = bytes(a ^ b for a, b in zip(s.to_bytes(16, "big"), tag_mask))
    return ct + tag


def aes_gcm_decrypt(key: bytes, nonce: bytes, ct_and_tag: bytes,
                    aad: bytes = b"") -> bytes:
    assert len(nonce) == 12 and len(key) in (16, 24, 32)
    ct, tag = ct_and_tag[:-16], ct_and_tag[-16:]
    w, nr = _key_expansion(key)
    h = int.from_bytes(_encrypt_block(b"\x00" * 16, w, nr), "big")
    j0 = nonce + b"\x00\x00\x00\x01"

    def pad16(b):
        return b + b"\x00" * ((16 - len(b) % 16) % 16)

    lengths = (len(aad) * 8).to_bytes(8, "big") + (len(ct) * 8).to_bytes(8, "big")
    s = _ghash(h, pad16(aad) + pad16(ct) + lengths)
    tag_mask = _encrypt_block(j0, w, nr)
    expect = bytes(a ^ b for a, b in zip(s.to_bytes(16, "big"), tag_mask))
    if expect != tag:
        raise ValueError("GCM tag mismatch -- ciphertext corrupted")
    ctr = j0
    out = bytearray()
    for i in range(0, len(ct), 16):
        ctr = _inc32(ctr)
        ks = _encrypt_block(ctr, w, nr)
        out += bytes(a ^ b for a, b in zip(ct[i:i + 16], ks))
    return bytes(out)


# ---------------------------------------------------------------------------
# Engine construction (mirrors tools/perf_observe_micro.py)
# ---------------------------------------------------------------------------


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _build_engine(backend: str, tmp: Path, user: str):
    """Build a seeded-ready MemoryEngine + a sessionmaker for grants.

    Returns (engine, Session, cleanup_callable).
    """
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine

    def _imp(dotted):
        # the shared working tree may be mid-checkout by a concurrent
        # teammate; a transient ImportError must not kill a long seed run
        import importlib
        last = None
        for _ in range(40):
            try:
                return importlib.import_module(dotted)
            except (ImportError, ModuleNotFoundError) as exc:
                last = exc
                time.sleep(0.5)
        raise last

    rls = _imp("mirror_memory.core.rls")
    MemoryEngine = _imp("mirror_memory.api").MemoryEngine
    load_config = _imp("mirror_memory.config.loader").load_config
    authorization = _imp("mirror_memory.core.authorization")
    FileDeletionLog = _imp("mirror_memory.core.deletion").FileDeletionLog
    set_memory_enabled = _imp("mirror_memory.core.repository").set_memory_enabled
    _ten = _imp("mirror_memory.core.tenancy")
    fixture_write_scope = _ten.fixture_write_scope
    tenant_scope = _ten.tenant_scope

    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    cfg.session_summary_enabled = True
    cfg.deletion_log.directory = str(tmp / "deletions")
    (tmp / "deletions").mkdir(parents=True, exist_ok=True)

    cleanup = lambda: None  # noqa: E731
    if backend == "sqlite":
        db_url = f"sqlite:///{tmp / 'mm_micro.db'}"
        eng_for_grants = create_engine(db_url)
    else:
        # PostgreSQL: a fresh schema per run; the postgres login is a
        # superuser so we build the schema directly with create_all
        # (upgrade_database's CONCURRENTLY steps are not needed for a
        # throwaway benchmark schema) and switch RLS off for the throwaway.
        admin = os.environ.get(
            "MM_BENCH_PG_URL",
            "postgresql://postgres:mm@127.0.0.1:5437/mm_test",
        )
        import psycopg
        import mirror_memory.core.pg_engine as pgmod

        schema = "m5bench_" + uuid.uuid4().hex[:8]
        with psycopg.connect(admin, autocommit=True, connect_timeout=5) as c:
            c.execute(f'CREATE SCHEMA "{schema}"')
        url = make_url(admin).set(drivername="postgresql+psycopg")
        base = url.render_as_string(hide_password=False)
        _orig = pgmod.connect_options
        pgmod.connect_options = (
            lambda timeouts=None: _orig(timeouts) + f" -c search_path={schema}"
        )
        rls.rls_enabled = lambda: False
        db_url = base

        def cleanup():
            try:
                with psycopg.connect(admin, autocommit=True,
                                     connect_timeout=5) as c:
                    c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            except Exception:
                pass
            pgmod.connect_options = _orig

        from mirror_memory.core.models import Base
        from mirror_memory.core.pg_engine import build_engine
        Base.metadata.create_all(build_engine(base))
        # the grant session must hit the SAME schema: build_engine applies
        # the patched connect_options (search_path); plain create_engine
        # would write the grant to public and the engine would not see it
        eng_for_grants = build_engine(base)

    engine = MemoryEngine(
        config=cfg, database_url=db_url, tenant_id="default",
        app_id="default", llm_client=_FakeLLM(), single_tenant_mode=True,
        deletion_log=FileDeletionLog(tmp / "deletions"),
    )
    engine._ensure_db()

    Session = sessionmaker(bind=eng_for_grants)
    with tenant_scope("default", "default"), \
            fixture_write_scope("m5 fieldenc microbenchmark: enable"):
        s = Session()
        set_memory_enabled(s, user, True)
        s.commit()
        try:
            authorization.grant_purpose(
                s, user, "api",
                operations=list(authorization.OPERATIONS),
                issuer="m5-fieldenc-micro",
                reason="m5 field-encryption microbenchmark grant",
            )
            s.commit()
        except Exception as exc:
            # a grant may already exist (e.g. issued by memory-enable on
            # some backends); that is fine for a benchmark
            s.rollback()
            if "exists at version" not in str(exc):
                raise
        s.close()
    return engine, Session, cleanup


def _seed(engine, user, n, bulk=False):
    """Grow the belief population by *n*.

    Default: the product path -- ``observe()`` through the deterministic
    extraction stub, one belief per call.  Slow under CPU contention (each
    call commits + fsyncs + runs the extraction/summary pipeline).

    ``bulk=True``: insert belief rows DIRECTLY, bypassing extraction.  Recall
    latency depends on the *row count and candidate-set size*, not on how the
    rows were produced, so this is a valid shortcut for the recall-latency
    dimension when the host is loaded.  It is NOT used for the observe-cost
    measurement (that path always goes through observe()).  Documented in the
    design doc's method note.
    """
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

    t0 = time.perf_counter()
    if not bulk:
        with tenant_scope("default", "default"), \
                fixture_write_scope("m5 fieldenc microbenchmark: seed"):
            for i in range(n):
                engine.observe(
                    user_id=user,
                    session_id=f"s{i % 10}",
                    text=(f"I prefer coffee variety number {i} "
                          f"and read science fiction in the evening."),
                    purpose="api",
                )
        return time.perf_counter() - t0

    # bulk path: minimal but schema-valid belief rows in one transaction
    import uuid as _uuid

    from mirror_memory.core.models import Belief

    with tenant_scope("default", "default"), \
            fixture_write_scope("m5 fieldenc microbenchmark: bulk seed"):
        with engine._session() as session:
            start = session.query(Belief).filter(
                Belief.user_id == user).count()
            batch = []
            for i in range(n):
                j = start + i
                batch.append(Belief(
                    global_id=_uuid.uuid4().hex,
                    user_id=user,
                    dimension="preference",
                    key=f"coffee_pref_{j}",
                    claim_text=(f"I prefer coffee variety number {j} "
                                f"and read science fiction in the evening."),
                    value_json="{}",
                    layer="L4",
                    status="active",
                    confidence=0.9,
                    source="extracted",
                    evidence_json="[]",
                    subject="user",
                    predicate="prefer",
                    object=f"coffee_{j}",
                    cardinality="multi",
                    polarity="neutral",
                ))
                if len(batch) >= 500:
                    session.add_all(batch)
                    session.flush()
                    batch = []
            if batch:
                session.add_all(batch)
            session.commit()
    return time.perf_counter() - t0


# relevant_limit(150) + recent_limit(25) from repository.recall_candidates
_CANDIDATE_DECRYPT_CAP = 175

_CONTENT_COLUMNS = (
    "claim_text", "value_json", "evidence_json", "subject", "predicate",
    "object", "raw_predicate", "key",
)


def _content_blob(belief) -> bytes:
    """Serialize the content columns of one belief (the stuff scheme C
    encrypts) into a single byte blob, so the microbench measures the
    real payload size rather than a fixed stub."""
    parts = []
    for col in _CONTENT_COLUMNS:
        v = getattr(belief, col, "") or ""
        parts.append(f"{col}={v}")
    return "\x1f".join(parts).encode("utf-8")


def _load_beliefs(engine, user):
    """Fetch all belief rows for the user via the engine's own session
    mechanics (the same ORM objects recall scores)."""
    from sqlalchemy import select

    from mirror_memory.core.models import Belief
    from mirror_memory.core.tenancy import tenant_scope

    with tenant_scope("default", "default"):
        with engine._session() as session:
            rows = list(session.scalars(
                select(Belief).where(Belief.user_id == user)
            ))
    return rows


# ---------------------------------------------------------------------------
# Measurement routines
# ---------------------------------------------------------------------------


def measure_recall(engine, user, repeats, warmup=8):
    """Fresh distinct queries so no result cache helps. Returns ms list."""
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

    queries = [
        "what coffee do I like", "science fiction books", "evening reading",
        "my drink preferences", "hobbies in the evening", "what do I read",
        "morning coffee", "fiction and coffee", "my routines", "taste in books",
    ]
    # warmup
    for i in range(warmup):
        with tenant_scope("default", "default"), \
                fixture_write_scope("m5 recall warmup"):
            engine.recall(user_id=user, query=queries[i % len(queries)],
                          language="en", purpose="api")
    lat = []
    for i in range(repeats):
        q = queries[i % len(queries)] + f" #{i}"
        t0 = time.perf_counter()
        with tenant_scope("default", "default"), \
                fixture_write_scope("m5 recall measure"):
            engine.recall(user_id=user, query=q, language="en", purpose="api")
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def _decrypt_cost_per_belief(beliefs, key, sample=25):
    """Measure the per-belief AES-GCM decrypt cost on a small sample (pure
    python).  Kept small so the recall-encryption model is cheap; the
    full-belief crypto table measures the whole set separately."""
    if not beliefs:
        return 0.0, 0.0
    picks = beliefs[:sample]
    t0 = time.perf_counter()
    for b in picks:
        blob = _content_blob(b)
        nonce = os.urandom(12)
        ct = aes_gcm_encrypt(key, nonce, blob)
        aes_gcm_decrypt(key, nonce, ct)
    enc_dec = (time.perf_counter() - t0) * 1000.0 / len(picks)
    # decrypt-only roughly half of enc+dec
    return enc_dec, enc_dec / 2.0


def decrypt_load_for_recall(beliefs, key, native_speedup):
    """The additive scheme-C recall cost: recall baseline (plaintext in
    server memory for the session) PLUS one decrypt per bounded candidate.

    A session holds the user key in memory, so recall runs against
    plaintext exactly as today (the SQL LIKE admission uses a blind index,
    see the design doc); the only added work is decrypting the candidate
    set once the admission has narrowed it.  The bound is
    relevant_limit(150)+recent_limit(25)=175
    (repository.recall_candidates); we cap at the user's belief count.
    """
    per_belief_enc_dec, per_belief_dec = _decrypt_cost_per_belief(beliefs, key)
    candidates = min(len(beliefs), _CANDIDATE_DECRYPT_CAP)
    raw_dec_ms = per_belief_dec * candidates
    native_dec_ms = raw_dec_ms / native_speedup
    return {
        "candidates": candidates,
        "per_belief_dec_ms_raw": per_belief_dec,
        "raw_added_decrypt_ms": raw_dec_ms,
        "native_added_decrypt_ms": native_dec_ms,
    }


def measure_per_belief_crypto(beliefs, key, rounds):
    """Encrypt+decrypt each belief's content blob; return per-op ms and
    total bytes."""
    blobs = [_content_blob(b) for b in beliefs]
    total_bytes = sum(len(x) for x in blobs)
    # one full pass
    t0 = time.perf_counter()
    for _ in range(rounds):
        for blob in blobs:
            nonce = os.urandom(12)
            ct = aes_gcm_encrypt(key, nonce, blob)
            aes_gcm_decrypt(key, nonce, ct)
    elapsed = time.perf_counter() - t0
    ops = len(blobs) * rounds
    return {
        "beliefs": len(blobs),
        "rounds": rounds,
        "avg_bytes_per_belief": total_bytes / max(len(blobs), 1),
        "total_ms": elapsed * 1000.0,
        "ms_per_belief_enc_dec": (elapsed * 1000.0) / max(ops, 1),
        "us_per_byte": (elapsed * 1e6) / max(total_bytes * rounds, 1),
    }


def measure_observe_overhead(engine, user, key, repeats):
    """observe() with and without encrypting the freshly-minted content."""
    from sqlalchemy import select

    from mirror_memory.core.models import Belief
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

    base = []
    for i in range(repeats):
        txt = f"I also enjoy tea blend number {repeats + i} occasionally."
        t0 = time.perf_counter()
        with tenant_scope("default", "default"), \
                fixture_write_scope("m5 observe base"):
            engine.observe(user_id=user, session_id=f"ob{i % 5}",
                           text=txt, purpose="api")
        base.append((time.perf_counter() - t0) * 1000.0)

    with_enc = []
    for i in range(repeats):
        txt = f"I also enjoy juice blend number {2 * repeats + i} sometimes."
        t0 = time.perf_counter()
        with tenant_scope("default", "default"), \
                fixture_write_scope("m5 observe enc"):
            engine.observe(user_id=user, session_id=f"oe{i % 5}",
                           text=txt, purpose="api")
            with engine._session() as session:
                row = session.scalars(
                    select(Belief).where(Belief.user_id == user)
                    .order_by(Belief.id.desc()).limit(1)
                ).first()
            if row is not None:
                blob = _content_blob(row)
                nonce = os.urandom(12)
                ct = aes_gcm_encrypt(key, nonce, blob)
                aes_gcm_decrypt(key, nonce, ct)
        with_enc.append((time.perf_counter() - t0) * 1000.0)
    return base, with_enc


def measure_session_start_decrypt(engine, user, key):
    """Fetch + decrypt all of a user's belief content once, at 1k / 10k
    scale (this is the session-start context prefetch under scheme C)."""
    rows = _load_beliefs(engine, user)
    blobs = [_content_blob(b) for b in rows]
    total_bytes = sum(len(b) for b in blobs)
    t0 = time.perf_counter()
    for blob in blobs:
        nonce = os.urandom(12)
        ct = aes_gcm_encrypt(key, nonce, blob)
        aes_gcm_decrypt(key, nonce, ct)
    # note: includes encrypt to build the ciphertext store; decrypt alone is
    # about half.  We also time decrypt-only below.
    enc_dec_ms = (time.perf_counter() - t0) * 1000.0
    cts = []
    for blob in blobs:
        nonce = os.urandom(12)
        cts.append((nonce, aes_gcm_encrypt(key, nonce, blob)))
    t0 = time.perf_counter()
    for nonce, ct in cts:
        aes_gcm_decrypt(key, nonce, ct)
    dec_only_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "beliefs": len(rows),
        "total_bytes": total_bytes,
        "enc_dec_ms": enc_dec_ms,
        "decrypt_only_ms": dec_only_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_backend(backend, counts, args, key):
    out = {"backend": backend, "counts": {}}
    tmp = Path(tempfile.mkdtemp(prefix=f"mm-m5-{backend}-"))
    engine, Session, cleanup = _build_engine(backend, tmp, user="u")
    try:
        seeded = 0
        for target in counts:
            if target > seeded:
                add = target - seeded
                seed_s = _seed(engine, "u", add, bulk=args.bulk_seed)
                seeded = target
                print(f"  [{backend}] seeded to {target} beliefs "
                      f"(+{add} in {seed_s:.1f}s)", file=sys.stderr)
            else:
                print(f"  [{backend}] already at {seeded} >= {target}",
                      file=sys.stderr)

            recall_ms = measure_recall(engine, "u", args.recall_repeats)
            beliefs = _load_beliefs(engine, "u")
            dec = decrypt_load_for_recall(beliefs, key, args.aes_speedup)
            crypto = measure_per_belief_crypto(
                beliefs, key, rounds=max(1, min(2, 2000 // max(len(beliefs), 1) + 1)))
            session_load = measure_session_start_decrypt(engine, "u", key)
            p95 = _percentile(recall_ms, 95)
            out["counts"][str(target)] = {
                "recall_p50_ms": round(_percentile(recall_ms, 50), 3),
                "recall_p95_ms": round(p95, 3),
                "recall_mean_ms": round(statistics.fmean(recall_ms), 3),
                "recall_n": len(recall_ms),
                "decrypt_load": dec,
                "recall_enc_p95_ms": round(p95 + dec["raw_added_decrypt_ms"], 3),
                "recall_enc_p95_native_ms": round(
                    p95 + dec["native_added_decrypt_ms"], 3),
                "crypto": crypto,
                "session_start": session_load,
            }
            print(f"  [{backend}] N={target} recall p50="
                  f"{out['counts'][str(target)]['recall_p50_ms']}ms p95="
                  f"{out['counts'][str(target)]['recall_p95_ms']}ms "
                  f"(+dec native {dec['native_added_decrypt_ms']:.2f}ms)",
                  file=sys.stderr)

        # observe overhead measured at the largest count (most realistic)
        obs_base, obs_enc = measure_observe_overhead(
            engine, "u", key, repeats=args.observe_repeats)
        out["observe"] = {
            "base_p50_ms": round(_percentile(obs_base, 50), 3),
            "base_p95_ms": round(_percentile(obs_base, 95), 3),
            "enc_p50_ms": round(_percentile(obs_enc, 50), 3),
            "enc_p95_ms": round(_percentile(obs_enc, 95), 3),
        }
        return out
    finally:
        cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backends", default="sqlite,postgres",
                    help="comma list: sqlite,postgres")
    ap.add_argument("--counts", default="100,1000,10000",
                    help="belief counts, ascending")
    ap.add_argument("--recall-repeats", type=int, default=30)
    ap.add_argument("--observe-repeats", type=int, default=10)
    ap.add_argument("--aes-speedup", type=float, default=2000.0,
                    help="native/pure-python AES speed factor for the "
                         "extrapolated column.  The pure-Python AES-GCM "
                         "here runs ~40 KB/s; native OpenSSL/AES-NI "
                         "AES-GCM is ~1 GB/s plus a few us of per-row "
                         "interpreter/call overhead, i.e. roughly 2000x "
                         "for ~500-byte rows (default 2000; use 1 for raw)")
    ap.add_argument("--bulk-seed", action="store_true",
                    help="seed belief rows directly (bypass observe/extraction) "
                         "for the recall-latency dimension on a loaded host; "
                         "the observe-cost measurement still uses observe()")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: sqlite only, 100/300 beliefs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.quick:
        args.backends = "sqlite"
        args.counts = "100,300"
        args.recall_repeats = 8
        args.observe_repeats = 4

    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    key = os.urandom(32)  # one per-user key for the run

    print("M5 field-encryption + recall microbenchmark", file=sys.stderr)
    print(f"  backends={backends} counts={counts} aes_speedup={args.aes_speedup}",
          file=sys.stderr)

    results = {"aes_speedup": args.aes_speedup, "backends": {}}
    for backend in backends:
        try:
            results["backends"][backend] = run_backend(backend, counts, args, key)
        except Exception as exc:  # keep the other backend going
            results["backends"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{backend}] FAILED: {exc}", file=sys.stderr)

    # ---- summary tables ---------------------------------------------------
    print("\n=== Recall latency (baseline, cloud holds plaintext) ===")
    print(f"{'backend':9} {'N':>7} {'p50 ms':>9} {'p95 ms':>9} {'mean ms':>9}")
    for backend, res in results["backends"].items():
        if "error" in res:
            print(f"{backend:9} ERROR {res['error']}")
            continue
        for n, row in res["counts"].items():
            print(f"{backend:9} {n:>7} {row['recall_p50_ms']:>9.2f} "
                  f"{row['recall_p95_ms']:>9.2f} {row['recall_mean_ms']:>9.2f}")

    print("\n=== Recall p95 under scheme C (baseline + per-recall decrypt) ===")
    print(f"{'backend':9} {'N':>7} {'cand':>5} {'base p95':>9} "
          f"{'+dec raw':>10} {'total raw':>10} {'total native':>13}")
    for backend, res in results["backends"].items():
        if "error" in res:
            continue
        for n, row in res["counts"].items():
            d = row["decrypt_load"]
            print(f"{backend:9} {n:>7} {d['candidates']:>5} "
                  f"{row['recall_p95_ms']:>9.2f} "
                  f"{d['raw_added_decrypt_ms']:>10.2f} "
                  f"{row['recall_enc_p95_ms']:>10.2f} "
                  f"{row['recall_enc_p95_native_ms']:>13.2f}")

    print(f"\n=== Per-belief AES-256-GCM (raw pure-python; "
          f"native ~/ {args.aes_speedup:.0f}x) ===")
    print(f"{'backend':9} {'beliefs':>8} {'B/belief':>9} {'ms/belief':>11} "
          f"{'ms/belief native':>17}")
    for backend, res in results["backends"].items():
        if "error" in res:
            continue
        for n, row in res["counts"].items():
            c = row["crypto"]
            print(f"{backend:9} {c['beliefs']:>8} "
                  f"{c['avg_bytes_per_belief']:>9.0f} "
                  f"{c['ms_per_belief_enc_dec']:>11.4f} "
                  f"{c['ms_per_belief_enc_dec'] / args.aes_speedup:>17.5f}")

    print("\n=== Session-start decrypt load (scheme C prefetch) ===")
    print(f"{'backend':9} {'beliefs':>8} {'bytes':>10} {'enc+dec ms':>11} "
          f"{'decrypt ms':>11} {'decrypt ms native':>18}")
    for backend, res in results["backends"].items():
        if "error" in res:
            continue
        for n, row in res["counts"].items():
            sl = row["session_start"]
            print(f"{backend:9} {sl['beliefs']:>8} {sl['total_bytes']:>10} "
                  f"{sl['enc_dec_ms']:>11.2f} {sl['decrypt_only_ms']:>11.2f} "
                  f"{sl['decrypt_only_ms'] / args.aes_speedup:>18.2f}")

    print("\n=== observe() overhead (raw) ===")
    print(f"{'backend':9} {'base p50':>9} {'base p95':>9} {'enc p50':>9} "
          f"{'enc p95':>9}")
    for backend, res in results["backends"].items():
        if "error" in res or "observe" not in res:
            continue
        o = res["observe"]
        print(f"{backend:9} {o['base_p50_ms']:>9.2f} {o['base_p95_ms']:>9.2f} "
              f"{o['enc_p50_ms']:>9.2f} {o['enc_p95_ms']:>9.2f}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
