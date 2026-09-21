"""
WRDS connectivity check for 230GA Project 2.

Provenance: copied from `asset_embeddings/data/wrds_connection.py` in the
230ZA project, with the project-root hop re-derived for this layout.  Copied,
not imported — a teammate cloning this repo gets a working connection check
without any of my other coursework.

What is going on here, from the top
-----------------------------------
WRDS (Wharton Research Data Services) stores its datasets (CRSP, Compustat,
Fama-French factors, ...) in a **PostgreSQL database** on a server at Wharton.
PostgreSQL is a relational database: tables with rows and columns that you
query with SQL.

Talking to a database is a client/server conversation:

    your Python process  --(TCP + SSL, port 9737)-->  wrds-pgdata.wharton.upenn.edu
        "here is my username/password"                 "ok, session open"
        "SELECT 1"                                     "1"
        "bye"                                          (session closed)

Every step can fail independently, and a good "is the connection on?" check
tells you *which* one did:

  1. Network        — DNS cannot resolve the hostname, or the port is blocked
                      (campus firewall, no internet, VPN).
  2. Authentication — the server is reachable but rejects the credentials.
  3. Session        — logged in, but the session is unusable (rare).

`wrds_connection_on()` walks those three steps in order and returns True only
if all of them succeed.  Run it first in the diagnostic notebook: everything
else in part 1 fails confusingly if this is False.

Vocabulary used below
---------------------
* **driver**: the library that speaks the database's wire protocol for you.
  For PostgreSQL in Python that is `psycopg2`.  The official `wrds` package is
  a thin convenience layer on top of it.
* **connection**: an open, authenticated session with the server.  It is a
  resource on *their* machine too, so we always close it when done.
* **cursor**: the object you send SQL through and read results back from.
* **credentials**: username + password.  They must never be hard-coded in
  source files that get committed, so we read them from a secrets file.

Where the credentials come from (first match wins)
--------------------------------------------------
1. `username` / `password` passed explicitly to the function;
2. the environment variables `WRDS_USERNAME` / `WRDS_PASSWORD`.  The project's
   `.env` file (git-ignored; `.env.example` is the committed template) is
   loaded into the environment at import time with `python-dotenv`;
3. the matching line in `~/.pgpass`, the standard PostgreSQL password file
   that the `wrds` package itself also reads.
The password is never printed or logged.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737       # not the default PostgreSQL port (5432) — WRDS uses 9737
WRDS_DBNAME = "wrds"   # one server can host several databases; we want "wrds"

# Locate `<project>/.env` relative to THIS file, not the current working
# directory: __file__ is .../project2/src/lead_lag/data/wrds_connection.py, so
# four `.parent` hops (parents[3]) give .../project2.  That way the secrets
# load whether you run from the project root, from src/, or from a notebook in
# notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load the secrets file into os.environ.  Two things worth knowing:
#  * variables ALREADY set in the shell are NOT overridden (override=False is
#    the default) — a shell export beats the file, the usual convention;
#  * a missing file is silently ignored, so the module still imports fine on a
#    machine that only has ~/.pgpass.
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Helper: read credentials from ~/.pgpass
# ---------------------------------------------------------------------------

def _credentials_from_pgpass(
    host: str, port: int, dbname: str
) -> tuple[str | None, str | None]:
    """Return (username, password) for `host:port:dbname` from ~/.pgpass.

    `~/.pgpass` is PostgreSQL's own password file.  Each line has five
    colon-separated fields:

        hostname:port:database:username:password

    and `*` in a field means "matches anything".  The file must have
    permission 600 (owner read/write only) or PostgreSQL tools refuse to use
    it.

    Returns (None, None) when the file is missing or no line matches, so the
    caller can fall through to "no credentials found" without special-casing.
    """
    pgpass = Path(os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))
    if not pgpass.exists():
        return None, None

    for line in pgpass.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):  # skip blanks and comments
            continue
        # Split into AT MOST 5 pieces: a password may itself contain ':' and
        # `split(":", 4)` keeps everything after the 4th colon intact.
        parts = line.split(":", 4)
        if len(parts) != 5:  # malformed line — ignore rather than crash
            continue
        f_host, f_port, f_db, f_user, f_pw = parts
        if (
            f_host in ("*", host)
            and f_port in ("*", str(port))
            and f_db in ("*", dbname)
        ):
            return f_user, f_pw
    return None, None


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------

def wrds_connection_on(
    username: str | None = None,
    password: str | None = None,
    host: str = WRDS_HOST,
    port: int = WRDS_PORT,
    dbname: str = WRDS_DBNAME,
    timeout: int = 10,
) -> bool:
    """Return True if we can log in to WRDS with these credentials and query.

    Args:
        username: WRDS username.  If None, taken from $WRDS_USERNAME (i.e. the
            .env file), then from ~/.pgpass.
        password: WRDS password.  Same fallback order as `username`.
        host, port, dbname: the PostgreSQL endpoint.  Defaults are the real
            WRDS server; override only for testing.
        timeout: seconds to wait for the network handshake before giving up.
            Without it an unreachable host can hang for minutes.

    Returns:
        True  -> network reachable, credentials accepted, `SELECT 1` returned 1.
        False -> some step failed; the reason is printed (never the password).
    """
    # ---- Step 1: resolve credentials -------------------------------------
    # Python's `or` returns the first "truthy" operand and stops evaluating, so
    #   explicit argument  >  environment / .env  >  ~/.pgpass
    # and an empty string ("") from an unfilled `.env` line counts as "not set".
    pg_user, pg_pw = _credentials_from_pgpass(host, port, dbname)
    username = username or os.environ.get("WRDS_USERNAME") or pg_user
    password = password or os.environ.get("WRDS_PASSWORD") or pg_pw
    if not username or not password:
        print(
            "WRDS connection: no username/password given, and none found in "
            ".env (WRDS_USERNAME / WRDS_PASSWORD) or ~/.pgpass"
        )
        return False

    # Declared before the try-block so the `finally` clause can refer to it
    # even if `psycopg2.connect` itself raised (then it is still None).
    conn = None
    try:
        # ---- Step 2: open the session -------------------------------------
        # `psycopg2.connect` does the whole handshake in one call: DNS, TCP,
        # SSL upgrade (WRDS refuses plain-text), then username/password.  Any
        # of those failing raises `psycopg2.OperationalError`, caught below.
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=username,
            password=password,
            sslmode="require",
            connect_timeout=timeout,
        )

        # ---- Step 3: prove the session works ------------------------------
        # `SELECT 1` touches no table, so it cannot fail because of
        # permissions or missing data — if it works, the session works.
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            # fetchone() returns the next row as a tuple, e.g. (1,), or None.
            row = cur.fetchone()
        if row is None:
            print("WRDS connection: logged in, but SELECT 1 returned no row")
            return False
        return row[0] == 1

    except psycopg2.OperationalError as exc:
        # Covers DNS failure, timeout, connection refused (firewall) and bad
        # credentials — the server's message says which ("PAM authentication
        # failed" = bad login).  The username is printed (useful for debugging
        # which login was tried); the password never is.
        print(f"WRDS connection failed for user '{username}': {str(exc).strip()}")
        return False

    except Exception as exc:  # noqa: BLE001 — anything unexpected
        print(f"WRDS connection check raised {type(exc).__name__}: {exc}")
        return False

    finally:
        # Runs however we left the try-block.  WRDS caps simultaneous sessions
        # per user, so leaking them would eventually lock you out.
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    # `python -m lead_lag.data.wrds_connection` prints True or False using the
    # .env / ~/.pgpass credentials.  Importing from a notebook does NOT run it.
    print(wrds_connection_on())
