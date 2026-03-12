"""
Database initialization script for Omni-Help.

Creates the SQLite order management database at data/db/orders.db and
populates it with representative dummy data for local development and testing.

Tables created:
  - orders     : core order record (order_id, customer, status, total)
  - shipments  : logistics record linked to an order (tracking, carrier, ETA)

Run once from the repo root:
    python src/utils/init_db.py

Safe to re-run — uses CREATE TABLE IF NOT EXISTS and clears dummy rows first.
"""

import os
import sqlite3
from datetime import date, timedelta

DB_PATH = "./data/db/orders.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    customer_email  TEXT NOT NULL,
    status          TEXT NOT NULL,       -- 'Processing', 'Shipped', 'Delivered', 'Cancelled'
    total_amount    REAL NOT NULL,
    created_at      TEXT NOT NULL        -- ISO-8601 date string
);
"""

CREATE_SHIPMENTS = """
CREATE TABLE IF NOT EXISTS shipments (
    tracking_number TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    carrier         TEXT NOT NULL,       -- 'UPS', 'FedEx', 'USPS'
    eta             TEXT NOT NULL,       -- ISO-8601 date string
    shipped_at      TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------

today = date.today()

ORDERS = [
    ("ORD-1001", "alice@example.com",   "Shipped",    129.99, str(today - timedelta(days=3))),
    ("ORD-1002", "bob@example.com",     "Processing",  49.95, str(today - timedelta(days=1))),
    ("ORD-1003", "carol@example.com",   "Delivered",   89.00, str(today - timedelta(days=10))),
    ("ORD-1004", "dave@example.com",    "Cancelled",   15.50, str(today - timedelta(days=5))),
    ("ORD-1005", "alice@example.com",   "Shipped",    220.00, str(today - timedelta(days=2))),
]

SHIPMENTS = [
    # ORD-1001: shipped 3 days ago, arriving tomorrow
    ("1Z999AA1012345678", "ORD-1001", "UPS",   str(today + timedelta(days=1)), str(today - timedelta(days=3))),
    # ORD-1003: already delivered
    ("9400111899223186546057", "ORD-1003", "USPS", str(today - timedelta(days=2)), str(today - timedelta(days=8))),
    # ORD-1005: shipped 2 days ago, arriving in 2 days
    ("274899172137", "ORD-1005", "FedEx", str(today + timedelta(days=2)), str(today - timedelta(days=2))),
]


def init_db() -> None:
    """
    Create tables and populate with dummy data.

    Uses CREATE TABLE IF NOT EXISTS so it is safe to run against an
    existing database. Dummy rows are deleted before re-insertion to
    avoid duplicate-key errors on repeated runs.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create tables
    cur.execute(CREATE_ORDERS)
    cur.execute(CREATE_SHIPMENTS)

    # Clear existing dummy rows (idempotent re-runs)
    cur.execute("DELETE FROM shipments WHERE order_id IN (SELECT order_id FROM orders WHERE customer_email LIKE '%@example.com')")
    cur.execute("DELETE FROM orders WHERE customer_email LIKE '%@example.com'")

    # Insert dummy data
    cur.executemany(
        "INSERT INTO orders (order_id, customer_email, status, total_amount, created_at) VALUES (?,?,?,?,?)",
        ORDERS,
    )
    cur.executemany(
        "INSERT INTO shipments (tracking_number, order_id, carrier, eta, shipped_at) VALUES (?,?,?,?,?)",
        SHIPMENTS,
    )

    conn.commit()

    order_count = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    shipment_count = cur.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
    conn.close()

    print(f"\n✅ Database initialised at '{DB_PATH}'")
    print(f"   orders    : {order_count} rows")
    print(f"   shipments : {shipment_count} rows")
    print("\nSample data:")
    print("  ORD-1001 — alice@example.com — Shipped  — ETA tomorrow  (tracking: 1Z999AA1012345678)")
    print("  ORD-1002 — bob@example.com   — Processing (no shipment yet)")
    print("  ORD-1003 — carol@example.com — Delivered")
    print("  ORD-1004 — dave@example.com  — Cancelled")
    print("  ORD-1005 — alice@example.com — Shipped  — ETA in 2 days (tracking: 274899172137)")


if __name__ == "__main__":
    init_db()
