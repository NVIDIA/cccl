import os
import sqlite3

import fpzip
import numpy as np
import pandas as pd

db_name = "cccl_meta_bench.db"

# PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


def get_postgres_config():
    """Get PostgreSQL configuration from environment variables."""
    if not POSTGRES_AVAILABLE:
        return None

    # Check if all required environment variables are set
    required_vars = [
        "CCCL_BENCH_PG_HOST",
        "CCCL_BENCH_PG_USER",
        "CCCL_BENCH_PG_DB",
        "CCCL_BENCH_PG_PASSWORD",
    ]
    config = {}

    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            return None  # Fall back to SQLite if any required var is missing
        config[var] = value

    # Optional port (default to 5432)
    config["CCCL_BENCH_PG_PORT"] = os.environ.get("CCCL_BENCH_PG_PORT", "5432")

    return config


def get_bench_table_name(subbench, algname):
    return "{}.{}".format(algname, subbench)


def blob_to_samples(blob):
    return np.squeeze(fpzip.decompress(blob))


class StorageBase:
    """Abstract base class for storage backends."""

    def connection(self):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def algnames(self):
        raise NotImplementedError

    def subbenches(self, algname):
        raise NotImplementedError

    def alg_to_df(self, algname, subbench):
        raise NotImplementedError

    def store_df(self, algname, df):
        raise NotImplementedError


class SQLiteStorage(StorageBase):
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def connection(self):
        return self.conn

    def exists(self):
        return os.path.exists(self.db_path)

    def algnames(self):
        with self.conn:
            rows = self.conn.execute(
                "SELECT DISTINCT algorithm FROM subbenches"
            ).fetchall()
            return [row[0] for row in rows]

    def subbenches(self, algname):
        with self.conn:
            rows = self.conn.execute(
                "SELECT DISTINCT bench FROM subbenches WHERE algorithm=?", (algname,)
            ).fetchall()
            return [row[0] for row in rows]

    def alg_to_df(self, algname, subbench):
        table = get_bench_table_name(subbench, algname)
        with self.conn:
            df = pd.read_sql_query('SELECT * FROM "{}"'.format(table), self.conn)
            df["samples"] = df["samples"].apply(blob_to_samples)

        return df

    def store_df(self, algname, df):
        df["samples"] = df["samples"].apply(fpzip.compress)
        df.to_sql(algname, self.conn, if_exists="replace", index=False)


class PostgreSQLConnectionWrapper:
    """Wrapper to make psycopg2 connection compatible with sqlite3 interface."""

    def __init__(self, pg_conn):
        self.pg_conn = pg_conn
        self.pg_conn.autocommit = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.pg_conn.commit()
        else:
            self.pg_conn.rollback()

    def execute(self, query, params=None):
        """Execute query with SQLite-style parameter substitution."""
        # Convert SQLite-style ? placeholders to PostgreSQL %s
        if params:
            query = query.replace("?", "%s")

        # Convert SQLite BLOB type to PostgreSQL BYTEA
        query = query.replace(" BLOB", " BYTEA")

        # Fix SQLite-style double-quoted string literals to PostgreSQL single quotes
        # This is a simple approach - in production you'd want a proper SQL parser
        import re

        # Match patterns like = "value" and convert to = 'value'
        query = re.sub(r'= "([^"]*)"', r"= '\1'", query)

        # Handle ON CONFLICT DO NOTHING (SQLite) -> ON CONFLICT DO NOTHING (PostgreSQL)
        # Both databases support this syntax, so no conversion needed

        cur = self.pg_conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return cur

    def commit(self):
        self.pg_conn.commit()

    def rollback(self):
        self.pg_conn.rollback()

    def close(self):
        self.pg_conn.close()


if POSTGRES_AVAILABLE:

    class PostgreSQLStorage(StorageBase):
        def __init__(self, config):
            self.config = config
            self.pg_conn = psycopg2.connect(
                host=config["CCCL_BENCH_PG_HOST"],
                port=config["CCCL_BENCH_PG_PORT"],
                database=config["CCCL_BENCH_PG_DB"],
                user=config["CCCL_BENCH_PG_USER"],
                password=config["CCCL_BENCH_PG_PASSWORD"],
            )
            self.conn = PostgreSQLConnectionWrapper(self.pg_conn)

        def connection(self):
            return self.conn

        def exists(self):
            # For PostgreSQL, check if the subbenches table exists
            with self.conn:
                cur = self.conn.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'subbenches'
                    );
                """)
                return cur.fetchone()[0]

        def algnames(self):
            with self.conn:
                cur = self.conn.execute("SELECT DISTINCT algorithm FROM subbenches")
                rows = cur.fetchall()
                return [row[0] for row in rows]

        def subbenches(self, algname):
            with self.conn:
                cur = self.conn.execute(
                    "SELECT DISTINCT bench FROM subbenches WHERE algorithm=?",
                    (algname,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]

        def alg_to_df(self, algname, subbench):
            table = get_bench_table_name(subbench, algname)
            with self.conn:
                # Use proper quoting for PostgreSQL
                query = 'SELECT * FROM "{}"'.format(table.replace('"', '""'))
                df = pd.read_sql_query(query, self.pg_conn)
                df["samples"] = df["samples"].apply(lambda x: blob_to_samples(bytes(x)))
            return df

        def store_df(self, algname, df):
            df["samples"] = df["samples"].apply(fpzip.compress)
            # For PostgreSQL, we need to use a different approach
            # as pandas doesn't support direct to_sql with psycopg2
            # We'll need to implement this separately or use SQLAlchemy
            raise NotImplementedError(
                "DataFrame storage for PostgreSQL not yet implemented"
            )
else:
    # Define a dummy class when psycopg2 is not available
    PostgreSQLStorage = None


class DualStorageWrapper:
    """Wrapper that writes to multiple storage backends."""

    def __init__(self, primary, secondary=None):
        self.primary = primary
        self.secondary = secondary
        self._primary_conn = None
        self._secondary_conn = None

    def connection(self):
        # Return a wrapper that forwards operations to both backends
        if not self._primary_conn:
            self._primary_conn = DualConnectionWrapper(
                self.primary.connection(),
                self.secondary.connection() if self.secondary else None,
            )
        return self._primary_conn

    def exists(self):
        # Check primary storage
        return self.primary.exists()

    def algnames(self):
        # Read from primary only
        return self.primary.algnames()

    def subbenches(self, algname):
        # Read from primary only
        return self.primary.subbenches(algname)

    def alg_to_df(self, algname, subbench):
        # Read from primary only
        return self.primary.alg_to_df(algname, subbench)

    def store_df(self, algname, df):
        # Write to both databases
        self.primary.store_df(algname, df)
        if self.secondary:
            try:
                self.secondary.store_df(algname, df)
            except Exception as e:
                print(f"Warning: Failed to write to secondary storage: {e}")


class DualCursorWrapper:
    """Wrapper for cursor results from dual storage."""

    def __init__(self, primary_cursor):
        self.primary_cursor = primary_cursor

    def fetchone(self):
        return self.primary_cursor.fetchone()

    def fetchall(self):
        return self.primary_cursor.fetchall()


class DualConnectionWrapper:
    """Wrapper that forwards connection operations to both backends."""

    def __init__(self, primary_conn, secondary_conn=None):
        self.primary_conn = primary_conn
        self.secondary_conn = secondary_conn

    def __enter__(self):
        # SQLite connections are their own context managers
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Commit or rollback based on exception
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def execute(self, query, params=None):
        # Execute on primary
        if params:
            primary_result = self.primary_conn.execute(query, params)
        else:
            primary_result = self.primary_conn.execute(query)

        # Also execute on secondary if available
        if self.secondary_conn:
            try:
                if params:
                    self.secondary_conn.execute(query, params)
                else:
                    self.secondary_conn.execute(query)
            except Exception:
                # Don't print warnings for every query, too noisy
                pass

        # Return a wrapper that delegates to the primary result
        return DualCursorWrapper(primary_result)

    def fetchone(self):
        # Delegate to primary connection
        return self.primary_conn.fetchone()

    def fetchall(self):
        # Delegate to primary connection
        return self.primary_conn.fetchall()

    def commit(self):
        self.primary_conn.commit()
        if self.secondary_conn:
            try:
                self.secondary_conn.commit()
            except Exception as e:
                print(f"Warning: Failed to commit to secondary storage: {e}")

    def rollback(self):
        self.primary_conn.rollback()
        if self.secondary_conn:
            try:
                self.secondary_conn.rollback()
            except Exception as e:
                print(f"Warning: Failed to rollback secondary storage: {e}")


class Storage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)

            # Always use SQLite as primary
            sqlite_storage = SQLiteStorage(db_name)

            # Try to add PostgreSQL as secondary if configured
            pg_config = get_postgres_config()
            pg_storage = None

            if pg_config and PostgreSQLStorage is not None:
                try:
                    pg_storage = PostgreSQLStorage(pg_config)
                    print(
                        "Using dual storage: SQLite (primary) + PostgreSQL (secondary)"
                    )
                except Exception as e:
                    print(f"Failed to connect to PostgreSQL: {e}")
                    print("Using SQLite only")

            # Create wrapper with SQLite as primary and PostgreSQL as optional secondary
            cls._instance.base = DualStorageWrapper(sqlite_storage, pg_storage)

        return cls._instance

    def connection(self):
        return self.base.connection()

    def exists(self):
        return self.base.exists()

    def algnames(self):
        return self.base.algnames()

    def alg_to_df(self, algname, subbench):
        return self.base.alg_to_df(algname, subbench)
