import os
import fpzip
import sqlite3
import numpy as np
import pandas as pd


db_name = "cub_bench_meta.db"


def blob_to_samples(blob):
    return np.squeeze(fpzip.decompress(blob))


class StorageBase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def connection(self):
        return self.conn

    def exists(self):
        return os.path.exists(db_name)

    def algnames(self):
        with self.conn:
            result = self.conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cub.bench.%';
            """).fetchall()

        algnames = [algname[0] for algname in result]
        return algnames

    def alg_to_df(self, algname):
        with self.conn:
            df = pd.read_sql_query("SELECT * FROM \"{}\"".format(algname), self.conn)
            df['samples'] = df['samples'].apply(blob_to_samples)

        return df
    
    def store_df(self, algname, df):
        df['samples'] = df['samples'].apply(fpzip.compress)
        df.to_sql(algname, self.conn, if_exists='replace', index=False)


class Storage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.base = StorageBase(db_name)
        return cls._instance

    def connection(self):
        return self.base.connection()

    def exists(self):
        return self.base.exists()

    def algnames(self):
        return self.base.algnames()

    def alg_to_df(self, algname):
        return self.base.alg_to_df(algname)
