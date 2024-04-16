import os
import fpzip
import sqlite3
import numpy as np
import pandas as pd


db_name = "cccl_meta_bench.db"


def get_bench_table_name(subbench, algname):
    return "{}.{}".format(algname, subbench)


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
            rows = self.conn.execute('SELECT DISTINCT algorithm FROM subbenches').fetchall()
            return [row[0] for row in rows]

    def subbenches(self, algname):
        with self.conn:
            rows = self.conn.execute('SELECT DISTINCT bench FROM subbenches WHERE algorithm=?', (algname,)).fetchall()
            return [row[0] for row in rows]

    def alg_to_df(self, algname, subbench):
        table = get_bench_table_name(subbench, algname)
        with self.conn:
            df = pd.read_sql_query("SELECT * FROM \"{}\"".format(table), self.conn)
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

    def alg_to_df(self, algname, subbench):
        return self.base.alg_to_df(algname, subbench)
