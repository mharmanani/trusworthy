import pymysql
import json


def connect():
    from dotenv import load_dotenv

    load_dotenv()

    import os

    cfg = {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    return pymysql.connect(**cfg, database="Exact")
