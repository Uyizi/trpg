"""Immutable JSON snapshots plus verified content-addressed evidence."""
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from .domain import canonical, digest, utcnow

SCHEMA = '''
CREATE TABLE IF NOT EXISTS objects(kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL,
 hash TEXT UNIQUE NOT NULL,body TEXT NOT NULL,PRIMARY KEY(kind,id,revision));
CREATE TABLE IF NOT EXISTS artifacts(hash TEXT PRIMARY KEY,size INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS manifest_links(parent_id TEXT NOT NULL,child_id TEXT UNIQUE NOT NULL,
 phase TEXT NOT NULL,arm TEXT NOT NULL,position INTEGER NOT NULL,PRIMARY KEY(parent_id,position));
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,manifest_hash TEXT NOT NULL,state TEXT NOT NULL,
 cancel INTEGER NOT NULL DEFAULT 0,created TEXT NOT NULL,heartbeat TEXT,started REAL,
 requests INTEGER NOT NULL DEFAULT 0,tokens INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,position INTEGER NOT NULL,
 seed INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'NOT_RUN',outcome TEXT,grade TEXT,attempt_id TEXT,
 UNIQUE(job_id,position));
CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,observation_id TEXT NOT NULL,started TEXT NOT NULL,
 ended TEXT,state TEXT NOT NULL,request_hash TEXT NOT NULL,response_hash TEXT,error TEXT,
 reserved_tokens INTEGER NOT NULL,charged_tokens INTEGER,elapsed REAL);
'''


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class Store:
    def __init__(self, root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'artifacts').mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            # Freeze persisted conditions at the database boundary as well.
            db.executescript('''
CREATE TRIGGER IF NOT EXISTS immutable_object_update BEFORE UPDATE ON objects
 BEGIN SELECT RAISE(ABORT, 'immutable snapshot'); END;
CREATE TRIGGER IF NOT EXISTS immutable_object_delete BEFORE DELETE ON objects
 BEGIN SELECT RAISE(ABORT, 'immutable snapshot'); END;
''')

    def connect(self):
        db=sqlite3.connect(self.root/'debugger.sqlite',timeout=10,factory=ClosingConnection)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA foreign_keys=ON')
        return db

    @contextmanager
    def transaction(self):
        db=self.connect()
        try:
            db.execute('BEGIN IMMEDIATE'); yield db; db.commit()
        except BaseException:
            db.rollback(); raise
        finally: db.close()

    def put_object(self,kind,id,revision,body):
        text=canonical(body); h=digest(body)
        with self.transaction() as db:
            old=db.execute('SELECT hash FROM objects WHERE kind=? AND id=? AND revision=?',(kind,id,revision)).fetchone()
            if old and old['hash']!=h: raise ValueError('不可覆盖已有修订；请增加 revision 并保留父引用')
            if not old: db.execute('INSERT INTO objects VALUES(?,?,?,?,?)',(kind,id,revision,h,text))
        return h

    def object(self,h):
        with self.connect() as db: row=db.execute('SELECT body FROM objects WHERE hash=?',(h,)).fetchone()
        if not row: raise ValueError('快照不存在：'+h)
        obj=json.loads(row['body'])
        if digest(obj)!=h: raise ValueError('快照散列损坏')
        return obj

    def put_case(self,c):
        from .integrity import pool_from_rows,validate_case
        h=digest(c)
        with self.transaction() as db:
            pool=pool_from_rows(db.execute("SELECT * FROM objects WHERE kind='case'").fetchall())
            validate_case(c,pool)
            old=db.execute("SELECT hash FROM objects WHERE kind='case' AND id=? AND revision=?",(c['case_id'],c['revision'])).fetchone()
            if old and old['hash']!=h:raise ValueError('不可覆盖已有修订')
            if not old:db.execute('INSERT INTO objects VALUES(?,?,?,?,?)',('case',c['case_id'],c['revision'],h,canonical(c)))
        return h

    def case(self,id,revision=None):
        with self.connect() as db:
            sql='SELECT hash FROM objects WHERE kind=? AND id=?'
            args=['case',id]
            if revision is not None: sql+=' AND revision=?'; args.append(revision)
            row=db.execute(sql+' ORDER BY revision DESC LIMIT 1',args).fetchone()
        if not row: raise ValueError('案例不存在：'+id)
        return self.object(row['hash'])

    def artifact_path(self,h):
        if not re.fullmatch('[0-9a-f]{64}',h): raise ValueError('无效 artifact hash')
        return self.root/'artifacts'/h

    def artifact(self,data):
        h=hashlib.sha256(data).hexdigest(); dest=self.artifact_path(h)
        fd,tmp=tempfile.mkstemp(dir=dest.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,dest)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?)',(h,len(data)))
        return h

    def read_artifact(self,h):
        with self.connect() as db: row=db.execute('SELECT size FROM artifacts WHERE hash=?',(h,)).fetchone()
        if row is None: raise ValueError('证据未登记')
        try: data=self.artifact_path(h).read_bytes()
        except FileNotFoundError as exc: raise ValueError('证据文件缺失：'+h) from exc
        if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=h: raise ValueError('证据损坏：'+h)
        return data

    def rows(self,table):
        if table not in {'jobs','observations','attempts','objects','manifest_links'}: raise ValueError('未知表')
        with self.connect() as db: return [dict(r) for r in db.execute('SELECT * FROM '+table)]
