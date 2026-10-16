"""SQLite BM25 检索与会话存储；原理来源和简化范围见 docs/retrieval.md。"""
import argparse
import json
from pathlib import Path
import re
import sqlite3

if __package__:
    from .retrieval_policy import content_query, relevant_text
else:
    from retrieval_policy import content_query, relevant_text


_WORDS = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")


def search_terms(text, *, indexing=False):
    """英文按词；中文用相邻双字，索引另存单字以支持单字查询。"""
    terms = []
    for word in _WORDS.findall(text.lower()):
        if "\u3400" <= word[0] <= "\u9fff":
            if indexing or len(word) == 1:
                terms.extend(word)
            terms.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            terms.append(word)
    return terms


def _session(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session 必须是非空字符串")
    return value


class RetrievalStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        try:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    name TEXT PRIMARY KEY, tokenizer TEXT
                );
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('knowledge', 'memory', 'turn')),
                    source TEXT NOT NULL, origin TEXT,
                    session TEXT REFERENCES sessions(name),
                    text TEXT NOT NULL, terms TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS entries_origin ON entries(origin);
                CREATE INDEX IF NOT EXISTS entries_session ON entries(session, kind, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(terms);
                CREATE TRIGGER IF NOT EXISTS entries_insert AFTER INSERT ON entries BEGIN
                    INSERT INTO search_index(rowid, terms) VALUES(new.id, new.terms);
                END;
                CREATE TRIGGER IF NOT EXISTS entries_delete AFTER DELETE ON entries BEGIN
                    DELETE FROM search_index WHERE rowid = old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS entries_update AFTER UPDATE ON entries BEGIN
                    DELETE FROM search_index WHERE rowid = old.id;
                    INSERT INTO search_index(rowid, terms) VALUES(new.id, new.terms);
                END;
                CREATE TABLE IF NOT EXISTS turns (
                    entry_id INTEGER PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
                    tokens TEXT NOT NULL, token_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_turns (
                    entry_id INTEGER PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
                    messages TEXT NOT NULL, token_count INTEGER NOT NULL
                );
            """)
        except sqlite3.OperationalError as exc:
            self.connection.close()
            raise RuntimeError("无法初始化检索数据库；需要支持 FTS5 的 SQLite") from exc

    def close(self):
        self.connection.close()

    def session_exists(self, name):
        return self.connection.execute('SELECT 1 FROM sessions WHERE name=?',
                                       (_session(name),)).fetchone() is not None

    def list_sessions(self, limit=20):
        rows = self.connection.execute(
            "SELECT s.name, COUNT(m.entry_id) AS turns FROM sessions s "
            "LEFT JOIN entries e ON e.session=s.name "
            "LEFT JOIN message_turns m ON m.entry_id=e.id "
            "GROUP BY s.name ORDER BY MAX(e.id) DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def _insert(self, kind, source, text, session=None, origin=None):
        cursor = self.connection.execute(
            "INSERT INTO entries(kind, source, text, session, origin, terms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (kind, source, text, session, origin,
             " ".join(search_terms(text, indexing=True))),
        )
        return cursor.lastrowid

    def ingest(self, path, chunk_chars=1200, overlap=200):
        """按文件绝对路径替换索引；重导入不会不断追加相同文档。"""
        if (type(chunk_chars) is not int or type(overlap) is not int
                or chunk_chars <= 0 or not 0 <= overlap < chunk_chars):
            raise ValueError("需要 chunk_chars > 0 且 0 <= overlap < chunk_chars")
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files = sorted({p.resolve() for p in path.rglob("*")
                            if p.is_file() and p.suffix.lower() in (".txt", ".md")})
        elif path.suffix.lower() in (".txt", ".md"):
            files = [path]
        else:
            raise ValueError("仅支持 UTF-8 .txt/.md 文件或包含它们的目录")
        count = 0
        # 文档与 FTS 触发器在同一事务中更新；读取失败时一起回滚。
        with self.connection:
            for file in files:
                origin = str(file)
                self.connection.execute(
                    "DELETE FROM entries WHERE kind = 'knowledge' AND origin = ?",
                    (origin,),
                )
                with file.open(encoding="utf-8") as stream:
                    chunk = stream.read(chunk_chars)
                    number = 1
                    while chunk:
                        if chunk.strip():
                            self._insert("knowledge", f"{origin}#chunk={number}",
                                         chunk.strip(), origin=origin)
                            count += 1
                        extra = stream.read(chunk_chars - overlap)
                        if not extra:
                            break
                        chunk = (chunk[-overlap:] if overlap else "") + extra
                        number += 1
        return count

    def search(self, query, limit=4, session=None, include_memory=True, *, include_history=True, include_knowledge=True):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit 必须是非负整数")
        if session is not None:
            _session(session)
        # 用户文本先分词再加引号；不会把 OR、冒号等当作用户提供的查询语法。
        terms = list(dict.fromkeys(search_terms(content_query(query))))[:128]
        if not terms or limit == 0:
            return []
        match = " OR ".join('"' + word.replace('"', '""') + '"' for word in terms)
        rows = self.connection.execute(
            "SELECT e.id, e.kind, e.source, e.text FROM search_index "
            "JOIN entries e ON e.id = search_index.rowid "
            "WHERE search_index MATCH ? AND "
            "((? AND e.kind = 'knowledge') OR (? AND e.session = ? "
            "AND (e.kind = 'memory' OR (? AND e.kind = 'turn')))) "
            "ORDER BY bm25(search_index), e.id DESC LIMIT ?",
            (match, bool(include_knowledge), bool(include_memory and session is not None), session, bool(include_history), max(limit, min(limit * 8, 1024))),
        )
        return [dict(row) for row in rows if relevant_text(query, row['text'])][:limit]

    def _ensure_session(self, session):
        self.connection.execute(
            "INSERT OR IGNORE INTO sessions(name) VALUES (?)", (_session(session),)
        )

    def remember(self, text, session):
        if not text.strip():
            raise ValueError("记忆内容不能为空")
        with self.connection:
            self._ensure_session(session)
            return self._insert("memory", f"用户显式记忆：{session}", text.strip(), session)

    def list_memories(self, session):
        rows = self.connection.execute(
            "SELECT id, kind, source, text FROM entries "
            "WHERE kind = 'memory' AND session = ? ORDER BY id", (_session(session),)
        )
        return [dict(row) for row in rows]

    def forget(self, id, session):
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM entries WHERE id = ? AND session = ? AND kind = 'memory'",
                (id, _session(session)),
            )
        return cursor.rowcount > 0

    def clear_session(self, session):
        with self.connection:
            self.connection.execute("DELETE FROM entries WHERE session = ?",
                                    (_session(session),))
            self.connection.execute("DELETE FROM sessions WHERE name = ?", (session,))

    def _identity(self, session, identity):
        _session(session)
        if not isinstance(identity, dict) or not identity.get("fingerprint"):
            raise ValueError("必须提供完整 tokenizer_identity")
        serialized = json.dumps(identity, ensure_ascii=True, sort_keys=True)
        row = self.connection.execute(
            "SELECT tokenizer FROM sessions WHERE name = ?", (session,)
        ).fetchone()
        if row and row["tokenizer"] is not None and row["tokenizer"] != serialized:
            raise ValueError("此会话对应的模型或分词器不匹配；请使用新会话，旧记录仍保留。")
        return serialized

    def save_turn(self, session, question, answer, token_ids, tokenizer_identity):
        ids = list(token_ids)
        vocab_size = tokenizer_identity.get("vocab_size", 0)
        if not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
            raise ValueError("token_ids 必须包含当前分词器词表内的整数编号")
        with self.connection:
            self._ensure_session(session)
            identity = self._identity(session, tokenizer_identity)
            self.connection.execute("UPDATE sessions SET tokenizer = ? WHERE name = ?",
                                    (identity, session))
            # 历史回答可能有错误，单独标记为对话记录，绝不自动写为事实记忆。
            entry_id = self._insert(
                "turn", f"历史对话（未经事实确认）：{session}",
                f"用户：{question}\n助手：{answer}", session,
            )
            self.connection.execute(
                "INSERT INTO turns(entry_id, tokens, token_count) VALUES (?, ?, ?)",
                (entry_id, json.dumps(ids, separators=(",", ":")), len(ids)),
            )

    def load_turns(self, session, tokenizer_identity, token_budget=8192):
        """只读取预算内最近的连续完整问答，返回顺序与实际对话一致。"""
        if type(token_budget) is not int or token_budget < 0:
            raise ValueError("token_budget 必须是非负整数")
        self._identity(session, tokenizer_identity)
        rows = self.connection.execute(
            "SELECT t.entry_id, t.token_count FROM turns t "
            "JOIN entries e ON e.id = t.entry_id "
            "WHERE e.session = ? ORDER BY e.id DESC", (session,),
        )
        kept = []
        for row in rows:
            if row["token_count"] > token_budget:
                break
            payload = self.connection.execute(
                "SELECT tokens FROM turns WHERE entry_id = ?", (row["entry_id"],)
            ).fetchone()
            kept.append(json.loads(payload["tokens"]))
            token_budget -= row["token_count"]
        kept.reverse()
        return kept

    def save_messages(self, session, question, answer, token_count, identity):
        """现成模型保存消息结构，再用其官方模板编码，不复用自研模型的角色编号。"""
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("token_count 必须是正整数")
        messages = [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]
        with self.connection:
            self._ensure_session(session)
            serialized = self._identity(session, identity)
            self.connection.execute("UPDATE sessions SET tokenizer = ? WHERE name = ?", (serialized, session))
            key = self._insert("turn", f"历史对话（未经事实确认）：{session}",
                               f"用户：{question}\n助手：{answer}", session)
            self.connection.execute("INSERT INTO message_turns VALUES (?, ?, ?)",
                                    (key, json.dumps(messages, ensure_ascii=False), token_count))

    def load_messages(self, session, identity, token_budget=8192):
        if type(token_budget) is not int or token_budget < 0:
            raise ValueError("token_budget 必须是非负整数")
        self._identity(session, identity)
        rows = self.connection.execute(
            "SELECT m.messages, m.token_count FROM message_turns m JOIN entries e ON e.id=m.entry_id "
            "WHERE e.session=? ORDER BY e.id DESC", (session,))
        turns = []
        for row in rows:
            if row["token_count"] > token_budget:
                break
            turns.append(json.loads(row["messages"]))
            token_budget -= row["token_count"]
        return [message for turn in reversed(turns) for message in turn]


def main():
    parser = argparse.ArgumentParser(description="本地文档检索和会话记忆管理")
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--chunk-chars", type=int, default=1200)
    ingest.add_argument("--overlap", type=int, default=200)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=4)
    search.add_argument("--session")
    for name in ("memories", "forget", "clear"):
        command = commands.add_parser(name)
        command.add_argument("--session", required=True)
        if name == "forget":
            command.add_argument("id", type=int)
    args = parser.parse_args()
    store = RetrievalStore(args.db)
    try:
        if args.command == "ingest":
            print(f"已写入 {store.ingest(args.path, args.chunk_chars, args.overlap)} 个文档片段")
        elif args.command == "search":
            print(json.dumps(store.search(args.query, args.limit, args.session),
                             ensure_ascii=False, indent=2))
        elif args.command == "memories":
            print(json.dumps(store.list_memories(args.session), ensure_ascii=False, indent=2))
        elif args.command == "forget":
            print("已删除" if store.forget(args.id, args.session) else "该会话中没有此记忆")
        else:
            store.clear_session(args.session)
            print("已清除此会话的历史和记忆")
    finally:
        store.close()


if __name__ == "__main__":
    main()
