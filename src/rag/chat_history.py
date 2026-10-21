import os
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

class ChatHistoryManager:
    """
    Manages persistent user chat sessions stored on disk as JSON files.
    Allows loading, updating, listing, and deleting conversations just like ChatGPT.
    """
    _cached_chats: Optional[List[Dict[str, Any]]] = None
    _last_cache_time: float = 0.0
    _cached_archived_chats: Optional[List[Dict[str, Any]]] = None
    _last_archived_cache_time: float = 0.0
    _initial_sync_done: bool = False

    def __init__(self, storage_dir: Optional[Path | str] = None, *args, pg_db: Optional[Any] = None, **kwargs):
        if storage_dir is None:
            # Default to data/chats under project root safely
            _curr = Path(__file__).resolve()
            _root = None
            for _p in [_curr.parent, *_curr.parents]:
                if (_p / "data").exists() or (_p / "pyproject.toml").exists():
                    _root = _p
                    break
            project_root = _root or _curr.parents[2]
            storage_dir = project_root / "data" / "chats"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.pg_db = pg_db or kwargs.get("pg_db") or (args[0] if args else None)

        # Auto-connect to PostgreSQL / Supabase if environment variables or secrets exist
        keys_to_check = ["DATABASE_URL", "POSTGRES_HOST", "POSTGRES_URL", "database_url", "postgres_host", "postgres_url"]
        has_pg = any(os.getenv(k) for k in keys_to_check)
        if not has_pg:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    has_pg = any(k in st.secrets for k in keys_to_check + ["postgres", "supabase"])
            except Exception:
                pass

        if self.pg_db is None and has_pg:
            try:
                from rag.database import PostgresVectorDB
                cand = PostgresVectorDB()
                if cand.is_connected:
                    self.pg_db = cand
            except Exception:
                pass

        # Auto-sync existing local chats to cloud database so both sides see them (runs once per process)
        if not ChatHistoryManager._initial_sync_done and self.pg_db and self.pg_db.is_connected:
            ChatHistoryManager._initial_sync_done = True
            try:
                cloud_chats = self.pg_db.list_chat_sessions(include_archived=True)
                cloud_ids = {c["id"] for c in cloud_chats}
                for file_path in self.storage_dir.glob("*.json"):
                    if file_path.stem not in cloud_ids:
                        with open(file_path, "r", encoding="utf-8") as f:
                            c_data = json.load(f)
                        self.pg_db.save_chat_session(
                            c_data.get("id", file_path.stem),
                            c_data.get("title", "Saved Conversation"),
                            c_data.get("messages", []),
                            c_data.get("is_archived", False)
                        )
            except Exception as e_c_sync:
                print(f"[WARN] Failed to auto-sync local chats to Supabase: {e_c_sync}")
        
        # Ensure .gitkeep exists
        gitkeep = self.storage_dir / ".gitkeep"
        if not gitkeep.exists():
            try:
                gitkeep.touch()
            except Exception:
                pass

    @staticmethod
    def generate_chat_id() -> str:
        """Create a sortable, unique chat identifier."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        return f"chat_{timestamp}_{rand_suffix}"

    def list_chats(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """List all saved conversations, sorted with the most recent first."""
        now = time.time()
        if not include_archived and ChatHistoryManager._cached_chats is not None and (now - ChatHistoryManager._last_cache_time < 15):
            return ChatHistoryManager._cached_chats

        # 1. Check Cloud PostgreSQL / Supabase
        if self.pg_db and self.pg_db.is_connected:
            try:
                cloud_chats = self.pg_db.list_chat_sessions(include_archived=include_archived)
                if cloud_chats:
                    if not include_archived:
                        ChatHistoryManager._cached_chats = cloud_chats
                        ChatHistoryManager._last_cache_time = now
                    return cloud_chats
            except Exception as e:
                print(f"[WARN] Failed to list chats from cloud database: {e}")

        # 2. Local JSON Storage fallback
        chats = []
        if not self.storage_dir.exists():
            return chats

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    is_archived = data.get("is_archived", False)
                    if not include_archived and is_archived:
                        continue
                    chats.append({
                        "id": data.get("id", file_path.stem),
                        "title": data.get("title", "Untitled Chat"),
                        "created_at": data.get("created_at", file_path.stat().st_ctime),
                        "updated_at": data.get("updated_at", file_path.stat().st_mtime),
                        "message_count": len(data.get("messages", [])),
                        "is_archived": is_archived
                    })
            except Exception as e:
                print(f"[WARN] Failed to read chat history {file_path.name}: {e}")

        # Sort descending by updated_at
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        if not include_archived:
            ChatHistoryManager._cached_chats = chats
            ChatHistoryManager._last_cache_time = now
        return chats

    def list_archived_chats(self) -> List[Dict[str, Any]]:
        """List all archived conversations."""
        now = time.time()
        if ChatHistoryManager._cached_archived_chats is not None and (now - ChatHistoryManager._last_archived_cache_time < 15):
            return ChatHistoryManager._cached_archived_chats

        # 1. Check Cloud PostgreSQL / Supabase
        if self.pg_db and self.pg_db.is_connected:
            try:
                cloud_chats = self.pg_db.list_chat_sessions(include_archived=True)
                archived = [c for c in cloud_chats if c.get("is_archived")]
                ChatHistoryManager._cached_archived_chats = archived
                ChatHistoryManager._last_archived_cache_time = now
                return archived
            except Exception as e:
                print(f"[WARN] Failed to list archived chats from cloud database: {e}")

        archived = []
        if not self.storage_dir.exists():
            return archived

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("is_archived", False):
                        archived.append({
                            "id": data.get("id", file_path.stem),
                            "title": data.get("title", "Untitled Chat"),
                            "created_at": data.get("created_at", file_path.stat().st_ctime),
                            "updated_at": data.get("updated_at", file_path.stat().st_mtime),
                            "message_count": len(data.get("messages", []))
                        })
            except Exception as e:
                print(f"[WARN] Failed to read archived chat {file_path.name}: {e}")

        archived.sort(key=lambda x: x["updated_at"], reverse=True)
        ChatHistoryManager._cached_archived_chats = archived
        ChatHistoryManager._last_archived_cache_time = now
        return archived

    def archive_chat(self, chat_id: str) -> bool:
        """Mark a chat session as archived."""
        ChatHistoryManager._cached_chats = None
        ChatHistoryManager._cached_archived_chats = None
        if self.pg_db and self.pg_db.is_connected:
            self.pg_db.set_chat_archived(chat_id, True)

        file_path = self.storage_dir / f"{chat_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["is_archived"] = True
                data["updated_at"] = time.time()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"[ERROR] Failed to archive chat {chat_id}: {e}")
        return bool(self.pg_db and self.pg_db.is_connected)

    def unarchive_chat(self, chat_id: str) -> bool:
        """Restore an archived chat session back to active list."""
        ChatHistoryManager._cached_chats = None
        ChatHistoryManager._cached_archived_chats = None
        if self.pg_db and self.pg_db.is_connected:
            self.pg_db.set_chat_archived(chat_id, False)

        file_path = self.storage_dir / f"{chat_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["is_archived"] = False
                data["updated_at"] = time.time()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"[ERROR] Failed to unarchive chat {chat_id}: {e}")
        return bool(self.pg_db and self.pg_db.is_connected)

    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Load messages and metadata for a specific chat."""
        # 1. Try cloud database first
        if self.pg_db and self.pg_db.is_connected:
            try:
                cloud_data = self.pg_db.load_chat_session(chat_id)
                if cloud_data:
                    return cloud_data
            except Exception as e:
                print(f"[WARN] Failed to load chat from cloud database: {e}")

        # 2. Local fallback
        file_path = self.storage_dir / f"{chat_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load chat {chat_id}: {e}")
            return None

    def save_chat(
        self,
        chat_id: str,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> None:
        """Atomically persist or update conversation messages and metadata."""
        if not messages:
            return
        ChatHistoryManager._cached_chats = None
        ChatHistoryManager._cached_archived_chats = None

        file_path = self.storage_dir / f"{chat_id}.json"
        existing_data = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}

        # Derive title from first user message if not provided
        if not title:
            title = existing_data.get("title")
            if not title:
                for m in messages:
                    if m.get("role") == "user":
                        raw_title = m.get("content", "").strip()
                        # Clean title
                        cleaned = " ".join(raw_title.splitlines())
                        title = cleaned[:40] + ("..." if len(cleaned) > 40 else "")
                        break
            if not title:
                title = "New Conversation"

        # 1. Sync to Cloud PostgreSQL / Supabase
        if self.pg_db and self.pg_db.is_connected:
            try:
                self.pg_db.save_chat_session(chat_id, title, messages, is_archived=existing_data.get("is_archived", False))
            except Exception as e_pg_save:
                print(f"[WARN] Failed to mirror chat to cloud database: {e_pg_save}")

        # 2. Sync to local JSON
        now = time.time()
        chat_payload = {
            "id": chat_id,
            "title": title,
            "created_at": existing_data.get("created_at", now),
            "updated_at": now,
            "messages": messages
        }

        temp_path = self.storage_dir / f"{chat_id}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(chat_payload, f, ensure_ascii=False, indent=2)
            temp_path.replace(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to save chat {chat_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a saved chat history file."""
        ChatHistoryManager._cached_chats = None
        ChatHistoryManager._cached_archived_chats = None
        if self.pg_db and self.pg_db.is_connected:
            self.pg_db.delete_chat_session(chat_id)

        file_path = self.storage_dir / f"{chat_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"[ERROR] Failed to delete chat {chat_id}: {e}")
        return bool(self.pg_db and self.pg_db.is_connected)
