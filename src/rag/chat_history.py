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
    _cached_chats: Dict[str, Any] = {}
    _last_cache_time: Dict[str, float] = {}
    _cached_archived_chats: Dict[str, Any] = {}
    _last_archived_cache_time: Dict[str, float] = {}
    _initial_sync_done: bool = False

    def __init__(self, storage_dir: Optional[Path | str] = None, *args, pg_db: Optional[Any] = None, user_id: Optional[str] = None, **kwargs):
        # Gracefully handle when user_id is passed as first positional argument
        if isinstance(storage_dir, str) and ("@" in storage_dir or storage_dir.startswith("user_")):
            if not user_id:
                user_id = storage_dir
            storage_dir = None

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
        self.user_id = user_id or kwargs.get("user_id") or "default_user"

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
                try:
                    from rag.database import PostgresVectorDB
                except ImportError:
                    from src.rag.database import PostgresVectorDB
                cand = PostgresVectorDB()
                if cand.is_connected:
                    self.pg_db = cand
            except Exception:
                pass

        # Auto-sync existing local chats to cloud database so both sides see them
        if not hasattr(ChatHistoryManager, "_synced_keys"):
            ChatHistoryManager._synced_keys = set()
        sync_key = f"{self.user_id}_{id(self.pg_db)}"
        if sync_key not in ChatHistoryManager._synced_keys and self.pg_db and self.pg_db.is_connected:
            ChatHistoryManager._synced_keys.add(sync_key)
            try:
                cloud_chats = self.pg_db.list_chat_sessions(include_archived=True)
                cloud_ids = {c["id"] for c in cloud_chats}
                for file_path in self.storage_dir.glob("*.json"):
                    if file_path.stem not in cloud_ids:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                c_data = json.load(f)
                            target_uid = c_data.get("user_id")
                            if not target_uid or target_uid == "default_user":
                                target_uid = self.user_id or "default_user"
                            self.pg_db.save_chat_session(
                                c_data.get("id", file_path.stem),
                                c_data.get("title", "Saved Conversation"),
                                c_data.get("messages", []),
                                is_archived=c_data.get("is_archived", False),
                                user_id=target_uid
                            )
                        except Exception:
                            continue
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
        """List all saved conversations, combining cloud database and local files with seamless deduplication."""
        cache_key = str(self.user_id or "default_user").strip().lower()
        now = time.time()
        if not include_archived and cache_key in ChatHistoryManager._cached_chats and (now - ChatHistoryManager._last_cache_time.get(cache_key, 0) < 5):
            return ChatHistoryManager._cached_chats[cache_key]

        clean_uid = str(self.user_id or "").strip().lower()

        # 1. Fetch from Cloud PostgreSQL / Supabase
        cloud_chats = []
        if self.pg_db and self.pg_db.is_connected:
            try:
                query_uid = None if (not clean_uid or clean_uid == "default_user") else self.user_id
                c_list = self.pg_db.list_chat_sessions(include_archived=include_archived, user_id=query_uid)
                if c_list:
                    cloud_chats = c_list
            except Exception as e:
                print(f"[WARN] Failed to list chats from cloud database: {e}")

        # 2. Fetch from Local JSON Storage
        local_chats = []
        if self.storage_dir.exists():
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        file_user_id = str(data.get("user_id", "")).strip().lower()
                        if clean_uid and clean_uid != "default_user" and file_user_id and file_user_id != "default_user" and file_user_id != clean_uid:
                            continue
                        is_archived = data.get("is_archived", False)
                        if not include_archived and is_archived:
                            continue
                        local_chats.append({
                            "id": data.get("id", file_path.stem),
                            "title": data.get("title", "Untitled Chat"),
                            "created_at": data.get("created_at", file_path.stat().st_ctime),
                            "updated_at": data.get("updated_at", file_path.stat().st_mtime),
                            "message_count": len(data.get("messages", [])),
                            "is_archived": is_archived,
                            "user_id": data.get("user_id")
                        })
                except Exception as e:
                    print(f"[WARN] Failed to read chat history {file_path.name}: {e}")

        # 3. Seamlessly Merge Cloud & Local (deduplicate by id, keep newest updated_at)
        by_id: Dict[str, Dict[str, Any]] = {}
        for c in cloud_chats + local_chats:
            cid = c["id"]
            if cid not in by_id or c.get("updated_at", 0) > by_id[cid].get("updated_at", 0):
                by_id[cid] = c

        all_chats = list(by_id.values())
        all_chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

        if not include_archived:
            ChatHistoryManager._cached_chats[cache_key] = all_chats
            ChatHistoryManager._last_cache_time[cache_key] = now
        return all_chats

    def list_archived_chats(self) -> List[Dict[str, Any]]:
        """List all archived conversations combining cloud and local storage."""
        cache_key = str(self.user_id or "default_user").strip().lower()
        now = time.time()
        if cache_key in ChatHistoryManager._cached_archived_chats and (now - ChatHistoryManager._last_archived_cache_time.get(cache_key, 0) < 5):
            return ChatHistoryManager._cached_archived_chats[cache_key]

        clean_uid = str(self.user_id or "").strip().lower()

        # 1. Fetch from Cloud PostgreSQL / Supabase
        cloud_archived = []
        if self.pg_db and self.pg_db.is_connected:
            try:
                query_uid = None if (not clean_uid or clean_uid == "default_user") else self.user_id
                cloud_chats = self.pg_db.list_chat_sessions(include_archived=True, user_id=query_uid)
                cloud_archived = [c for c in cloud_chats if c.get("is_archived")]
            except Exception as e:
                print(f"[WARN] Failed to list archived chats from cloud database: {e}")

        # 2. Fetch from Local JSON Storage
        local_archived = []
        if self.storage_dir.exists():
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        file_user_id = str(data.get("user_id", "")).strip().lower()
                        if clean_uid and clean_uid != "default_user" and file_user_id and file_user_id != "default_user" and file_user_id != clean_uid:
                            continue
                        if data.get("is_archived", False):
                            local_archived.append({
                                "id": data.get("id", file_path.stem),
                                "title": data.get("title", "Untitled Chat"),
                                "created_at": data.get("created_at", file_path.stat().st_ctime),
                                "updated_at": data.get("updated_at", file_path.stat().st_mtime),
                                "message_count": len(data.get("messages", [])),
                                "user_id": data.get("user_id")
                            })
                except Exception as e:
                    print(f"[WARN] Failed to read archived chat {file_path.name}: {e}")

        # 3. Merge & Deduplicate
        by_id: Dict[str, Dict[str, Any]] = {}
        for c in cloud_archived + local_archived:
            cid = c["id"]
            if cid not in by_id or c.get("updated_at", 0) > by_id[cid].get("updated_at", 0):
                by_id[cid] = c

        archived = list(by_id.values())
        archived.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

        ChatHistoryManager._cached_archived_chats[cache_key] = archived
        ChatHistoryManager._last_archived_cache_time[cache_key] = now
        return archived

    def archive_chat(self, chat_id: str) -> bool:
        """Mark a chat session as archived."""
        clean_uid = str(self.user_id or "default_user").strip().lower()
        ChatHistoryManager._cached_chats.pop(clean_uid, None)
        ChatHistoryManager._cached_archived_chats.pop(clean_uid, None)
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
        clean_uid = str(self.user_id or "default_user").strip().lower()
        ChatHistoryManager._cached_chats.pop(clean_uid, None)
        ChatHistoryManager._cached_archived_chats.pop(clean_uid, None)
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
        """Load messages and metadata for a specific chat from local or cloud storage."""
        clean_uid = str(self.user_id or "").strip().lower()
        local_data = None
        file_path = self.storage_dir / f"{chat_id}.json"

        # 1. Local storage check
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_user_id = str(data.get("user_id", "")).strip().lower()
                    if not clean_uid or clean_uid == "default_user" or not file_user_id or file_user_id == "default_user" or file_user_id == clean_uid:
                        local_data = data
            except Exception as e:
                print(f"[WARN] Failed to read local chat {chat_id}: {e}")

        # 2. Cloud database check
        cloud_data = None
        if self.pg_db and self.pg_db.is_connected:
            try:
                query_uid = None if (not clean_uid or clean_uid == "default_user") else self.user_id
                cloud_data = self.pg_db.load_chat_session(chat_id, user_id=query_uid)
                if not cloud_data and query_uid is not None:
                    cloud_data = self.pg_db.load_chat_session(chat_id, user_id=None)
            except Exception as e:
                print(f"[WARN] Failed to load chat from cloud database: {e}")

        # 3. Synchronize: pick version with most messages or latest updates
        if cloud_data and local_data:
            c_msgs = len(cloud_data.get("messages", []))
            l_msgs = len(local_data.get("messages", []))
            best_data = cloud_data if c_msgs >= l_msgs else local_data
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(best_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return best_data
        elif cloud_data:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cloud_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return cloud_data
        elif local_data:
            return local_data

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
        ChatHistoryManager._cached_chats.pop(self.user_id, None)
        ChatHistoryManager._cached_archived_chats.pop(self.user_id, None)

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
                self.pg_db.save_chat_session(
                    chat_id,
                    title,
                    messages,
                    is_archived=existing_data.get("is_archived", False),
                    user_id=self.user_id
                )
            except Exception as e_pg_save:
                print(f"[WARN] Failed to mirror chat to cloud database: {e_pg_save}")

        # 2. Sync to local JSON
        now = time.time()
        chat_payload = {
            "id": chat_id,
            "title": title,
            "created_at": existing_data.get("created_at", now),
            "updated_at": now,
            "messages": messages,
            "user_id": self.user_id
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
        """Delete a saved chat history file with strict user isolation."""
        ChatHistoryManager._cached_chats.pop(self.user_id, None)
        ChatHistoryManager._cached_archived_chats.pop(self.user_id, None)
        if self.pg_db and self.pg_db.is_connected:
            self.pg_db.delete_chat_session(chat_id, user_id=self.user_id)

        file_path = self.storage_dir / f"{chat_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"[ERROR] Failed to delete chat {chat_id}: {e}")
        return bool(self.pg_db and self.pg_db.is_connected)
