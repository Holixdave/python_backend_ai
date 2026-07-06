import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple


class UserDocManager:
    """
    Manages per-user persistent documents. Each user has a directory
    `/home/$userid/doc/` where they can store files (SVG, HTML, Markdown,
    images, etc.). Every file can have a hint block at the top:

    Markdown example:
    # @hint: "My Recipe Collection"
    # @tags: cooking, nigerian, spicy
    # @date: 2026-07-06

    HTML example:
    <!-- @hint: "Dashboard" -->
    <!-- @tags: analytics, live -->
    <html>...

    The manager indexes these hints so users can say "remember my recipe"
    and the AI can find it without relying on chat history.
    """

    BASE_DIR = Path("/home")
    DOCS_SUBDIR = "doc"
    METADATA_FILENAME = ".hints.json"  # Per-user index of all docs

    def __init__(self, userid: str):
        self.userid = userid
        self.user_dir = self.BASE_DIR / userid / self.DOCS_SUBDIR
        self.metadata_file = self.user_dir / self.METADATA_FILENAME
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create user doc directory if it doesn't exist."""
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def save_doc(
        self,
        filename: str,
        content: str,
        hint: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Save a file to the user's doc directory. If hint is provided, it's
        prepended to the file. Returns the saved doc's full metadata.

        Args:
            filename: e.g. "recipe.md", "dashboard.html", "logo.svg"
            content: file content (SVG, HTML, Markdown, etc.)
            hint: display name for this doc, e.g. "My Recipe Collection"
            tags: list of searchable tags, e.g. ["cooking", "nigerian"]
            metadata: extra metadata dict (optional)

        Returns:
            {"id": "...", "filename": "...", "hint": "...", "tags": [...], "date": "...", "size": ...}
        """
        if not filename:
            raise ValueError("filename cannot be empty")

        filepath = self.user_dir / filename
        date_str = datetime.now().isoformat()

        # Build hint block to prepend to content (if hint is provided)
        hint_block = ""
        if hint:
            hint_block = self._build_hint_block(hint, tags, date_str)
            content = hint_block + "\n" + content

        # Write file
        try:
            filepath.write_text(content, encoding="utf-8")
        except Exception as e:
            raise IOError(f"Failed to write {filename}: {e}")

        # Update metadata index
        doc_id = filename  # use filename as ID for now
        doc_meta = {
            "id": doc_id,
            "filename": filename,
            "hint": hint or filename,
            "tags": tags or [],
            "date": date_str,
            "size": len(content),
            **(metadata or {}),
        }
        self._update_metadata(doc_id, doc_meta)

        return doc_meta

    def _build_hint_block(self, hint: str, tags: Optional[List[str]], date_str: str) -> str:
        """
        Build a hint block to prepend. Auto-detects file type from context.
        For now, defaults to Markdown comment style.
        """
        tags_str = ", ".join(tags) if tags else ""
        return f"# @hint: \"{hint}\"\n# @tags: {tags_str}\n# @date: {date_str}"

    def search_by_hint(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search user's docs by hint/tag/filename. Returns matching docs
        in order of relevance (exact matches first, then partial matches).

        Args:
            query: search string, e.g. "recipe", "nigerian", "dashboard"
            limit: max results to return

        Returns:
            List of matching doc metadata dicts
        """
        if not self.metadata_file.exists():
            return []

        try:
            metadata = json.loads(self.metadata_file.read_text())
        except Exception as e:
            print(f"[UserDocManager] Failed to read metadata: {e}")
            return []

        query_lower = query.lower()
        matches = []

        for doc_id, doc_meta in metadata.items():
            score = 0
            hint = (doc_meta.get("hint") or "").lower()
            tags = [t.lower() for t in doc_meta.get("tags", [])]
            filename = doc_meta.get("filename", "").lower()

            # Exact match on hint: highest score
            if hint == query_lower:
                score = 100
            # Hint contains query: high score
            elif query_lower in hint:
                score = 80
            # Any tag contains query
            elif any(query_lower in tag for tag in tags):
                score = 60
            # Filename contains query
            elif query_lower in filename:
                score = 40
            # Partial match anywhere
            elif any(query_lower in part for part in [hint, filename]):
                score = 20

            if score > 0:
                matches.append((score, doc_meta))

        # Sort by score descending, return top limit
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def list_all_docs(self) -> List[Dict]:
        """
        List all docs for this user (with metadata, no content).

        Returns:
            List of doc metadata dicts
        """
        if not self.metadata_file.exists():
            return []

        try:
            metadata = json.loads(self.metadata_file.read_text())
            return list(metadata.values())
        except Exception as e:
            print(f"[UserDocManager] Failed to read metadata: {e}")
            return []

    def get_doc(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific doc by ID (filename). Returns metadata + content.

        Args:
            doc_id: e.g. "recipe.md"

        Returns:
            {"id": "...", "filename": "...", "hint": "...", "content": "...", "size": ...}
            or None if not found
        """
        filepath = self.user_dir / doc_id
        if not filepath.exists():
            return None

        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[UserDocManager] Failed to read {doc_id}: {e}")
            return None

        # Get metadata
        if self.metadata_file.exists():
            try:
                metadata = json.loads(self.metadata_file.read_text())
                doc_meta = metadata.get(doc_id, {})
            except Exception:
                doc_meta = {}
        else:
            doc_meta = {}

        return {
            "id": doc_id,
            "filename": doc_id,
            "content": content,
            "size": len(content),
            **doc_meta,
        }

    def _update_metadata(self, doc_id: str, doc_meta: Dict):
        """
        Update the metadata index file (`.hints.json`) with a doc's metadata.
        """
        metadata = {}
        if self.metadata_file.exists():
            try:
                metadata = json.loads(self.metadata_file.read_text())
            except Exception:
                pass

        metadata[doc_id] = doc_meta

        try:
            self.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[UserDocManager] Failed to update metadata: {e}")

    def delete_doc(self, doc_id: str) -> bool:
        """
        Delete a doc by ID (filename).

        Returns:
            True if deleted, False if not found or error
        """
        filepath = self.user_dir / doc_id
        if not filepath.exists():
            return False

        try:
            filepath.unlink()
            # Remove from metadata
            if self.metadata_file.exists():
                metadata = json.loads(self.metadata_file.read_text())
                metadata.pop(doc_id, None)
                self.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            print(f"[UserDocManager] Failed to delete {doc_id}: {e}")
            return False

    def parse_hints_from_content(self, content: str) -> Tuple[Optional[str], List[str]]:
        """
        Extract hint and tags from file content (first 5 lines).
        Handles both Markdown and HTML comment styles.

        Returns:
            (hint_string, tags_list)
        """
        lines = content.split("\n")[:5]
        hint = None
        tags = []

        for line in lines:
            # Markdown: # @hint: "My Title"
            hint_match = re.search(r'@hint:\s*["\']?([^"\']+)["\']?', line)
            if hint_match:
                hint = hint_match.group(1).strip()

            # @tags: tag1, tag2, tag3
            tags_match = re.search(r'@tags:\s*([^#\n]+)', line)
            if tags_match:
                tags = [t.strip() for t in tags_match.group(1).split(",")]

        return hint, tags
