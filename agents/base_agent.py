"""
agents/base_agent.py
====================
Foundational parent class for every agent in the Multi-Agent Dataset Builder.

Architecture Overview
---------------------
All agents communicate through an asynchronous *state bus* — a shared directory
containing JSONL / JSON files that act as queues and result stores. Each agent:

  1. Reads work items from an *input queue* JSONL file.
  2. Processes them independently (no direct agent-to-agent calls).
  3. Writes results or logs to *output* JSONL / JSON files.

This decoupling lets agents run concurrently, be retried independently, and
added or replaced without touching any other agent's code.

Inheritance Pattern
-------------------
    class QueryArchitectAgent(BaseAgent):
        async def run(self):
            items = self.read_input_queue("query_requests.jsonl")
            for item in items:
                result = await self._build_query(item)
                self.write_output("search_queries.jsonl", result)
                self.log_status("build_query", "success", {"topic": item.get("topic")})
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

from tools.llm_client import LLMClient
from tools.console_setup import console


class BaseAgent:
    """
    Abstract base class that all specialised agents inherit from.

    Provides:
      • Shared configuration / path setup
      • Thread-safe JSONL / JSON file I/O
      • Structured logging to agent_logs.jsonl
      • An async `run()` stub that subclasses must implement
    """

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, agent_id: str, config: dict):
        """
        Parameters
        ----------
        agent_id : str
            A unique, human-readable identifier for this agent instance
            (e.g. "query_architect", "ingestion_agent").  Used in logs.

        config : dict
            Must contain at minimum:
              - "state_dir" (str | Path): root directory shared by all agents
                for reading/writing state files.
            Any additional keys are available via ``self.config``.
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string.")

        if "state_dir" not in config:
            raise KeyError("config must include a 'state_dir' key.")

        self.agent_id: str  = agent_id
        self.config:   dict = config

        # All queue / result files live under this shared directory.
        self.state_dir: Path = Path(config["state_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Shared LLM client — subclasses call self.llm.complete() etc.
        self.llm: LLMClient = LLMClient()

        console.print(
            f"[dim]  ✓ {self.agent_id} initialised "
            f"(state_dir={self.state_dir})[/dim]"
        )

    # ------------------------------------------------------------------ #
    #  State Reader                                                        #
    # ------------------------------------------------------------------ #

    def read_input_queue(self, filename: str) -> list[dict]:
        """
        Read a JSONL file from the shared state directory and return its
        contents as a list of parsed dictionaries.

        Each non-blank line in the file is expected to be a valid JSON object.

        Parameters
        ----------
        filename : str
            Basename of the JSONL file (e.g. "search_queries.jsonl").
            The full path will be ``self.state_dir / filename``.

        Returns
        -------
        list[dict]
            Parsed records.  Returns an empty list when:
              • The file does not exist yet (expected during cold starts).
              • The file is empty or contains only blank lines.
              • A line fails JSON parsing (that line is skipped with a warning).
        """
        filepath = self.state_dir / filename

        if not filepath.exists():
            # Normal during cold starts — the upstream agent has not yet
            # produced any output.
            console.print(
                f"[dim yellow]  ⚠ {self.agent_id}: queue file not found "
                f"(will retry when available): {filepath.name}[/dim yellow]"
            )
            return []

        records: list[dict] = []
        with filepath.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue  # skip blank lines silently
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        # Each JSONL line should be an object, not an array or scalar.
                        console.print(
                            f"[yellow]  ⚠ {self.agent_id}: line {line_no} in "
                            f"{filepath.name} is not a JSON object — skipped.[/yellow]"
                        )
                except json.JSONDecodeError as exc:
                    console.print(
                        f"[yellow]  ⚠ {self.agent_id}: JSON parse error on "
                        f"line {line_no} of {filepath.name}: {exc} — skipped.[/yellow]"
                    )

        return records

    # ------------------------------------------------------------------ #
    #  State Writer                                                        #
    # ------------------------------------------------------------------ #

    def write_output(
        self,
        filename: str,
        data: Union[dict, list],
    ) -> None:
        """
        Write data to a file inside ``self.state_dir``.

        Behaviour is determined by the file extension:

        *.jsonl*  — **Append** mode.  Each dictionary in ``data`` is written
                    as a separate JSON line.  Supports both a single ``dict``
                    and a ``list[dict]`` as input.

        *.json*   — **Overwrite** mode.  The entire ``data`` object is
                    serialised and the file is replaced atomically-ish
                    (write → rename).  This avoids leaving a half-written
                    snapshot on disk.

        Parameters
        ----------
        filename : str
            Target file basename (e.g. "results.jsonl" or "summary.json").
        data : dict | list
            Payload to write.

        Raises
        ------
        ValueError
            If the file extension is neither .jsonl nor .json.
        TypeError
            If ``data`` is of an unsupported type for the chosen extension.
        """
        filepath = self.state_dir / filename

        # Ensure the parent directory exists (handles subdirectory filenames).
        filepath.parent.mkdir(parents=True, exist_ok=True)

        suffix = filepath.suffix.lower()

        if suffix == ".jsonl":
            # Normalise: always work with a list of dicts for uniformity.
            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                raise TypeError(
                    f"{self.agent_id}.write_output: for .jsonl files, "
                    f"'data' must be a dict or list[dict], got {type(data).__name__}."
                )

            with filepath.open("a", encoding="utf-8") as fh:
                for record in records:
                    if not isinstance(record, dict):
                        console.print(
                            f"[yellow]  ⚠ {self.agent_id}: non-dict item in "
                            f"list skipped during write to {filepath.name}.[/yellow]"
                        )
                        continue
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        elif suffix == ".json":
            # Write to a temp file first, then rename — prevents a half-written
            # file from being read by another agent if a crash occurs mid-write.
            tmp_path = filepath.with_suffix(".json.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                tmp_path.replace(filepath)   # atomic on POSIX; best-effort on Windows
            except Exception:
                # Clean up the temp file if the rename fails.
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise

        else:
            raise ValueError(
                f"{self.agent_id}.write_output: unsupported extension '{suffix}'. "
                "Only '.jsonl' (append) and '.json' (overwrite) are supported."
            )

    # ------------------------------------------------------------------ #
    #  Standardised Logging                                                #
    # ------------------------------------------------------------------ #

    def log_status(
        self,
        stage: str,
        status: str,
        details: dict,
    ) -> None:
        """
        Append a structured log entry to ``agent_logs.jsonl``.

        Log entry schema
        ----------------
        {
            "ts":       <float>  Unix timestamp (UTC),
            "agent_id": <str>    e.g. "ingestion_agent",
            "stage":    <str>    e.g. "fetch_pdf",
            "status":   <str>    e.g. "success" | "error" | "skip",
            "details":  <dict>   arbitrary caller-supplied context
        }

        Parameters
        ----------
        stage : str
            Short identifier for the processing stage being logged
            (e.g. "fetch_html", "extract_rows", "validate_schema").
        status : str
            Outcome of the stage.  Recommended values:
            "success", "error", "warning", "skip", "retry".
        details : dict
            Any additional key/value pairs relevant to the stage
            (URLs, row counts, error messages, durations, etc.).
        """
        log_entry: dict = {
            "ts":       time.time(),          # seconds since epoch (UTC)
            "agent_id": self.agent_id,
            "stage":    stage,
            "status":   status,
            "details":  details if isinstance(details, dict) else {"raw": str(details)},
        }
        self.write_output("agent_logs.jsonl", log_entry)

    # ------------------------------------------------------------------ #
    #  Execution Stub                                                      #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """
        Entry point for the agent's main work loop.

        Every concrete subclass **must** override this method.
        The orchestrator (or an asyncio task runner) calls ``await agent.run()``
        to start the agent.

        Raises
        ------
        NotImplementedError
            Always — this base implementation is intentionally left empty.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' must implement the async `run()` method."
        )
