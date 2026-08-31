# Archived scripts

Exploratory work, kept for reference but not part of the running system.
Nothing in `src/`, `dashboard/`, or `scripts/` imports from this directory.

## Postgres migration (abandoned 2026-08-31)

SQLite is the authoritative store. These were an unfinished attempt to move to
Postgres; the migration was never completed and nothing depended on it.

- `sql_agent.py` — LangChain SQL agent for natural-language queries over the
  predictions table. Requires Postgres, `langchain-anthropic`, and
  `langchain-community`. Note the model id is stale (`claude-sonnet-4-5`);
  use `claude-opus-5` if you revive this.
- `migrate_data.py` — one-shot SQLite -> Postgres copy. Its INSERT references
  `delta_predicted` and `outcome_timestamp`, which never existed in the SQLite
  schema, so those columns always copied as NULL.
- `test_pg.py` — connection smoke test against localhost:5433.

All three hardcoded `password='mypassword'` as a default. If you revive them,
read the password from the environment instead.

## Superseded

- `check_db.py` — table/row-count dump with a hardcoded absolute path.
  Replaced by `scripts/db_health.py`, which resolves the path through
  `src/config.py` and also reports liveness.
