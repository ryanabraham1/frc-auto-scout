# Connect the scouting app to a computer running Auto-Scout

Target app: [ryanabraham1/scouting-app](https://github.com/ryanabraham1/scouting-app),
inspected at `ca5b0afd42450906e184e9ffd6a6aa8d9477c105`.

The computer runs the video processing. The app's backend sends a list of matches,
then polls for progress/results. Closing the phone browser does not stop processing.
The computer must be awake, online, and running the integration service.

```text
Scouting app: select matches → Run Auto-Scout
        ↓ existing app backend (keeps the API token secret)
        ↓ HTTPS tunnel → 127.0.0.1:8001
Computer: persistent queue → video processing → per-team results
        ↑ backend fetches results → stores them → app displays them
```

## What is ready

- `autoscout serve-integration`: separate authenticated API, bound to localhost.
- SQLite queue survives service restarts; unfinished jobs resume on next startup.
- One match processes at a time to limit GPU/RAM use. Up to 100 matches per request,
  20 unfinished jobs in the queue. Jobs may span events.
- Stable request IDs prevent duplicate submission on a lost network response.
- Each completed match gets a stable JSON snapshot. Partial results survive a later failure.
- Cached results are reused unless the app requests `force: true`.
- [Server-only TypeScript client](../integrations/scouting-app/client.ts) for the app backend.

This repository prepares the processing side. It does **not** change or deploy the
scouting app, create its buttons, create Supabase tables/functions, configure a tunnel,
or install a startup task on your computer. The remaining app work is specified below.

## Set up the computer

Follow [Windows/Linux PC setup](SETUP_PC.md) or [Mac setup](SETUP_MAC.md) first.
Use Python 3.10+, install the package, configure `TBA_API_KEY` in `.secrets/.env`,
and confirm `autoscout run-match MATCH_KEY` works locally. Video downloads may
require YouTube cookies as described in those setup guides.

Generate a separate API secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add the generated value to the git-ignored `.secrets/.env`:

```dotenv
AUTOSCOUT_API_TOKEN=YOUR_GENERATED_SECRET
```

Start the worker from the project directory, using its virtual environment:

```bash
autoscout serve-integration
```

On Windows, after activating `.venv\Scripts\Activate.ps1`, the same command works.
On Mac/Linux, activate with `source .venv/bin/activate`.
`python -m autoscout.cli serve-integration` is an alternative if the launcher is unavailable.

Use `autoscout serve` on port 8000 locally to calibrate each new event's broadcast
camera. The integration API is on **8001**. Only expose **8001** through an HTTPS
tunnel, e.g. a named Cloudflare Tunnel whose origin is `http://127.0.0.1:8001`.
Use a stable tunnel hostname for the app backend. No router port forwarding is needed.
Do not expose port 8000: that dashboard has local editing/training controls.
The app backend cannot reach the computer via its own `localhost` address.

Keep the API token in the app backend's secret store as `AUTOSCOUT_API_TOKEN` and the
tunnel URL as `AUTOSCOUT_BASE_URL`. Neither belongs in a `VITE_*` variable or browser code.
No Supabase service-role key is needed on the computer. Rotate the API token on both
sides if it is exposed. There is deliberately no browser CORS support on the computer API.

For unattended use, configure **two** startup entries: the integration process and
the tunnel. Windows Task Scheduler can launch `.venv\Scripts\python.exe` with arguments
`-m autoscout.cli serve-integration` and the repository as its working directory.
On Mac, use a LaunchAgent with the virtual environment's Python and the same arguments
and working directory. Disable automatic sleep while plugged in. Restart services after
updating the package or credentials. An API health check confirms connectivity, not
that an event is calibrated or that YouTube will allow a download.

## HTTP contract

Every endpoint requires `Authorization: Bearer <AUTOSCOUT_API_TOKEN>`.
The app calls these through its backend. Responses must not be cached by that proxy.

| Method/path | Purpose |
|---|---|
| `GET /v1/health` | Connectivity, schema version, supported seasons |
| `POST /v1/jobs` | Submit selected matches; returns 202 and a job |
| `GET /v1/jobs/{id}` | Status and completed match keys |
| `GET /v1/jobs/{id}/results` | Job plus completed result snapshots, including partial failures |

POST requires `Content-Type: application/json` and `Idempotency-Key: <persisted UUID>`:

```json
{
  "match_keys": ["2026txhou_qm11", "2026txhou_qm12"],
  "force": false
}
```

Use a new request UUID for a new intentional run. Reuse the original UUID and identical
body after a timeout. Reusing it with a different request returns 409. Missing headers,
invalid keys, unsupported seasons, or extra request fields return 422; a full queue returns
429. Only 2025 and 2026 and full TBA qualification/playoff match keys are supported.
The app cannot submit commands, local paths, video URLs, weights, or callback destinations.

Job statuses are `queued`, `running`, `succeeded`, `failed`. Poll every 3–5 seconds while
the screen is open. A match may take several minutes. A failed job retains earlier completed
matches; retry failed/uncompleted matches as a new request. Remote errors are concise;
full logs are on the computer under `data/integration/<job-id>.log`.

The results response is `{schema_version: 1, job: {...}, results: [...]}`. Each entry is:

```json
{
  "schema_version": 1,
  "source": "autoscout",
  "match": {
    "match_key": "2026txhou_qm11",
    "event_key": "2026txhou",
    "year": 2026,
    "teams": {
      "254": {"alliance": "red", "station": 1, "fuel_scored_auto": 12}
    }
  }
}
```

This is an abbreviated example. Actual output preserves all season metrics in `teams`,
official/alliance summaries (including unattributed fuel), identity confidence/notes,
timestamped events, video reference, clock, and available processing metadata. Large raw
tracks and detection arrays are excluded. Optional fields may be absent in older results.
`source` identifies automated estimates; it is not a claim of verified robot attribution.

## App implementation handoff

The linked app is React/Vite with Supabase Edge Functions and anonymous sessions.
Keep its existing shared-team authorization model. Its `LocalMatchReport` in
`src/db/types.ts` and `src/sync/mapReport.ts` describe human capture; `ingest-reports`
calls `upsert_match_report`, which recomputes aggregates from raw human inputs.
Posting the envelope above to that endpoint will **not** import it correctly.

Implement these pieces in the scouting-app repository:

1. Add an `autoscout` backend bridge with `start`, `status`, and `results` actions.
   Use the existing session validation conventions and the server-only client supplied
   here. Read the tunnel URL and API token from backend secrets, never request parameters.
   Validate/limit match selection and keep the existing team's shared access model.
   Return an offline/unavailable state when the tunnel/computer cannot be reached.
2. Persist request UUID, returned job ID, requested matches, status, and latest progress
   in app storage so reloads can resume polling. A backend network timeout can happen
   after the computer accepts a request: retry with the saved request UUID.
3. Store the versioned envelopes in a **separate automated-results table**, keyed by
   `match_key`, with `event_key`, source job ID and import time. Enable RLS and explicit
   grants consistent with the app. Make re-import of the same job/match a no-op; compare
   job creation time before replacing a match so delayed polling of an older job cannot
   overwrite a newer result. Persist all completed matches even if the overall job failed.
4. Add match selection → **Run Auto-Scout**, a progress panel, **Retry unfinished**, and
   team/match views of the imported estimates. Fetch/persist results on partial progress
   and terminal status. Later app visits can finish importing a job after the browser closes;
   immediate import with every browser closed requires a backend scheduled poller.
5. Keep human scouting reports intact. Display an automated source label and confidence.
   If merging these estimates into predictions, explicitly define source precedence and
   avoid counting a human and automated report as two independent observations of a match.

Useful 2026 metrics include `fuel_scored_auto`, `fuel_scored_teleop`,
`fuel_scored_by_period`, `fuel_fed`, `auto_tower`, and `endgame_tower`.
Period definitions and units must be checked against the app's `src/scoring/windows.ts`
before converting them. `inactive_shots` is a shot estimate, not made fuel. Subjective
ratings, foul reasons, human feeding-slider inputs, intake sources and exact path samples
cannot be inferred from these aggregate fields; leave them unknown rather than fabricate
human reports. 2025 uses different metrics and needs a separate display adapter.

## Operations and verification

Queue state, result snapshots and logs live under `AUTOSCOUT_DATA/integration` (default
`data/integration`). Back up that directory with `data/autoscout.db` and event calibrations.
Use one integration service per data directory; a file lock prevents two workers claiming
the same queue. Avoid running dashboard/CLI processing simultaneously on a small GPU.
Normal shutdown finishes the current job; forced shutdown requeues it on startup, reusing
completed snapshots. An interrupted individual match can be processed again.
Results/logs remain on disk until manually archived; there is no automatic retention cleanup.

Run the isolated contract suite with `python -m pytest tests/test_integration.py` after
installing `pip install -e '.[dev]'`. It tests auth, validation, duplicate submissions,
partial failure, snapshots, cache/forced processing and restart recovery without vision models.
For final live acceptance, start the computer API, run one calibrated match from the app,
confirm all six team results reach its automated-results table, reload the app during a job,
and test an offline computer. The live app/tunnel flow is not verified by the contract suite.
