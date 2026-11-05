# Connect the scouting app to a computer running Auto-Scout

Target app: [ryanabraham1/scouting-app](https://github.com/ryanabraham1/scouting-app),
inspected at `ca5b0afd42450906e184e9ffd6a6aa8d9477c105`.

The computer runs the video processing. The app's **Auto-Scout** tab (team lead area) creates
jobs in Supabase, and `autoscout bridge` on this computer calls out to collect them and push
status and results back.

```text
Auto-Scout tab -> Supabase (autoscout_job)
                      ^
                      | outbound HTTPS only, every 5-15 s
Computer: autoscout bridge -> persistent queue -> clip/record -> processing -> results
```

Nothing on the computer is exposed: no tunnel, no port forwarding. Jobs wait in Supabase while it
is off, and results land with every browser closed. It holds one worker token, never a Supabase key.
The computer must be awake and online for work to move.

## Job types

| Type | Footage |
|---|---|
| Live stream | Recorded from the live edge in 20 s wall-clock segments; each match is cut once TBA posts its `actual_time`. Only matches after recording starts: YouTube refuses to clip backwards from a stream still live |
| Ended stream | yt-dlp fetches only the window around `actual_time - stream start`. Other days' matches are skipped with a reason |
| Match videos | TBA's per-match videos |

Stream links are allow-listed to YouTube and Twitch and rebuilt from their parsed parts
(`autoscout/streams.py`), so the shared no-login app can't point this computer anywhere else or
pass yt-dlp options. Each event must be calibrated first with `autoscout serve` on port 8000.

A job refuses to start under 10 GB free. Processed clips are capped at 15 GB, and live segments
are kept for 3 hours.

## Set up the computer

Follow [Windows/Linux PC setup](SETUP_PC.md) or [Mac setup](SETUP_MAC.md) first, and confirm
`autoscout run-match MATCH_KEY` works locally. YouTube blocks anonymous downloads, so set
`YT_COOKIES_BROWSER` (or `YT_COOKIES`) as described there.

Generate a worker token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to the git-ignored `.secrets/.env`:

```dotenv
AUTOSCOUT_BRIDGE_URL=https://YOUR-PROJECT.supabase.co/functions/v1/autoscout-worker
AUTOSCOUT_WORKER_TOKEN=YOUR_GENERATED_TOKEN
```

Store only its SHA-256 in the app's private `autoscout_worker_secret` table (see the app's
`docs/autoscout.md`), then on Windows install the startup task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_bridge_task.ps1 -Python .venv\Scripts\python.exe
```

Disable automatic sleep while plugged in. Output goes to `bridge.log`, and full job logs to
`data/integration/<job-id>.log`. Rotate the token on both sides if it is exposed.

The app side lives in the scouting-app repo: see its `docs/autoscout.md`.

## Local HTTP API

`autoscout serve-integration` is the older token-protected API on `127.0.0.1:8001`, kept for local
scripts. The app does not use it. Do not run it together with the bridge (one worker per data
directory), and do not expose it or the port 8000 dashboard.

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
`tests/test_bridge.py` covers the bridge, stream parsing, clipping and live sessions the same way.
