/** Server-only adapter for ryanabraham1/scouting-app.
 * Call from a backend/Edge Function; never bundle AUTOSCOUT_API_TOKEN in Vite.
 */
export interface AutoScoutJob {
  id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  match_keys: string[];
  completed_matches: string[];
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface AutoScoutResult {
  schema_version: 1;
  source: 'autoscout';
  match: {
    match_key: string;
    event_key: string;
    year: number;
    teams: Record<string, { alliance: 'red' | 'blue'; station: number; [metric: string]: unknown }>;
    identity_conf?: Record<string, unknown>;
    alliances?: Record<string, unknown>;
    [evidence: string]: unknown;
  };
}

export class AutoScoutClient {
  constructor(private baseUrl: string, private token: string) {
    const url = new URL(baseUrl);
    if (url.protocol !== 'https:' &&
        !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) {
      throw new Error('Auto-scout needs HTTPS (HTTP allowed for local development)');
    }
    if (url.username || url.password || url.search || url.hash) {
      throw new Error('Use a plain auto-scout base URL');
    }
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, '')}/v1/${path}`, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${this.token}` },
      signal: AbortSignal.timeout(15_000),
      redirect: 'error',
    });
    if (!response.ok) {
      // Avoid surfacing proxy/tunnel HTML or credentials to a browser.
      throw new Error(`Auto-scout request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }

  health() {
    return this.request<{ status: 'online'; schema_version: 1; supported_years: number[] }>('health');
  }

  start(matchKeys: string[], requestId: string, force = false) {
    // Persist requestId before sending; reuse it if the network response is lost.
    return this.request<AutoScoutJob>('jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': requestId },
      body: JSON.stringify({ match_keys: matchKeys, force }),
    });
  }

  status(jobId: string) {
    return this.request<AutoScoutJob>(`jobs/${encodeURIComponent(jobId)}`);
  }

  results(jobId: string) {
    return this.request<{ schema_version: 1; job: AutoScoutJob; results: AutoScoutResult[] }>(
      `jobs/${encodeURIComponent(jobId)}/results`,
    );
  }
}
