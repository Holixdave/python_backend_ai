# Card tools — what the app receives

Everything below is **additive**. Old builds of the app keep working as long as they ignore block types and
fields they don't know (`card` blocks, `continueFlag`, `continueNote`).

## 1. Cards arrive as a normal block

In the `/ai-job` reply doc, `blocks` (live and final) can now contain:

```json
{ "type": "card", "card": { "type": "places", ... } }
```

`card.type` is one of: `places`, `scores`, `standings`, `election`, `drafts`, `recipe`, `doc`.
The model is told **not** to repeat a card's content in its text, so render the card where the block is.

| card.type | Fields |
|---|---|
| `places` | `title`, `center {lat,lng}`, `places[] {name, address, lat, lng, rating?, rating_count?, open_now?, phone?, category?, maps_url, place_id, distance_km?}`, `provider` (`google`/`photon`/`nominatim`), **`attribution` — must be shown under the map** |
| `scores` | `title`, `league`, `games[] {id, status: live\|upcoming\|final\|postponed, status_text, start (ISO), home{name,abbr,logo,score}, away{...}, venue}`, `provider`, `fetched_at`. `score` is `null` before kickoff |
| `standings` | `title`, `league`, `columns[]` (e.g. `["#","P","W","D","L","GD","PTS"]`), `rows[] {group?, team, abbr, logo, stats{<column>: value}}`, `provider` |
| `election` | `race`, `state`, `called` (bool), `candidates[] {name, party, votes, pct, winner, incumbent}`, `precincts_reporting_pct`, `total_votes`, `provider:"ap"`, `attribution`. Only exists when the AP feed answered; the free web fallback returns text only (no card) |
| `drafts` | `kind` (`email`/`textMessage`/`other`), `title`, `variants[] {label, subject? (email only), body}`. Buttons: Copy, Open in Mail / Messages |
| `recipe` | `title`, `description`, `base_servings`, `ingredients[] {id, name, amount, unit\|null}`, `steps[] {id, title, content, timer_seconds?}`, `notes`. `content` may contain `{0001}` tokens = the amount of ingredient `0001`; scale amounts by `servings / base_servings` |
| `doc` | `doc_id`, `title`, `version`, `role`, `preview`, `updatedAt`. Open the editor with `GET /documents/{doc_id}` |

A document export is **not** a card: it arrives as the existing `file` block.

## 2. "Continue" divider

When a reply stops early, the reply doc gets `continueFlag`:

```json
"continueFlag": { "show": true, "reason": "cut_off", "text": "Cut off here · send \"continue\" to pick up" }
```

* `reason`: `cut_off` (every model died mid-answer), `step_failed` (a step produced nothing), `failed` (server error — doc `status` is `"error"`).
* Draw a small divider line with `text` **after the last block**. Show it only if `continueFlag.show` is true.
* The text the user sends to resume is just `continue` (the backend's memory already understands it).
* `cut_off` / `step_failed` docs also contain a text block with `"continueNote": true` (the old "Send continue" sentence,
  kept so older app builds and the model's memory behave exactly as before). **New builds should hide blocks with
  `continueNote: true`** and show the divider instead, otherwise the user sees the message twice. The top-level `text`
  field still contains that sentence.
* The same field is sent on the SSE `final` event as `continue_flag` (snake_case) for `/ai-query-stream`.

## 3. New approval kind: `mcp`

A connected-app call that isn't read-only asks first. The approval request has `kind: "mcp"`,
`title` ("OOOR wants to use <app>"), `target` (`<app> › <tool> <arguments>`), `why`, `risk: "high"`.
Render it like the existing `fetch` approval, without the "always allow" option. An app that doesn't know this kind
simply never answers, which the backend treats as **denied** (fails closed).

## 4. REST endpoints (header `Authorization: Bearer <Firebase ID token>`)

Documents (collection `shared_docs`):

| Method + path | Body | Notes |
|---|---|---|
| `GET /documents` | | mine + shared with me (no markdown) |
| `POST /documents` | `{title, markdown}` | 201 |
| `GET /documents/{id}` | | owner/collaborator; returns `markdown`, `version`, `role` |
| `PUT /documents/{id}` | `{markdown?, title?, base_version?}` | **send the `version` you loaded as `base_version`.** `409` returns `detail.current` (their latest text) — merge or ask the user |
| `DELETE /documents/{id}` | | owner only |
| `POST /documents/{id}/share` | `{mode: "view"\|"edit"\|"off"}` | owner only → `{url, token}`; `off` kills the link |
| `POST /documents/{id}/export` | `{format: "docx"\|"pdf"\|"md"}` | → `{url, filename}` |
| `GET /documents/shared/{token}` | | **no auth**, read-only |
| `POST /documents/shared/{token}/join` | | auth; adds the user as `view`/`edit` collaborator |

"Live" collaboration = poll `GET /documents/{id}` (or compare `version`) — there is no websocket. Every save keeps the
last 20 versions in `shared_docs/{id}/versions/{n}`.

Connected apps:

| Method + path | Body |
|---|---|
| `GET /mcp/servers` | → `{servers:[{id,name,url,hasToken}]}` (tokens are never returned) |
| `POST /mcp/servers` | `{name, url (https), token?, header_name? = "Authorization"}` → 201. The server is **contacted first**; a bad URL/token is rejected and nothing is saved |
| `DELETE /mcp/servers/{id}` | 204 |

Error bodies: `{"detail": {"code": "...", "message": "..."}}` (e.g. `blocked_address`, `unauthorized`, `too_many`,
`duplicate`, `encryption_not_configured`).

## 5. Environment variables (all optional except where noted)

| Variable | Effect when set | When missing |
|---|---|---|
| `GOOGLE_PLACES_API_KEY` (or existing `GOOGLE_API_KEY`) | Google Places first | Photon → Nominatim (OpenStreetMap) |
| `API_FOOTBALL_KEY` | API-Football first (soccer only) | ESPN public feed |
| `AP_ELECTIONS_API_KEY` | AP results (US) | web search, labelled **unofficial** |
| `MCP_TOKEN_KEY` | **required to save any app token** (Fernet key) | tokens are refused; token-less apps still work |
| `MCP_ALLOWED_HOSTS` | only these hosts (and subdomains) may be connected | any public https host |
| `PUBLIC_BASE_URL` | share-link host | `RENDER_EXTERNAL_URL` |

## 6. What has NOT been verified (be honest with your users)

* Google Places, API-Football, ESPN, AP, Photon and Nominatim were **never called live** — the build sandbox had no
  access to them. Parsers are tested against hand-written samples of each provider's documented response shape.
  **The AP adapter is the least certain** (written without a contract key); it fails soft into the web fallback.
* MCP works over **Streamable HTTP with a bearer/API-key token**. Not supported: OAuth login flows, the old HTTP+SSE
  transport. Tested against a local fake server, not against Notion/Slack/Drive.
* SSRF guard checks DNS before every call but cannot fully stop DNS rebinding — set `MCP_ALLOWED_HOSTS` in production.
