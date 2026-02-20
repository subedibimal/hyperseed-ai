# AI SOC Stars - BOLD Data Generator

Hackathon project for **AI Enablement**: generate realistic high-volume HCM + Talent test data, benchmark query-style analytics, and export seed files for BOLD QA/performance testing.

## What this app does

1. Parses SQL schema files (`.sql` / `.txt`) including SQL Server UTF-16 exports.
2. Builds a schema-aware data profile using Gemini or Claude (with automatic fallback profile).
3. Generates synthetic data for:
   - HCM users (account-level)
   - Talent users (end-user level)
4. Runs benchmark queries (distribution + aggregation checks).
5. Exports data as:
   - JSON
   - Portable SQL inserts
   - SQL Server-safe BOLD seed SQL (staging tables)
6. Optionally calls BOLD Front Office API endpoint to insert HCM users directly:
   - `CreateHCMUser` (`POST /v1/user`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit.

### LLM setup (Gemini + Claude)

Provide keys via environment variables (`.env` or shell):

```env
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_claude_key_here
# optional alias supported:
# CLAUDE_API_KEY=your_claude_key_here
```

In the UI, choose:
- LLM on/off
- Provider: Gemini or Claude
- Model from available model list

Defaults:
- Gemini: `gemini-1.5-flash`
- Claude: `claude-3-5-sonnet-latest`

If selected LLM is unavailable (missing key / SDK / API error), app auto-falls back to the built-in profile so generation still works.

## Recommended UI flow

1. Upload schema (`.sql`/`.txt`) or use local `BOLD_schema.sql`.
2. Choose generation mode: HCM / Talent / Both.
3. Set row counts and random seed.
4. Click **Generate**.
5. Review preview + benchmark section.
6. Download JSON / SQL outputs.
7. (Optional) Use **Insert to BOLD API** to push generated HCM rows to BOLD.

## Insert to BOLD API (optional)

The app includes an **Insert HCM users to BOLD** button in section **5) Insert to BOLD API**.
The UI is intentionally simple and reads API settings from environment variables.

### Endpoints used

- `POST /v1/user` (`CreateHCMUser`)

### Authentication / headers

The app now generates bearer token automatically from env-based credentials and sends:
- `x-api-key`
- `Authorization: Bearer <generated_token>`
- `Tenant` or `FrontOfficeTenantId`

Set these env vars in `.env`:
- `BOLD_AUTH_URL`
- `BOLD_API_KEY`
- `BOLD_CLIENT_ID`
- `BOLD_CLIENT_SECRET`
- `BOLD_GRANT_TYPE` (default: `client_credentials`)
- `BOLD_SCOPE`

For base URL, set either:
- `BOLD_API_BASE_URL`
- or both `BOLD_SERVICE_URL` + `BOLD_URL_PATH`

For HCM payload defaults:
- `BOLD_HOME_OFFICE_ID`
- `BOLD_USER_TYPE_ID`

Optional:
- `BOLD_BEARER_TOKEN` (if set, token-generation call is skipped)
- `BOLD_AUTH_TIMEOUT`, `BOLD_CALL_TIMEOUT`, `BOLD_CACHE_TIMEOUT`

Compatibility aliases are also supported for your existing credential names (e.g., `AvionteAuthURL`, `ClientID`, `ClientSecret`, `Scope`, `SubscriptionKey`, `Tenant`, `AviontePrivateURL`, `AvionteServiceURL`, `AvionteURLPath`).

### Important behavior

- HCM insertion uses generated rows and maps them to `HcmUser` payload (`firstName`, `lastName`, `emailAddress`, `homeOfficeId`, `userTypeId`, etc.).
- Current UI keeps this flow intentionally simple (HCM-only).
- Talent user insertion (`CreateTalentUser`) can be added in a later phase.

## Notes for BOLD schema

- `BOLD_schema.sql` appears to be a SQL Server UTF-16 export and is large.
- This app handles UTF-16 decoding automatically.
- For hackathon safety, `BOLD Seed SQL` exports into:
  - `dbo.hack_hcm_user_seed`
  - `dbo.hack_talent_user_seed`

This avoids writing directly into strict production-like tables (`dbo.Company`, `dbo.users`) that may require many mandatory fields, FK rules, or business triggers.

## Where to customize

- Gemini + Claude integration is in `llm_generate_profile()`.
- If you switch providers later, replace this function and keep the same return shape.
- Tune role distributions, salary ranges, skills, and location weights in the profile.
- Adjust benchmark queries in `run_query_benchmarks()` for scenario-specific stress tests.

## Demo pitch alignment (from your PDF)

- **Speed:** generate large test datasets in seconds.
- **Realism:** weighted profile-based demographic/job distributions.
- **Validation:** benchmark query timings and top aggregates shown in UI.
- **Operational fit:** SQL export supports data seeding workflows.
