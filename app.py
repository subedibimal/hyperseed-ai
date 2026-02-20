from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from faker import Faker

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency at runtime
    Anthropic = None

try:
    import requests
except Exception:  # pragma: no cover - optional dependency at runtime
    requests = None

fake = Faker()
PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "BOLD_schema.sql"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_CLAUDE_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
]
DEFAULT_BOLD_API_BASE_URL = "https://api.avionte.com/front-office"
BOLD_CREATE_HCM_USER_PATH = "/v1/user"
BOLD_CREATE_TALENT_PATH = "/v1/talent"
BOLD_CREATE_TALENT_USER_PATH = "/v1/user/talent-user"
DEFAULT_HCM_EMAIL_DOMAIN = "seed.bold.local"
# ---------------------------
# 1) Schema decoding + parser
# ---------------------------
def decode_sql_bytes(raw: bytes) -> Tuple[str, str]:
    """Decode uploaded SQL bytes while handling UTF-16LE exports from SQL Server."""
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16", errors="ignore"), "utf-16"
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be", errors="ignore"), "utf-16-be"
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore"), "utf-8-ignore"


def normalize_identifier(identifier: str) -> str:
    ident = identifier.strip()
    if ident.startswith("[") and ident.endswith("]"):
        return ident[1:-1]
    if ident.startswith('"') and ident.endswith('"'):
        return ident[1:-1]
    if ident.startswith("`") and ident.endswith("`"):
        return ident[1:-1]
    return ident


def normalize_table_name(raw_name: str) -> str:
    # Handles [dbo].[users], "dbo"."users", dbo.users
    bracket_parts = re.findall(r"\[([^\]]+)\]", raw_name)
    if bracket_parts:
        return ".".join(part.strip() for part in bracket_parts if part.strip())

    quoted_parts = re.findall(r'"([^"]+)"', raw_name)
    if quoted_parts:
        return ".".join(part.strip() for part in quoted_parts if part.strip())

    parts = [normalize_identifier(p) for p in raw_name.split(".") if p.strip()]
    return ".".join(parts)


@st.cache_data(show_spinner=False)
def parse_sql_schema_bytes(raw: bytes) -> Dict[str, Any]:
    sql_text, encoding = decode_sql_bytes(raw)

    create_table_pattern = re.compile(
        r"^\s*CREATE\s+TABLE\s+(?P<name>.+?)\s*\(\s*$", re.IGNORECASE
    )
    table_end_pattern = re.compile(r"^\s*\)\s*(?:WITH\b|ON\b|TEXTIMAGE_ON\b|;|$)", re.IGNORECASE)
    constraint_pattern = re.compile(
        r"^\s*(CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK)\b", re.IGNORECASE
    )
    column_pattern = re.compile(
        r"^\s*(\[[^\]]+\]|\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)\s+"
    )

    tables: Dict[str, List[str]] = {}
    current_table: str | None = None
    current_columns: List[str] = []

    for raw_line in sql_text.splitlines():
        line = raw_line.strip()

        if current_table is None:
            m = create_table_pattern.match(line)
            if m:
                current_table = normalize_table_name(m.group("name"))
                current_columns = []
            continue

        if table_end_pattern.match(line):
            tables[current_table] = current_columns
            current_table = None
            current_columns = []
            continue

        if not line or line.startswith("--") or constraint_pattern.match(line):
            continue

        candidate = line.rstrip(",")
        cm = column_pattern.match(candidate)
        if not cm:
            continue

        column_name = normalize_identifier(cm.group(1))
        if column_name and column_name.upper() not in {"PRIMARY", "FOREIGN"}:
            current_columns.append(column_name)

    if current_table is not None:
        tables[current_table] = current_columns

    return {
        "tables": tables,
        "encoding": encoding,
        "table_count": len(tables),
    }


def find_relevant_tables(tables: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    keywords = ["company", "user", "applicant", "talent", "req", "job"]
    rows: List[Dict[str, Any]] = []
    for table_name, cols in tables.items():
        low = table_name.lower()
        if any(k in low for k in keywords):
            rows.append(
                {
                    "table": table_name,
                    "columns_detected": len(cols),
                    "sample_columns": ", ".join(cols[:8]),
                }
            )
    rows.sort(key=lambda x: (x["table"].split(".")[-1], x["table"]))
    return rows[:40]


def get_claude_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("CLAUDE_API_KEY", "").strip()


def get_available_gemini_models(api_key: str) -> Tuple[List[str], str]:
    if not api_key:
        return DEFAULT_GEMINI_MODELS, "missing_api_key"

    if genai is None:
        return DEFAULT_GEMINI_MODELS, "google-generativeai_not_installed"

    try:
        genai.configure(api_key=api_key)
        models: List[str] = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue

            name = getattr(model, "name", "")
            if not name:
                continue

            if name.startswith("models/"):
                name = name.split("/", 1)[1]

            models.append(name)

        unique = sorted(set(models))
        if unique:
            return unique, "ok"
        return DEFAULT_GEMINI_MODELS, "empty_model_list"
    except Exception as exc:
        return DEFAULT_GEMINI_MODELS, f"list_models_error:{exc.__class__.__name__}"


def get_available_claude_models(api_key: str) -> Tuple[List[str], str]:
    if not api_key:
        return DEFAULT_CLAUDE_MODELS, "missing_api_key"

    if Anthropic is None:
        return DEFAULT_CLAUDE_MODELS, "anthropic_not_installed"

    try:
        client = Anthropic(api_key=api_key)
        models: List[str] = []

        try:
            model_response = client.models.list()
        except TypeError:
            model_response = client.models.list(limit=100)

        for model in getattr(model_response, "data", []) or []:
            model_id = getattr(model, "id", None) or getattr(model, "name", None)
            if model_id:
                models.append(model_id)

        unique = sorted(set(models))
        if unique:
            return unique, "ok"
        return DEFAULT_CLAUDE_MODELS, "empty_model_list"
    except Exception as exc:
        return DEFAULT_CLAUDE_MODELS, f"list_models_error:{exc.__class__.__name__}"


# ---------------------------
# 2) LLM profile generation (Gemini + fallback)
# ---------------------------
def build_fallback_profile(schema: Dict[str, Any]) -> Dict[str, Any]:
    table_names = {name.lower() for name in schema["tables"].keys()}

    profile = {
        "email_domains": ["test.bold.local", "seed.bold.local"],
        "locations": [
            {"value": "Remote", "weight": 0.30},
            {"value": "Dallas, TX", "weight": 0.14},
            {"value": "Chicago, IL", "weight": 0.12},
            {"value": "Minneapolis, MN", "weight": 0.10},
            {"value": "Atlanta, GA", "weight": 0.10},
            {"value": "Phoenix, AZ", "weight": 0.08},
            {"value": "Seattle, WA", "weight": 0.08},
            {"value": "Boston, MA", "weight": 0.08},
        ],
        "industries": [
            "Light Industrial Staffing",
            "Healthcare Staffing",
            "IT Consulting",
            "Finance & Accounting",
            "Retail Workforce",
        ],
        "company_names": [
            "Northstar Workforce",
            "BoldBridge Talent",
            "RapidHire Partners",
            "Pinnacle Staffing Group",
            "Summit Talent Ops",
        ],
        "job_titles": [
            {"value": "Software Engineer", "weight": 0.18},
            {"value": "Recruiter", "weight": 0.18},
            {"value": "Data Analyst", "weight": 0.12},
            {"value": "Account Manager", "weight": 0.12},
            {"value": "Customer Success Manager", "weight": 0.10},
            {"value": "QA Engineer", "weight": 0.10},
            {"value": "HR Generalist", "weight": 0.10},
            {"value": "DevOps Engineer", "weight": 0.10},
        ],
        "job_title_details": {
            "Software Engineer": {
                "skills": ["Python", "SQL", "AWS", "Docker", "REST", "Git", "CI/CD"],
                "salary_range": [90000, 185000],
                "experience_range": [1, 11],
            },
            "Recruiter": {
                "skills": ["Sourcing", "ATS", "Screening", "Pipeline Mgmt", "Negotiation"],
                "salary_range": [52000, 145000],
                "experience_range": [0, 10],
            },
            "Data Analyst": {
                "skills": ["SQL", "Excel", "Power BI", "Tableau", "Statistics", "ETL"],
                "salary_range": [65000, 152000],
                "experience_range": [0, 10],
            },
            "Account Manager": {
                "skills": ["Client Management", "CRM", "Negotiation", "Forecasting", "Communication"],
                "salary_range": [60000, 155000],
                "experience_range": [1, 12],
            },
            "Customer Success Manager": {
                "skills": ["Onboarding", "Retention", "QBRs", "CRM", "Escalation Management"],
                "salary_range": [60000, 142000],
                "experience_range": [0, 10],
            },
            "QA Engineer": {
                "skills": ["Test Plans", "Automation", "Selenium", "API Testing", "Regression", "PyTest"],
                "salary_range": [68000, 162000],
                "experience_range": [0, 10],
            },
            "HR Generalist": {
                "skills": ["Employee Relations", "Benefits", "Onboarding", "Compliance", "HRIS"],
                "salary_range": [52000, 132000],
                "experience_range": [0, 10],
            },
            "DevOps Engineer": {
                "skills": ["AWS", "Terraform", "Kubernetes", "Docker", "CI/CD", "Linux"],
                "salary_range": [98000, 215000],
                "experience_range": [2, 12],
            },
        },
    }

    # Tiny schema-aware tweaks
    if any("company" in t for t in table_names):
        profile["company_names"].extend(["Unified Branch Ops", "Metro Staffing Cloud"])

    if any("users" in t for t in table_names):
        profile["email_domains"].append("users.bold.local")

    return profile


def schema_prompt_snapshot(
    schema: Dict[str, Any], table_limit: int = 30, column_limit: int = 12
) -> List[Dict[str, Any]]:
    tables = schema.get("tables", {})
    keywords = ("company", "user", "applicant", "talent", "req", "job")
    ranked = sorted(
        tables.items(),
        key=lambda item: (0 if any(k in item[0].lower() for k in keywords) else 1, item[0].lower()),
    )
    return [
        {"table": table_name, "columns": columns[:column_limit]}
        for table_name, columns in ranked[:table_limit]
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty model output")

    candidates: List[str] = [cleaned]

    if cleaned.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
        if stripped and stripped not in candidates:
            candidates.append(stripped)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if block and block not in candidates:
            candidates.append(block)

    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for brace_match in re.finditer(r"\{", candidate):
            snippet = candidate[brace_match.start() :]
            try:
                parsed, _ = decoder.raw_decode(snippet)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return parsed

    raise ValueError("No valid JSON object found in model output")


def coerce_profile_shape(candidate: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return fallback

    profile = dict(fallback)
    for key in (
        "email_domains",
        "locations",
        "industries",
        "company_names",
        "job_titles",
        "job_title_details",
    ):
        value = candidate.get(key)
        if value:
            profile[key] = value

    def valid_weighted_list(items: Any) -> bool:
        return (
            isinstance(items, list)
            and bool(items)
            and all(isinstance(it, dict) and "value" in it and "weight" in it for it in items)
        )

    if not valid_weighted_list(profile.get("locations")):
        profile["locations"] = fallback["locations"]
    if not valid_weighted_list(profile.get("job_titles")):
        profile["job_titles"] = fallback["job_titles"]

    if not isinstance(profile.get("job_title_details"), dict):
        profile["job_title_details"] = {}

    fallback_default = {
        "skills": ["Communication", "Teamwork", "Problem Solving", "Time Management"],
        "salary_range": [50000, 120000],
        "experience_range": [0, 10],
    }

    for item in profile["job_titles"]:
        title = item.get("value") if isinstance(item, dict) else None
        if not title:
            continue

        fallback_details = fallback["job_title_details"].get(title, fallback_default)
        details = profile["job_title_details"].get(title, {})
        if not isinstance(details, dict):
            details = {}

        if not isinstance(details.get("skills"), list) or len(details["skills"]) < 4:
            details["skills"] = fallback_details["skills"]

        salary_range = details.get("salary_range")
        if not (isinstance(salary_range, list) and len(salary_range) == 2):
            details["salary_range"] = fallback_details["salary_range"]

        experience_range = details.get("experience_range")
        if not (isinstance(experience_range, list) and len(experience_range) == 2):
            details["experience_range"] = fallback_details["experience_range"]

        profile["job_title_details"][title] = details

    return profile


def llm_generate_profile_with_gemini(
    schema: Dict[str, Any],
    fallback: Dict[str, Any],
    gemini_model: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if genai is None:
        return fallback, {"provider": "fallback", "status": "google-generativeai_not_installed"}

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return fallback, {"provider": "fallback", "status": "missing_gemini_api_key"}

    model_name = gemini_model.strip() or DEFAULT_GEMINI_MODEL
    schema_snapshot = schema_prompt_snapshot(schema)

    prompt = (
        "You are creating synthetic QA data profile for a staffing platform.\n"
        "Return ONLY a valid JSON object (no markdown, no explanation).\n"
        "Required keys: email_domains, locations, industries, company_names, job_titles, job_title_details.\n"
        "locations and job_titles must be weighted lists with {value, weight}.\n"
        "job_title_details must include per-title: skills(list), salary_range([min,max]), experience_range([min,max]).\n"
        "Keep output realistic for high-volume HCM/Talent testing.\n\n"
        f"Schema summary:\n{json.dumps(schema_snapshot, indent=2)}"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)

        response_text = ""
        try:
            response_text = response.text or ""
        except Exception:
            response_text = ""

        candidate = extract_json_object(response_text)
        profile = coerce_profile_shape(candidate, fallback)
        return profile, {"provider": "gemini", "status": "ok", "model": model_name}
    except Exception as exc:
        return fallback, {
            "provider": "fallback",
            "status": f"gemini_error:{exc.__class__.__name__}",
            "model": model_name,
        }


def llm_generate_profile_with_claude(
    schema: Dict[str, Any],
    fallback: Dict[str, Any],
    claude_model: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if Anthropic is None:
        return fallback, {"provider": "fallback", "status": "anthropic_not_installed"}

    api_key = get_claude_api_key()
    if not api_key:
        return fallback, {"provider": "fallback", "status": "missing_claude_api_key"}

    model_name = claude_model.strip() or DEFAULT_CLAUDE_MODEL
    schema_snapshot = schema_prompt_snapshot(schema)

    prompt = (
        "You are creating synthetic QA data profile for a staffing platform.\n"
        "Return ONLY a valid JSON object (no markdown, no explanation).\n"
        "Required keys: email_domains, locations, industries, company_names, job_titles, job_title_details.\n"
        "locations and job_titles must be weighted lists with {value, weight}.\n"
        "job_title_details must include per-title: skills(list), salary_range([min,max]), experience_range([min,max]).\n"
        "Keep output realistic for high-volume HCM/Talent testing.\n\n"
        f"Schema summary:\n{json.dumps(schema_snapshot, indent=2)}"
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts: List[str] = []
        for block in getattr(response, "content", []) or []:
            if isinstance(block, dict):
                text = str(block.get("text", "")).strip()
            else:
                text = str(getattr(block, "text", "")).strip()
            if text:
                text_parts.append(text)

        response_text = "\n".join(text_parts).strip()
        candidate = extract_json_object(response_text)
        profile = coerce_profile_shape(candidate, fallback)
        return profile, {"provider": "claude", "status": "ok", "model": model_name}
    except Exception as exc:
        return fallback, {
            "provider": "fallback",
            "status": f"claude_error:{exc.__class__.__name__}",
            "model": model_name,
        }


def llm_generate_profile(
    schema: Dict[str, Any],
    use_llm: bool,
    llm_provider: str,
    llm_model: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    fallback = build_fallback_profile(schema)

    if not use_llm:
        return fallback, {"provider": "fallback", "status": "disabled_in_ui"}

    provider = llm_provider.strip().lower()
    if provider == "gemini":
        return llm_generate_profile_with_gemini(schema, fallback, llm_model)
    if provider == "claude":
        return llm_generate_profile_with_claude(schema, fallback, llm_model)

    return fallback, {"provider": "fallback", "status": f"unknown_provider:{provider}"}


# ---------------------------
# 3) Data generators
# ---------------------------
def pick_weighted(items: List[Dict[str, Any]]) -> str:
    values = [it["value"] for it in items]
    weights = [float(it["weight"]) for it in items]
    return random.choices(values, weights=weights, k=1)[0]


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def split_location(location: str) -> Tuple[str, str | None]:
    if "," not in location:
        return location, None
    city, state = [part.strip() for part in location.split(",", 1)]
    return city, state


def gen_hcm(profile: Dict[str, Any], i: int) -> Dict[str, Any]:
    location = pick_weighted(profile["locations"])
    city, state = split_location(location)
    return {
        "id": f"hcm_{i + 1:06d}",
        "company_name": random.choice(profile["company_names"]),
        "industry": random.choice(profile["industries"]),
        "location": location,
        "city": city,
        "state": state,
        "employee_count": random.randint(50, 5000),
        "account_tier": random.choice(["SMB", "Mid-Market", "Enterprise"]),
        "created_at": now_ts(),
    }


def gen_talent(profile: Dict[str, Any], hcm_id: str, i: int) -> Dict[str, Any]:
    job = pick_weighted(profile["job_titles"])
    rules = profile["job_title_details"][job]

    first = fake.first_name()
    last = fake.last_name()
    domain = random.choice(profile["email_domains"])
    email = f"{first.lower()}.{last.lower()}{i}@{domain}"

    skill_count = random.randint(4, min(8, len(rules["skills"])))
    skills = random.sample(rules["skills"], k=skill_count)
    location = pick_weighted(profile["locations"])

    return {
        "id": f"tal_{i + 1:07d}",
        "hcm_user_id": hcm_id,
        "first_name": first,
        "last_name": last,
        "email": email,
        "job_title": job,
        "experience_years": random.randint(*rules["experience_range"]),
        "skills": skills,
        "location": location,
        "salary_expectation": random.randint(*rules["salary_range"]),
        "availability": random.choice(["Immediate", "2 weeks", "30 days"]),
        "created_at": now_ts(),
    }


def generate_dataset(
    profile: Dict[str, Any],
    generate_hcm: bool,
    hcm_rows: int,
    generate_talent: bool,
    talent_rows: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hcm_data: List[Dict[str, Any]] = []
    talent_data: List[Dict[str, Any]] = []

    if generate_hcm and hcm_rows > 0:
        for i in range(hcm_rows):
            hcm_data.append(gen_hcm(profile, i))

    hcm_ids = [row["id"] for row in hcm_data] or ["hcm_000001"]

    if generate_talent and talent_rows > 0:
        for i in range(talent_rows):
            talent_data.append(gen_talent(profile, random.choice(hcm_ids), i))

    return hcm_data, talent_data


# ---------------------------
# 3b) BOLD API insert helpers
# ---------------------------
def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_first(names: List[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def env_first_int(names: List[str], default: int) -> int:
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def resolve_bold_base_url() -> str:
    base_url = env_first(["BOLD_API_BASE_URL", "AviontePrivateURL", "AVIONTE_PRIVATE_URL"])
    if base_url:
        return base_url

    service_url = env_first(["BOLD_SERVICE_URL", "AvionteServiceURL", "AVIONTE_SERVICE_URL"])
    url_path = env_first(["BOLD_URL_PATH", "AvionteURLPath", "AVIONTE_URL_PATH"])
    if service_url and url_path:
        return f"{service_url.rstrip('/')}/{url_path.lstrip('/')}"

    return DEFAULT_BOLD_API_BASE_URL


def get_bold_config_from_env() -> Dict[str, Any]:
    return {
        "base_url": resolve_bold_base_url(),
        "auth_url": env_first(["BOLD_AUTH_URL", "AvionteAuthURL", "AVIONTE_AUTH_URL"]),
        "api_key": env_first(["BOLD_API_KEY", "BOLD_SUBSCRIPTION_KEY", "SubscriptionKey", "SUBSCRIPTION_KEY"]),
        "tenant": env_first(["BOLD_TENANT", "Tenant", "TENANT"]),
        "front_office_tenant_id": env_first(
            ["BOLD_FRONT_OFFICE_TENANT_ID", "FrontOfficeTenantId", "FRONT_OFFICE_TENANT_ID"]
        ),
        "client_id": env_first(["BOLD_CLIENT_ID", "ClientID", "CLIENT_ID"]),
        "client_secret": env_first(["BOLD_CLIENT_SECRET", "ClientSecret", "CLIENT_SECRET"]),
        "grant_type": env_first(["BOLD_GRANT_TYPE", "grant_type", "GRANT_TYPE"], "client_credentials"),
        "scope": env_first(["BOLD_SCOPE", "Scope", "SCOPE"]),
        "auth_timeout": env_first_int(["BOLD_AUTH_TIMEOUT", "AuthTimeOut", "AUTH_TIMEOUT"], 10),
        "call_timeout": env_first_int(["BOLD_CALL_TIMEOUT", "CallTimeOut", "CALL_TIMEOUT"], 30),
        "cache_timeout": env_first_int(["BOLD_CACHE_TIMEOUT", "CacheTimeOut", "CACHE_TIMEOUT"], 300),
        "bearer_token": env_first(["BOLD_BEARER_TOKEN"]),
    }


def get_bold_bearer_token(config: Dict[str, Any], force: bool = False) -> Tuple[bool, str, str]:
    if requests is None:
        return False, "", "requests_not_installed"

    configured_bearer = str(config.get("bearer_token") or "").strip()
    if configured_bearer:
        return True, configured_bearer, "env_bearer_token"

    auth_url = str(config.get("auth_url") or "").strip()
    client_id = str(config.get("client_id") or "").strip()
    client_secret = str(config.get("client_secret") or "").strip()
    grant_type = str(config.get("grant_type") or "client_credentials").strip() or "client_credentials"
    scope = str(config.get("scope") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    auth_timeout = max(1, int(config.get("auth_timeout") or 10))
    cache_timeout = max(1, int(config.get("cache_timeout") or 300))

    missing = [
        label
        for label, value in (
            ("auth_url", auth_url),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("scope", scope),
            ("api_key", api_key),
        )
        if not value
    ]
    if missing:
        return False, "", f"missing_auth_config:{','.join(missing)}"

    cache_key = f"{auth_url}|{client_id}|{scope}|{api_key}"
    cache_bucket = st.session_state.setdefault("bold_token_cache", {})
    now = time.time()
    cached = cache_bucket.get(cache_key, {})
    if not force and cached.get("token") and float(cached.get("expires_at") or 0) > now + 5:
        return True, str(cached["token"]), "cached_token"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    body_params = {
        "grant_type": grant_type,
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }

    try:
        response = requests.post(auth_url, data=body_params, headers=headers, timeout=auth_timeout)
    except Exception as exc:
        return False, "", f"auth_request_error:{exc.__class__.__name__}:{exc}"

    if not response.ok:
        body_text = (response.text or "").strip()
        if len(body_text) > 300:
            body_text = body_text[:300] + "..."
        return False, "", f"auth_http_{response.status_code}:{body_text or 'empty_response'}"

    try:
        token_payload = response.json()
    except Exception:
        return False, "", "auth_invalid_json"

    access_token = str(token_payload.get("access_token") or "").strip()
    if not access_token:
        return False, "", "auth_missing_access_token"

    expires_in = 0
    try:
        expires_in = int(token_payload.get("expires_in") or 0)
    except (TypeError, ValueError):
        expires_in = 0

    ttl = cache_timeout
    if expires_in > 90:
        ttl = min(ttl, expires_in - 60)

    cache_bucket[cache_key] = {
        "token": access_token,
        "expires_at": now + max(30, ttl),
    }

    return True, access_token, "auth_generated"


def parse_talent_ids(raw: str) -> List[int]:
    ids: List[int] = []
    seen: set[int] = set()

    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue

        match = re.search(r"\d+", token)
        if not match:
            continue

        talent_id = int(match.group(0))
        if talent_id <= 0 or talent_id in seen:
            continue

        ids.append(talent_id)
        seen.add(talent_id)

    return ids


def build_bold_headers(
    api_key: str,
    bearer_token: str,
    tenant: str,
    front_office_tenant_id: str,
) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    if api_key.strip():
        headers["x-api-key"] = api_key.strip()

    token = bearer_token.strip()
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token

    if tenant.strip():
        headers["Tenant"] = tenant.strip()

    if front_office_tenant_id.strip():
        headers["FrontOfficeTenantId"] = front_office_tenant_id.strip()

    return headers


def build_hcm_user_payload(
    row: Dict[str, Any],
    home_office_id: int,
    user_type_id: int,
    email_run_suffix: str = "",
) -> Dict[str, Any]:
    row_id = str(row.get("id", "")).strip().lower()
    email_local_part = re.sub(r"[^a-z0-9]+", "", row_id) or f"hcm{int(time.time())}"
    if email_run_suffix:
        email_local_part = f"{email_local_part}-{email_run_suffix}"

    company_name = str(row.get("company_name", "Seed Company")).strip()
    first_name_parts = [part for part in re.split(r"[^A-Za-z]+", company_name) if part]
    first_name = (first_name_parts[0] if first_name_parts else "Seed")[:40]

    payload = {
        "firstName": first_name,
        "lastName": "User",
        "emailAddress": f"{email_local_part}@{DEFAULT_HCM_EMAIL_DOMAIN}".lower(),
        "city": str(row.get("city") or ""),
        "stateProvince": str(row.get("state") or ""),
        "country": "US",
        "homeOfficeId": int(home_office_id),
        "userTypeId": int(user_type_id),
        "sendWelcomeEmail": False,
    }

    return {k: v for k, v in payload.items() if v not in {"", None}}


def iso8601_utc(days_offset: int = 0) -> str:
    timestamp = time.time() + (int(days_offset) * 86400)
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(timestamp))


def build_talent_address(city: str, state_code: str) -> Dict[str, str]:
    return {
        "street1": fake.street_address(),
        "street2": "Suite 100",
        "city": city or "St. Paul",
        "state_Province": state_code or "MN",
        "postalCode": fake.numerify(text="#####"),
        "country": "US",
        "county": "Dakota",
        "geoCode": fake.numerify(text="#########"),
        "schoolDistrictCode": "00000",
    }


def build_talent_payload(
    row: Dict[str, Any],
    front_office_id: int,
    email_run_suffix: str = "",
) -> Dict[str, Any]:
    first_name = str(row.get("first_name") or "John").strip() or "John"
    middle_name = fake.first_name()
    last_name = str(row.get("last_name") or "Smith").strip() or "Smith"

    raw_email = str(row.get("email") or "").strip().lower()
    if "@" in raw_email:
        email_local_part, email_domain = raw_email.split("@", 1)
    else:
        default_local = re.sub(r"[^a-z0-9]+", "", f"{first_name}.{last_name}".lower())
        email_local_part = default_local or f"tal{int(time.time())}"
        email_domain = DEFAULT_HCM_EMAIL_DOMAIN

    email_local_part = re.sub(r"[^a-z0-9._-]+", "", email_local_part) or f"tal{int(time.time())}"
    email_domain = re.sub(r"[^a-z0-9.-]+", "", email_domain) or DEFAULT_HCM_EMAIL_DOMAIN
    if email_run_suffix:
        email_local_part = f"{email_local_part}-{email_run_suffix}"

    email_address = f"{email_local_part}@{email_domain}"
    email_address2 = f"{email_local_part}-alt@{email_domain}"

    location = str(row.get("location") or "").strip()
    city, state = split_location(location)
    city_name = city.strip() if city else "St. Paul"
    state_code = ((state or "MN").strip()[:2] or "MN").upper()

    resident_address = build_talent_address(city=city_name, state_code=state_code)
    mailing_address = dict(resident_address)
    payroll_address = dict(resident_address)
    extra_address = dict(resident_address)

    origin_record_id = re.sub(r"[^A-Za-z0-9]+", "", str(row.get("id") or "")).upper()[:24]
    if not origin_record_id:
        origin_record_id = uuid.uuid4().hex[:12].upper()

    entered_by_user = f"seedrecruiter@{email_domain}"
    representative_user_email = f"rep.{email_local_part}@{email_domain}"

    birthday = f"{fake.date_of_birth(minimum_age=21, maximum_age=65).isoformat()}T00:00:00.000Z"
    hire_date = iso8601_utc(days_offset=-random.randint(365, 9000))
    i9_validated_date = iso8601_utc(days_offset=-random.randint(365, 7000))
    latest_activity_date = iso8601_utc(days_offset=-random.randint(0, 90))
    availability_date = iso8601_utc(days_offset=-random.randint(30, 3650))
    created_date = iso8601_utc(days_offset=-random.randint(365, 4000))
    last_updated_date = iso8601_utc(days_offset=-random.randint(0, 180))
    last_contacted = iso8601_utc(days_offset=-random.randint(0, 120))
    rehire_date = iso8601_utc(days_offset=-random.randint(365, 9000))
    termination_date = iso8601_utc(days_offset=-random.randint(0, 365))

    payload = {
        "firstName": first_name,
        "middleName": middle_name,
        "lastName": last_name,
        "homePhone": fake.numerify(text="###-###-####"),
        "workPhone": fake.numerify(text="###-###-####"),
        "mobilePhone": f"+1{fake.numerify(text='##########')}",
        "pageNumber": f"({fake.numerify(text='###')}) {fake.numerify(text='###')}-{fake.numerify(text='####')}",
        "emailAddress": email_address,
        "emailAddress2": email_address2,
        "taxIdNumber": fake.numerify(text="#########"),
        "birthday": birthday,
        "gender": random.choice(["M", "F"]),
        "hireDate": hire_date,
        "residentAddress": resident_address,
        "mailingAddress": mailing_address,
        "payrollAddress": payroll_address,
        "addresses": [resident_address, extra_address],
        "status": "Active",
        "filingStatus": random.choice(["Single", "Married"]),
        "federalAllowances": 1,
        "stateAllowances": 1,
        "additionalFederalWithholding": 0,
        "i9ValidatedDate": i9_validated_date,
        "frontOfficeId": int(front_office_id),
        "latestActivityDate": latest_activity_date,
        "latestActivityName": "Interview",
        "link": f"http://example.myavionte.com/app/#/applicant/{fake.numerify(text='#######')}",
        "race": "White (Not Hispanic or Latino)",
        "disability": "Individual without Disabilities",
        "veteranStatus": "Non Veteran",
        "emailOptOut": False,
        "isArchived": False,
        "placementStatus": "Active Contractor",
        "w2Consent": True,
        "electronic1095CConsent": True,
        "referredBy": "Partner API Seed",
        "availabilityDate": availability_date,
        "officeName": "Branch A",
        "officeDivision": "ABC Staffing",
        "enteredByUserId": 19842,
        "enteredByUser": entered_by_user,
        "representativeUserEmail": representative_user_email,
        "createdDate": created_date,
        "lastUpdatedDate": last_updated_date,
        "latestWork": f"{str(row.get('job_title') or 'General Laborer')} - Contractor",
        "lastContacted": last_contacted,
        "flag": random.choice(["Green", "Yellow"]),
        "origin": "A+ PartnerAPI",
        "originRecordId": origin_record_id,
        "electronic1099Consent": True,
        "textConsent": "Opt In",
        "rehireDate": rehire_date,
        "terminationDate": termination_date,
        "employmentTypeId": 343,
        "employmentType": "W-2",
        "employmentTypeName": "Custom W2",
    }

    return payload


def extract_talent_id_from_response(response_body: str) -> Tuple[int, str]:
    def as_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                parsed = int(stripped)
                return parsed if parsed > 0 else None
        return None

    def from_json_obj(data: Any) -> int | None:
        direct = as_positive_int(data)
        if direct:
            return direct

        if isinstance(data, dict):
            for key in ("talentId", "talentID", "TalentId", "id", "Id"):
                parsed = as_positive_int(data.get(key))
                if parsed:
                    return parsed

            for container_key in ("data", "result", "value", "payload", "talent"):
                if container_key in data:
                    parsed = from_json_obj(data.get(container_key))
                    if parsed:
                        return parsed

            for key, value in data.items():
                if "talent" in str(key).lower():
                    parsed = from_json_obj(value)
                    if parsed:
                        return parsed

        if isinstance(data, list):
            for item in data:
                parsed = from_json_obj(item)
                if parsed:
                    return parsed

        return None

    body = str(response_body or "").strip()
    if not body:
        return 0, "empty_response"

    if body.isdigit():
        parsed = int(body)
        return (parsed, "plain_numeric_body") if parsed > 0 else (0, "plain_numeric_body")

    try:
        parsed_body = json.loads(body)
    except Exception:
        parsed_body = None

    parsed_from_json = from_json_obj(parsed_body)
    if parsed_from_json:
        return parsed_from_json, "json_body"

    match = re.search(r'"talentId"\s*:\s*(\d+)', body, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), "regex_body"

    return 0, "talent_id_not_found"


def mask_secret(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if len(raw) <= 8:
        return "*" * len(raw)
    return f"{raw[:4]}...{raw[-4:]}"


def redact_headers_for_log(headers: Dict[str, str], expose_sensitive: bool = False) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in headers.items():
        if not expose_sensitive and key.lower() in {"authorization", "x-api-key"}:
            redacted[key] = mask_secret(str(value))
        else:
            redacted[key] = str(value)
    return redacted


def post_bold_payload(
    base_url: str,
    path: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: int = 30,
    expose_sensitive_logs: bool = False,
) -> Tuple[bool, int, str, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    debug_log: Dict[str, Any] = {
        "method": "POST",
        "url": url,
        "timeout_seconds": int(timeout_seconds),
        "request_headers": redact_headers_for_log(headers, expose_sensitive=expose_sensitive_logs),
        "request_payload": payload,
    }

    if requests is None:
        debug_log["status_code"] = 0
        debug_log["ok"] = False
        debug_log["response_body"] = "requests_not_installed"
        return False, 0, "requests_not_installed", debug_log

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

        try:
            body_text = json.dumps(response.json())
        except Exception:
            body_text = (response.text or "").strip()

        response_preview = body_text
        if len(response_preview) > 400:
            response_preview = response_preview[:400] + "..."

        debug_log["status_code"] = int(response.status_code)
        debug_log["ok"] = bool(response.ok)
        debug_log["response_body"] = response_preview
        return response.ok, int(response.status_code), body_text, debug_log
    except Exception as exc:
        error_text = f"request_error:{exc.__class__.__name__}:{exc}"
        debug_log["status_code"] = 0
        debug_log["ok"] = False
        debug_log["response_body"] = error_text
        return False, 0, error_text, debug_log


def insert_hcm_users_to_bold(
    base_url: str,
    common_headers: Dict[str, str],
    hcm_rows: List[Dict[str, Any]],
    home_office_id: int,
    user_type_id: int,
    call_timeout_seconds: int,
    capture_debug_logs: bool,
    expose_sensitive_logs: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"attempted": 0, "success": 0, "failed": 0, "errors": [], "debug_logs": []}
    email_run_suffix = uuid.uuid4().hex[:8]

    for row in hcm_rows:
        headers = dict(common_headers)
        headers["RequestId"] = str(uuid.uuid4())

        payload = build_hcm_user_payload(
            row=row,
            home_office_id=home_office_id,
            user_type_id=user_type_id,
            email_run_suffix=email_run_suffix,
        )
        ok, status_code, response_body, debug_log = post_bold_payload(
            base_url=base_url,
            path=BOLD_CREATE_HCM_USER_PATH,
            headers=headers,
            payload=payload,
            timeout_seconds=call_timeout_seconds,
            expose_sensitive_logs=expose_sensitive_logs,
        )

        if capture_debug_logs and len(summary["debug_logs"]) < 25:
            debug_log["seed_id"] = row.get("id", "")
            summary["debug_logs"].append(debug_log)

        summary["attempted"] += 1
        if ok:
            summary["success"] += 1
        else:
            summary["failed"] += 1
            if len(summary["errors"]) < 10:
                summary["errors"].append(
                    {
                        "seed_id": row.get("id", ""),
                        "status_code": status_code,
                        "response": response_body,
                        "url": debug_log.get("url", ""),
                    }
                )

    return summary


def insert_generated_talent_to_bold(
    base_url: str,
    common_headers: Dict[str, str],
    talent_rows: List[Dict[str, Any]],
    front_office_id: int,
    legacy_work_n: bool,
    call_timeout_seconds: int,
    capture_debug_logs: bool,
    expose_sensitive_logs: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "talent_created": 0,
        "talent_user_created": 0,
        "errors": [],
        "debug_logs": [],
    }
    email_run_suffix = uuid.uuid4().hex[:8]

    for row in talent_rows:
        seed_id = str(row.get("id", ""))
        summary["attempted"] += 1

        create_headers = dict(common_headers)
        create_headers["RequestId"] = str(uuid.uuid4())
        create_payload = build_talent_payload(
            row=row,
            front_office_id=front_office_id,
            email_run_suffix=email_run_suffix,
        )

        create_ok, create_status, create_response, create_debug_log = post_bold_payload(
            base_url=base_url,
            path=BOLD_CREATE_TALENT_PATH,
            headers=create_headers,
            payload=create_payload,
            timeout_seconds=call_timeout_seconds,
            expose_sensitive_logs=expose_sensitive_logs,
        )

        if capture_debug_logs and len(summary["debug_logs"]) < 50:
            create_debug_log["stage"] = "create_talent"
            create_debug_log["seed_id"] = seed_id
            summary["debug_logs"].append(create_debug_log)

        create_response_preview = (
            create_response
            if len(create_response) <= 400
            else create_response[:400] + "..."
        )

        if not create_ok:
            summary["failed"] += 1
            if len(summary["errors"]) < 20:
                summary["errors"].append(
                    {
                        "seed_id": seed_id,
                        "stage": "create_talent",
                        "status_code": int(create_status),
                        "response": create_response_preview,
                        "url": create_debug_log.get("url", ""),
                    }
                )
            continue

        summary["talent_created"] += 1
        created_talent_id, talent_id_source = extract_talent_id_from_response(create_response)
        if created_talent_id <= 0:
            summary["failed"] += 1
            if len(summary["errors"]) < 20:
                summary["errors"].append(
                    {
                        "seed_id": seed_id,
                        "stage": "parse_talent_id",
                        "status_code": int(create_status),
                        "response": create_response_preview,
                        "url": create_debug_log.get("url", ""),
                    }
                )
            continue

        link_headers = dict(common_headers)
        link_headers["RequestId"] = str(uuid.uuid4())
        link_payload = {
            "talentId": int(created_talent_id),
            "legacyWorkN": bool(legacy_work_n),
        }

        link_ok, link_status, link_response, link_debug_log = post_bold_payload(
            base_url=base_url,
            path=BOLD_CREATE_TALENT_USER_PATH,
            headers=link_headers,
            payload=link_payload,
            timeout_seconds=call_timeout_seconds,
            expose_sensitive_logs=expose_sensitive_logs,
        )

        if capture_debug_logs and len(summary["debug_logs"]) < 50:
            link_debug_log["stage"] = "create_talent_user"
            link_debug_log["seed_id"] = seed_id
            link_debug_log["created_talent_id"] = int(created_talent_id)
            link_debug_log["talent_id_source"] = talent_id_source
            summary["debug_logs"].append(link_debug_log)

        link_response_preview = link_response if len(link_response) <= 400 else link_response[:400] + "..."

        if link_ok:
            summary["success"] += 1
            summary["talent_user_created"] += 1
        else:
            summary["failed"] += 1
            if len(summary["errors"]) < 20:
                summary["errors"].append(
                    {
                        "seed_id": seed_id,
                        "stage": "create_talent_user",
                        "talent_id": int(created_talent_id),
                        "status_code": int(link_status),
                        "response": link_response_preview,
                        "url": link_debug_log.get("url", ""),
                    }
                )

    return summary


def insert_talent_users_to_bold(
    base_url: str,
    common_headers: Dict[str, str],
    talent_ids: List[int],
    legacy_work_n: bool,
    call_timeout_seconds: int = 30,
    capture_debug_logs: bool = False,
    expose_sensitive_logs: bool = False,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"attempted": 0, "success": 0, "failed": 0, "errors": [], "debug_logs": []}

    for talent_id in talent_ids:
        headers = dict(common_headers)
        headers["RequestId"] = str(uuid.uuid4())

        payload = {
            "talentId": int(talent_id),
            "legacyWorkN": bool(legacy_work_n),
        }
        ok, status_code, response_body, debug_log = post_bold_payload(
            base_url=base_url,
            path=BOLD_CREATE_TALENT_USER_PATH,
            headers=headers,
            payload=payload,
            timeout_seconds=call_timeout_seconds,
            expose_sensitive_logs=expose_sensitive_logs,
        )

        if capture_debug_logs and len(summary["debug_logs"]) < 25:
            debug_log["stage"] = "create_talent_user"
            debug_log["talent_id"] = int(talent_id)
            summary["debug_logs"].append(debug_log)

        summary["attempted"] += 1
        if ok:
            summary["success"] += 1
        else:
            summary["failed"] += 1
            if len(summary["errors"]) < 10:
                summary["errors"].append(
                    {
                        "talent_id": talent_id,
                        "status_code": status_code,
                        "response": response_body,
                    }
                )

    return summary


# ---------------------------
# 4) Benchmarks
# ---------------------------
def run_query_benchmarks(hcm_data: List[Dict[str, Any]], talent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    timings_ms: Dict[str, float] = {}
    outputs: Dict[str, Any] = {}

    t0 = time.perf_counter()
    outputs["talent_by_location"] = Counter(row["location"] for row in talent_data).most_common(10)
    timings_ms["Q1 talent by location"] = round((time.perf_counter() - t0) * 1000, 2)

    t1 = time.perf_counter()
    title_counter = Counter(row["job_title"] for row in talent_data)
    outputs["talent_by_job_title"] = title_counter.most_common(10)
    timings_ms["Q2 talent by job title"] = round((time.perf_counter() - t1) * 1000, 2)

    t2 = time.perf_counter()
    salary_by_title: Dict[str, List[int]] = {}
    for row in talent_data:
        salary_by_title.setdefault(row["job_title"], []).append(int(row["salary_expectation"]))
    outputs["avg_salary_by_title"] = [
        {"job_title": title, "avg_salary": round(sum(vals) / len(vals), 2), "count": len(vals)}
        for title, vals in salary_by_title.items()
    ]
    outputs["avg_salary_by_title"].sort(key=lambda x: x["avg_salary"], reverse=True)
    timings_ms["Q3 avg salary by title"] = round((time.perf_counter() - t2) * 1000, 2)

    t3 = time.perf_counter()
    outputs["talent_per_hcm"] = Counter(row["hcm_user_id"] for row in talent_data).most_common(10)
    timings_ms["Q4 talent per HCM"] = round((time.perf_counter() - t3) * 1000, 2)

    return {"timings_ms": timings_ms, "outputs": outputs}


# ---------------------------
# 5) Exporters (JSON / SQL)
# ---------------------------
def to_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def sql_escape(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        s = json.dumps(val)
        return "'" + s.replace("'", "''") + "'"
    s = str(val)
    return "'" + s.replace("'", "''") + "'"


def to_sql_inserts(table: str, rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = []
    for row in rows:
        vals = ", ".join(sql_escape(row.get(c)) for c in columns)
        cols = ", ".join(columns)
        lines.append(f"INSERT INTO {table} ({cols}) VALUES ({vals});")
    return "\n".join(lines)


def to_portable_sql(hcm_data: List[Dict[str, Any]], talent_data: List[Dict[str, Any]]) -> str:
    hcm_cols = [
        "id",
        "company_name",
        "industry",
        "location",
        "city",
        "state",
        "employee_count",
        "account_tier",
        "created_at",
    ]
    talent_cols = [
        "id",
        "hcm_user_id",
        "first_name",
        "last_name",
        "email",
        "job_title",
        "experience_years",
        "skills",
        "location",
        "salary_expectation",
        "availability",
        "created_at",
    ]

    chunks: List[str] = []
    if hcm_data:
        chunks.append("-- HCM inserts\n" + to_sql_inserts("hcm_user", hcm_data, hcm_cols))
    if talent_data:
        chunks.append("-- Talent inserts\n" + to_sql_inserts("talent_user", talent_data, talent_cols))
    return "\n\n".join(chunks)


def to_bold_seed_sql(hcm_data: List[Dict[str, Any]], talent_data: List[Dict[str, Any]]) -> str:
    """
    Safe SQL Server seed output using dedicated staging tables.
    This avoids strict constraints on dbo.Company / dbo.users during hackathon demos.
    """
    lines = [
        "-- BOLD hackathon safe seed tables",
        "IF OBJECT_ID('dbo.hack_hcm_user_seed', 'U') IS NULL",
        "BEGIN",
        "    CREATE TABLE [dbo].[hack_hcm_user_seed] (",
        "        [seed_hcm_id] VARCHAR(32) NOT NULL,",
        "        [company_name] VARCHAR(255) NULL,",
        "        [industry] VARCHAR(100) NULL,",
        "        [city] VARCHAR(100) NULL,",
        "        [state] VARCHAR(20) NULL,",
        "        [location] VARCHAR(120) NULL,",
        "        [employee_count] INT NULL,",
        "        [account_tier] VARCHAR(30) NULL,",
        "        [created_at] DATETIME NULL",
        "    );",
        "END;",
        "",
        "IF OBJECT_ID('dbo.hack_talent_user_seed', 'U') IS NULL",
        "BEGIN",
        "    CREATE TABLE [dbo].[hack_talent_user_seed] (",
        "        [seed_talent_id] VARCHAR(32) NOT NULL,",
        "        [seed_hcm_id] VARCHAR(32) NOT NULL,",
        "        [first_name] VARCHAR(100) NULL,",
        "        [last_name] VARCHAR(100) NULL,",
        "        [email] VARCHAR(255) NULL,",
        "        [job_title] VARCHAR(120) NULL,",
        "        [experience_years] INT NULL,",
        "        [skills] NVARCHAR(MAX) NULL,",
        "        [location] VARCHAR(120) NULL,",
        "        [salary_expectation] INT NULL,",
        "        [availability] VARCHAR(30) NULL,",
        "        [created_at] DATETIME NULL",
        "    );",
        "END;",
        "",
    ]

    if hcm_data:
        hcm_cols = [
            "seed_hcm_id",
            "company_name",
            "industry",
            "city",
            "state",
            "location",
            "employee_count",
            "account_tier",
            "created_at",
        ]
        mapped_hcm = [
            {
                "seed_hcm_id": row["id"],
                "company_name": row["company_name"],
                "industry": row["industry"],
                "city": row["city"],
                "state": row["state"],
                "location": row["location"],
                "employee_count": row["employee_count"],
                "account_tier": row["account_tier"],
                "created_at": row["created_at"],
            }
            for row in hcm_data
        ]
        lines.append("-- HCM seed rows")
        lines.append(to_sql_inserts("[dbo].[hack_hcm_user_seed]", mapped_hcm, hcm_cols))
        lines.append("")

    if talent_data:
        talent_cols = [
            "seed_talent_id",
            "seed_hcm_id",
            "first_name",
            "last_name",
            "email",
            "job_title",
            "experience_years",
            "skills",
            "location",
            "salary_expectation",
            "availability",
            "created_at",
        ]
        mapped_talent = [
            {
                "seed_talent_id": row["id"],
                "seed_hcm_id": row["hcm_user_id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "email": row["email"],
                "job_title": row["job_title"],
                "experience_years": row["experience_years"],
                "skills": row["skills"],
                "location": row["location"],
                "salary_expectation": row["salary_expectation"],
                "availability": row["availability"],
                "created_at": row["created_at"],
            }
            for row in talent_data
        ]
        lines.append("-- Talent seed rows")
        lines.append(to_sql_inserts("[dbo].[hack_talent_user_seed]", mapped_talent, talent_cols))

    return "\n".join(lines)


# ---------------------------
# 6) Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI SOC Stars | BOLD Data Generator", layout="wide")
st.title("AI SOC Stars - BOLD High-Volume Data Generator")
st.caption(
    "Generate realistic HCM + Talent datasets, run benchmark queries, and export JSON/SQL for hackathon demos."
)

with st.expander("Hackathon objective", expanded=False):
    st.markdown(
        """
- Generate **realistic, high-volume** account-level (HCM) and end-user (Talent) test data.
- Seed data fast for QA/performance testing.
- Run benchmark queries to validate stability and response time under load.
        """
    )

schema_col, info_col = st.columns([2, 1])
with schema_col:
    uploaded = st.file_uploader("1) Upload schema (.sql/.txt)", type=["sql", "txt"])
with info_col:
    use_local_schema = st.checkbox(
        "Use local BOLD_schema.sql",
        value=DEFAULT_SCHEMA_PATH.exists(),
        help="Reads BOLD_schema.sql from this project folder if upload is empty.",
    )

schema_bytes: bytes | None = None
schema_origin = ""
if uploaded is not None:
    schema_bytes = uploaded.getvalue()
    schema_origin = uploaded.name
elif use_local_schema and DEFAULT_SCHEMA_PATH.exists():
    schema_bytes = DEFAULT_SCHEMA_PATH.read_bytes()
    schema_origin = str(DEFAULT_SCHEMA_PATH)

schema: Dict[str, Any] | None = None
if schema_bytes is not None:
    with st.spinner("Parsing schema..."):
        schema = parse_sql_schema_bytes(schema_bytes)

    st.success(
        f"Parsed {schema['table_count']} tables from {schema_origin} (detected encoding: {schema['encoding']})."
    )

    relevant = find_relevant_tables(schema["tables"])
    if relevant:
        with st.expander("Detected relevant BOLD tables", expanded=False):
            st.dataframe(relevant, width="stretch")

st.subheader("2) LLM profile settings")
env_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
env_claude_key = get_claude_api_key()

llm_col1, llm_col2, llm_col3 = st.columns([1, 1, 2])
with llm_col1:
    use_llm = st.checkbox("Use LLM", value=True)
with llm_col2:
    llm_provider = st.selectbox("Provider", options=["Gemini", "Claude"], disabled=not use_llm)

if llm_provider == "Gemini":
    available_models, model_discovery_status = get_available_gemini_models(env_gemini_key)
    default_model_name = DEFAULT_GEMINI_MODEL
    provider_key_present = bool(env_gemini_key)
    provider_env_hint = "GEMINI_API_KEY"
else:
    available_models, model_discovery_status = get_available_claude_models(env_claude_key)
    default_model_name = DEFAULT_CLAUDE_MODEL
    provider_key_present = bool(env_claude_key)
    provider_env_hint = "ANTHROPIC_API_KEY (or CLAUDE_API_KEY)"

default_model_index = 0
if default_model_name in available_models:
    default_model_index = available_models.index(default_model_name)

with llm_col3:
    llm_model = st.selectbox(
        f"{llm_provider} model",
        options=available_models,
        index=default_model_index,
        disabled=not use_llm,
    )

st.caption(
    "LLM API keys are loaded from env vars: GEMINI_API_KEY for Gemini, "
    "ANTHROPIC_API_KEY (or CLAUDE_API_KEY) for Claude."
)

if use_llm and not provider_key_present:
    st.info(
        f"No {llm_provider} API key detected yet. Generation will use fallback profile until {provider_env_hint} is set."
    )
elif use_llm and model_discovery_status.startswith("list_models_error"):
    st.warning(
        f"Could not fetch available models from {llm_provider} API. Showing default model list instead."
    )

st.subheader("3) Choose generation options")
mode = st.radio("Generate", options=["Both", "HCM only", "Talent only"], horizontal=True)

generate_hcm = mode in {"Both", "HCM only"}
generate_talent = mode in {"Both", "Talent only"}

c1, c2, c3 = st.columns(3)
with c1:
    hcm_rows = st.number_input("HCM rows", min_value=0, max_value=100000, value=25, step=5)
with c2:
    talent_rows = st.number_input("Talent rows", min_value=0, max_value=1000000, value=10000, step=1000)
with c3:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

output_format = st.multiselect(
    "Output format",
    options=["JSON", "Portable SQL", "BOLD Seed SQL"],
    default=["JSON", "BOLD Seed SQL"],
)

run_benchmark = st.checkbox("Run benchmark queries", value=True)

if "result" not in st.session_state:
    st.session_state["result"] = None

if st.button("4) Generate", type="primary", disabled=(schema is None)):
    random.seed(int(seed))
    Faker.seed(int(seed))

    t0 = time.perf_counter()
    profile, llm_meta = llm_generate_profile(
        schema=schema,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    hcm_data, talent_data = generate_dataset(
        profile=profile,
        generate_hcm=generate_hcm,
        hcm_rows=int(hcm_rows),
        generate_talent=generate_talent,
        talent_rows=int(talent_rows),
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    benchmarks = run_query_benchmarks(hcm_data, talent_data) if run_benchmark else None

    payload = {
        "meta": {
            "generated_at": now_ts(),
            "seed": int(seed),
            "schema_table_count": schema["table_count"],
            "schema_encoding": schema["encoding"],
            "llm": llm_meta,
            "counts": {"hcm": len(hcm_data), "talent": len(talent_data)},
            "generation_ms": gen_ms,
        },
        "hcm_user": hcm_data,
        "talent_user": talent_data,
    }

    st.session_state["result"] = {
        "profile": profile,
        "payload": payload,
        "hcm_data": hcm_data,
        "talent_data": talent_data,
        "llm_meta": llm_meta,
        "benchmarks": benchmarks,
        "generation_ms": gen_ms,
    }

result = st.session_state.get("result")

if result:
    st.subheader("3) Preview + benchmark")

    llm_meta = result.get("llm_meta", {"provider": "fallback", "status": "unknown"})
    llm_provider_used = str(llm_meta.get("provider", "fallback")).lower()
    if llm_provider_used in {"gemini", "claude"} and llm_meta.get("status") == "ok":
        provider_label = "Gemini" if llm_provider_used == "gemini" else "Claude"
        st.success(f"LLM profile source: {provider_label} ({llm_meta.get('model', '')})")
    else:
        st.warning(f"LLM profile source: fallback profile ({llm_meta.get('status', 'unknown')})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HCM rows", f"{len(result['hcm_data']):,}")
    m2.metric("Talent rows", f"{len(result['talent_data']):,}")
    m3.metric("Generation time", f"{result['generation_ms']:,} ms")
    total_rows = len(result["hcm_data"]) + len(result["talent_data"])
    rows_per_sec = 0 if result["generation_ms"] == 0 else int((total_rows / result["generation_ms"]) * 1000)
    m4.metric("Rows / sec", f"{rows_per_sec:,}")

    with st.expander("Data profile output", expanded=False):
        st.json(result["profile"])

    with st.expander("HCM sample", expanded=True):
        st.dataframe(result["hcm_data"][:10], width="stretch")

    with st.expander("Talent sample", expanded=True):
        st.dataframe(result["talent_data"][:10], width="stretch")

    if result["benchmarks"]:
        with st.expander("Query benchmark results", expanded=True):
            timings = result["benchmarks"]["timings_ms"]
            timing_rows = [{"query": k, "duration_ms": v} for k, v in timings.items()]
            st.dataframe(timing_rows, width="stretch")

            st.write("Top locations")
            st.dataframe(
                [
                    {"location": location, "count": count}
                    for location, count in result["benchmarks"]["outputs"]["talent_by_location"]
                ],
                width="stretch",
            )

            st.write("Average salary by title")
            st.dataframe(result["benchmarks"]["outputs"]["avg_salary_by_title"][:10], width="stretch")

    st.subheader("4) Download")
    if "JSON" in output_format:
        st.download_button(
            "Download JSON",
            data=to_json_bytes(result["payload"]),
            file_name="generated_data.json",
            mime="application/json",
        )

    if "Portable SQL" in output_format:
        portable_sql = to_portable_sql(result["hcm_data"], result["talent_data"]).encode("utf-8")
        st.download_button(
            "Download Portable SQL",
            data=portable_sql,
            file_name="generated_inserts.sql",
            mime="application/sql",
        )

    if "BOLD Seed SQL" in output_format:
        bold_sql = to_bold_seed_sql(result["hcm_data"], result["talent_data"]).encode("utf-8")
        st.download_button(
            "Download BOLD Seed SQL",
            data=bold_sql,
            file_name="bold_seed_inserts.sql",
            mime="application/sql",
        )

    st.subheader("5) Insert to BOLD API")
    with st.expander("Insert generated users into BOLD", expanded=False):
        bold_config = get_bold_config_from_env()
        st.caption(
            "Calls CreateHCMUser (POST /v1/user) and the Talent flow "
            "CreateTalent (POST /v1/talent) -> CreateTalentUser (POST /v1/user/talent-user), "
            "using .env credentials and auto-generated access token."
        )

        scope_mode = "Tenant" if bold_config["tenant"] else "FrontOfficeTenantId" if bold_config["front_office_tenant_id"] else "missing"
        st.caption(f"Base URL: {bold_config['base_url']}")
        st.caption(f"Tenant scope header: {scope_mode}")

        home_office_id = env_first_int(["BOLD_HOME_OFFICE_ID", "HomeOfficeId", "HOME_OFFICE_ID"], 1)
        user_type_id = env_first_int(["BOLD_USER_TYPE_ID", "UserTypeId", "USER_TYPE_ID"], 1)
        talent_front_office_id = env_first_int(
            ["BOLD_TALENT_FRONT_OFFICE_ID", "BOLD_FRONT_OFFICE_ID", "FrontOfficeId", "FRONT_OFFICE_ID"],
            111,
        )

        insert_col1, insert_col2 = st.columns(2)
        with insert_col1:
            max_hcm_rows_to_push = st.number_input(
                "Max HCM rows to push",
                min_value=1,
                max_value=5000,
                value=50,
                step=10,
            )
        with insert_col2:
            max_talent_rows_to_push = st.number_input(
                "Max Talent rows to push",
                min_value=1,
                max_value=5000,
                value=50,
                step=10,
            )

        hcm_rows_to_push = result["hcm_data"][: int(max_hcm_rows_to_push)]
        talent_rows_to_push = result["talent_data"][: int(max_talent_rows_to_push)]
        legacy_work_n = st.checkbox("legacyWorkN for CreateTalentUser", value=False)

        st.caption(
            f"Ready to push: HCM={len(hcm_rows_to_push)} row(s), Talent={len(talent_rows_to_push)} row(s), "
            f"Talent frontOfficeId={int(talent_front_office_id)}"
        )
        capture_debug_logs = st.checkbox(
            "Capture detailed API logs (URL, headers, payload, response)",
            value=True,
        )
        expose_sensitive_logs = False
        if capture_debug_logs:
            expose_sensitive_logs = st.checkbox(
                "Expose Authorization/x-api-key/access token in logs (unsafe)",
                value=False,
            )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            run_hcm_insert = st.button("Insert HCM users to BOLD", type="secondary")
        with button_col2:
            run_talent_insert = st.button("Insert Talent users to BOLD", type="secondary")

        if run_hcm_insert or run_talent_insert:
            if requests is None:
                st.error("`requests` is not installed. Install dependencies from requirements.txt and retry.")
            elif not bold_config["base_url"]:
                st.error("BOLD API base URL is required.")
            elif not (bold_config["tenant"] or bold_config["front_office_tenant_id"]):
                st.error("Provide Tenant or FrontOfficeTenantId header value.")
            else:
                token_ok, bearer_token, token_status = get_bold_bearer_token(bold_config)
                if not token_ok:
                    st.error(
                        "Unable to generate BOLD access token from .env settings "
                        f"({token_status})."
                    )
                else:
                    common_headers = build_bold_headers(
                        api_key=bold_config["api_key"],
                        bearer_token=bearer_token,
                        tenant=bold_config["tenant"],
                        front_office_tenant_id=bold_config["front_office_tenant_id"],
                    )

                    if capture_debug_logs and expose_sensitive_logs:
                        st.warning("Sensitive debug mode is ON. Do not share these values in screenshots.")
                        st.text_area(
                            "Authorization header (copy to Postman)",
                            value=common_headers.get("Authorization", ""),
                            height=70,
                        )
                        st.text_area(
                            "Access token (copy to Postman)",
                            value=bearer_token,
                            height=110,
                        )

                    if run_hcm_insert:
                        if not hcm_rows_to_push:
                            st.warning("No HCM rows available to push. Generate HCM data first.")
                        else:
                            hcm_summary = {"attempted": 0, "success": 0, "failed": 0, "errors": []}

                            with st.spinner("Calling CreateHCMUser endpoint..."):
                                hcm_summary = insert_hcm_users_to_bold(
                                    base_url=bold_config["base_url"],
                                    common_headers=common_headers,
                                    hcm_rows=hcm_rows_to_push,
                                    home_office_id=int(home_office_id),
                                    user_type_id=int(user_type_id),
                                    call_timeout_seconds=max(1, int(bold_config["call_timeout"])),
                                    capture_debug_logs=capture_debug_logs,
                                    expose_sensitive_logs=expose_sensitive_logs,
                                )

                            st.caption(f"Access token source: {token_status}")
                            st.write("CreateHCMUser results")
                            st.write(
                                f"Attempted: {hcm_summary['attempted']} | "
                                f"Succeeded: {hcm_summary['success']} | Failed: {hcm_summary['failed']}"
                            )
                            if hcm_summary["errors"]:
                                st.dataframe(hcm_summary["errors"], width="stretch")

                            if capture_debug_logs and hcm_summary["debug_logs"]:
                                with st.expander("Detailed HCM API debug logs", expanded=True):
                                    st.json(hcm_summary["debug_logs"])

                            if hcm_summary["failed"] == 0 and hcm_summary["attempted"] > 0:
                                st.success("CreateHCMUser insertion completed successfully.")

                    if run_talent_insert:
                        if not talent_rows_to_push:
                            st.warning("No Talent rows available to push. Generate Talent data first.")
                        else:
                            talent_summary = {
                                "attempted": 0,
                                "success": 0,
                                "failed": 0,
                                "talent_created": 0,
                                "talent_user_created": 0,
                                "errors": [],
                            }

                            with st.spinner("Calling CreateTalent and CreateTalentUser endpoints..."):
                                talent_summary = insert_generated_talent_to_bold(
                                    base_url=bold_config["base_url"],
                                    common_headers=common_headers,
                                    talent_rows=talent_rows_to_push,
                                    front_office_id=int(talent_front_office_id),
                                    legacy_work_n=legacy_work_n,
                                    call_timeout_seconds=max(1, int(bold_config["call_timeout"])),
                                    capture_debug_logs=capture_debug_logs,
                                    expose_sensitive_logs=expose_sensitive_logs,
                                )

                            st.caption(f"Access token source: {token_status}")
                            st.write("CreateTalent + CreateTalentUser results")
                            st.write(
                                f"Attempted: {talent_summary['attempted']} | "
                                f"Talent created: {talent_summary['talent_created']} | "
                                f"Talent user created: {talent_summary['talent_user_created']} | "
                                f"Failed: {talent_summary['failed']}"
                            )
                            if talent_summary["errors"]:
                                st.dataframe(talent_summary["errors"], width="stretch")

                            if capture_debug_logs and talent_summary["debug_logs"]:
                                with st.expander("Detailed Talent API debug logs", expanded=True):
                                    st.json(talent_summary["debug_logs"])

                            if talent_summary["failed"] == 0 and talent_summary["attempted"] > 0:
                                st.success("Talent insertion flow completed successfully.")

else:
    st.info("Upload a schema, choose options, then click Generate.")
