from __future__ import annotations

import json
import os
import random
import re
import time
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
            st.dataframe(relevant, use_container_width=True)

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
        st.dataframe(result["hcm_data"][:10], use_container_width=True)

    with st.expander("Talent sample", expanded=True):
        st.dataframe(result["talent_data"][:10], use_container_width=True)

    if result["benchmarks"]:
        with st.expander("Query benchmark results", expanded=True):
            timings = result["benchmarks"]["timings_ms"]
            timing_rows = [{"query": k, "duration_ms": v} for k, v in timings.items()]
            st.dataframe(timing_rows, use_container_width=True)

            st.write("Top locations")
            st.dataframe(
                [
                    {"location": location, "count": count}
                    for location, count in result["benchmarks"]["outputs"]["talent_by_location"]
                ],
                use_container_width=True,
            )

            st.write("Average salary by title")
            st.dataframe(result["benchmarks"]["outputs"]["avg_salary_by_title"][:10], use_container_width=True)

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

else:
    st.info("Upload a schema, choose options, then click Generate.")
