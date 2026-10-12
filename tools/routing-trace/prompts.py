"""Prompt set for routing traces. Mixes domains and languages on purpose:
expert locality inside one kind of task is the optimistic case, switching
between kinds is the pessimistic one, and the cache has to live with both."""

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _source(rel: str) -> str:
    return (_REPO / rel).read_text()


PROMPTS = [
    # --- code ---
    {
        "id": "code-rust-lru",
        "category": "code",
        "thinking": False,
        "text": "Write an LRU cache in Rust with O(1) get and put, generic over key "
        "and value, without external crates. Include unit tests.",
    },
    {
        "id": "code-python-parser",
        "category": "code",
        "thinking": False,
        "text": "Implement a recursive-descent parser in Python for arithmetic "
        "expressions with +, -, *, /, parentheses, unary minus and variables. "
        "Return an AST and an evaluator. Add docstrings.",
    },
    {
        "id": "code-explain-backend",
        "category": "code-long-context",
        "thinking": False,
        "text": "Explain what this Rust module does, how the threads synchronise, "
        "and point out any bugs or races:\n\n```rust\n"
        + _source("src/backend.rs")
        + "\n```",
    },
    {
        "id": "code-review-main",
        "category": "code-long-context",
        "thinking": True,
        "text": "Review this benchmark driver. Is the throughput it reports "
        "correct when lookahead > 0? Suggest concrete improvements.\n\n```rust\n"
        + _source("src/main.rs")
        + "\n```",
    },
    {
        "id": "code-sql",
        "category": "code",
        "thinking": False,
        "text": "Design a PostgreSQL schema for a multi-tenant invoicing system "
        "(tenants, customers, invoices, line items, payments, taxes). Write the "
        "DDL with indexes and three reporting queries.",
    },
    # --- chat ---
    {
        "id": "chat-pt-viagem",
        "category": "chat-pt",
        "thinking": False,
        "text": "Estou planejando uma viagem de 10 dias por Portugal em abril, "
        "saindo de São Paulo, com orçamento médio. Monte um roteiro dia a dia "
        "com dicas de transporte, comida e hospedagem.",
    },
    {
        "id": "chat-pt-carreira",
        "category": "chat-pt",
        "thinking": False,
        "text": "Sou desenvolvedor backend há 5 anos e quero migrar para "
        "engenharia de machine learning. Que caminho você recomenda? Seja "
        "específico sobre estudos, projetos e como me posicionar no mercado.",
    },
    {
        "id": "chat-en-advice",
        "category": "chat-en",
        "thinking": False,
        "text": "My team keeps missing sprint deadlines even though estimates "
        "look reasonable. What are the likely causes and how would you "
        "diagnose them over the next two weeks?",
    },
    # --- reasoning (thinking on) ---
    {
        "id": "math-probability",
        "category": "math",
        "thinking": True,
        "text": "A box has 5 red and 7 blue balls. You draw balls one at a time "
        "without replacement until you have drawn 3 red balls. What is the "
        "expected number of draws? Show your work.",
    },
    {
        "id": "math-proof",
        "category": "math",
        "thinking": True,
        "text": "Prove that for every positive integer n, the number "
        "n^5 - n is divisible by 30.",
    },
    {
        "id": "reasoning-ssd",
        "category": "reasoning",
        "thinking": True,
        "text": "A Mac reads from its SSD at 14 GB/s. A mixture-of-experts model "
        "needs 8 GB of expert weights per generated token if nothing is cached. "
        "RAM can cache 25% of all experts. Under what assumptions about routing "
        "locality could the model reach 5 tokens per second? Reason carefully.",
    },
    # --- writing / transformation ---
    {
        "id": "writing-story",
        "category": "writing",
        "thinking": False,
        "text": "Write a short science-fiction story (about 800 words) about a "
        "lighthouse keeper on Europa who receives a signal from beneath the ice.",
    },
    {
        "id": "translate-pt-en",
        "category": "translation",
        "thinking": False,
        "text": "Traduza para o inglês, mantendo o tom técnico:\n\n" + _source("README.md"),
    },
    {
        "id": "summarize-readme",
        "category": "summarization",
        "thinking": False,
        "text": "Resuma este documento em 10 tópicos e liste as perguntas que ele "
        "deixa em aberto:\n\n" + _source("README.md"),
    },
    {
        "id": "json-extract",
        "category": "structured",
        "thinking": False,
        "text": "Generate a JSON array of 25 fictional customer support tickets, "
        "each with id, customer_name, product, severity (low/medium/high), "
        "created_at (ISO 8601), and a one-paragraph description.",
    },
    {
        "id": "reasoning-scheduling",
        "category": "reasoning",
        "thinking": True,
        "text": "Five meetings must fit in one afternoon (13:00-18:00) across two "
        "rooms. A (60 min) must precede C (30 min); B (90 min) and D (45 min) "
        "share a speaker; E (60 min) needs room 2. Find a valid schedule and "
        "explain how you ruled out the alternatives.",
    },
    {
        "id": "writing-poem-pt",
        "category": "writing",
        "thinking": False,
        "text": "Escreva um conto curto (cerca de 600 palavras) sobre uma "
        "programadora que descobre que o servidor da empresa está escrevendo "
        "poemas durante a madrugada.",
    },
    {
        "id": "translate-en-pt",
        "category": "translation",
        "thinking": False,
        "text": "Translate into Brazilian Portuguese, keeping code identifiers "
        "unchanged:\n\n```rust\n" + _source("src/metal_io.rs") + "\n```\n\n"
        "Translate only the comments and doc comments; output the full file.",
    },
    {
        "id": "summarize-data-rs",
        "category": "summarization",
        "thinking": False,
        "text": "Summarize what this file does in plain English for a product "
        "manager, then list its public functions:\n\n```rust\n"
        + _source("src/data.rs") + "\n```",
    },
    {
        "id": "json-api",
        "category": "structured",
        "thinking": False,
        "text": "Write an OpenAPI 3.1 YAML spec for a REST API that manages "
        "library books, members and loans, with pagination, filtering and "
        "error schemas.",
    },
    {
        "id": "chat-zh-2",
        "category": "chat-zh",
        "thinking": False,
        "text": "我想学习 Rust 编程语言。请给我一个为期八周的学习计划，"
        "包括每周的主题、练习项目和推荐资源。",
    },
    {
        "id": "chat-zh",
        "category": "chat-zh",
        "thinking": False,
        "text": "请用中文解释什么是混合专家模型（Mixture of Experts），"
        "它的优点和缺点，以及为什么它适合在内存有限的设备上运行。",
    },
]
