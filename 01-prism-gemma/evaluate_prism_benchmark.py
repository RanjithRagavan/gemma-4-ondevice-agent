"""
Reproducible empirical evaluation suite for PRISM-Gemma.
Evaluates:
1. Multi-Hop Repository Bug Localization (BM25/Grep vs. Cosine-RAG vs. Uniform RepoMap vs. PRISM RW-PPR)
2. Token Budget Compression & Context Window Footprint on Consumer Hardware (1K-8K budgets)
3. Multi-Turn KV-Cache Prefix Hit Rate (Naive Relevance Sort vs. File Sort vs. Topo-KV)
4. Graph-Proximity Reward Shaping (GPRS) Signal-to-Noise & Group Advantage Variance for GRPO
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from prism_gemma import HeterogeneousCodeGraph


def build_benchmark_repositories() -> List[Dict]:
    """
    Constructs 12 multi-module software engineering benchmark scenarios modeled after
    real-world SWE-bench multi-hop issues (2 to 4 graph hops between symptom and root cause,
    where lexical overlap between the bug report and root-cause function is low).
    """
    scenarios = []

    # Scenario 1: Inheritance contract bug (3 hops: API view -> Serializer -> BaseField.to_internal)
    repo1_files = {
        "api/views.py": '''
from serializers.user_schema import AccountPayloadSerializer

class UserUpdateEndpoint:
    """Handles PATCH requests for account profile updates."""
    def handle_patch(self, request_dict):
        serializer = AccountPayloadSerializer(data=request_dict)
        if serializer.is_valid():
            return serializer.save()
        raise ValueError("Invalid account update payload")
''',
        "serializers/user_schema.py": '''
from core.fields import DecimalQuotaField, BaseSchemaSerializer

class AccountPayloadSerializer(BaseSchemaSerializer):
    """Validates user quota and billing tier attributes."""
    def __init__(self, data):
        super().__init__(data)
        self.quota_field = DecimalQuotaField(allow_null=True)

    def validate_fields(self):
        return self.quota_field.run_parse(self.raw_data.get("quota"))
''',
        "core/fields.py": '''
from core.coercion import normalize_numeric_primitive

class BaseField:
    """Abstract field contract for schema deserialization."""
    def __init__(self, allow_null=False):
        self.allow_null = allow_null

    def run_parse(self, value):
        cleaned = normalize_numeric_primitive(value)
        if cleaned is None and not self.allow_null:
            raise TypeError("Null value prohibited")
        return cleaned

class DecimalQuotaField(BaseField):
    """Specialized field for storage quota limits."""
    def format_display(self, val):
        return f"{val:.2f} GB"
''',
        "core/coercion.py": '''
def normalize_numeric_primitive(raw_val):
    """Coerces string or numeric primitives; strips whitespace before float conversion."""
    if raw_val is None:
        return 0.0  # BUG: returns 0.0 instead of preserving None when input is None/empty string
    if isinstance(raw_val, str) and raw_val.strip() == "":
        return 0.0
    return float(raw_val)
''',
        "utils/logging_helper.py": '''
def log_patch_request(request_dict):
    """Logs incoming PATCH request dictionaries for audit trail."""
    return f"PATCH keys: {list(request_dict.keys())}"
''',
        "tests/test_user_patch.py": '''
from api.views import UserUpdateEndpoint

def test_null_quota_patch():
    endpoint = UserUpdateEndpoint()
    res = endpoint.handle_patch({"quota": None})
    assert res is None
''',
    }
    scenarios.append({
        "id": "SWE-HOP-01",
        "category": "Inheritance & Coercion Chain (3-Hop)",
        "files": repo1_files,
        "query": (
            "Bug Report: UserUpdateEndpoint.handle_patch silently overwrites user storage quota "
            "with 0.0 when a PATCH request passes {'quota': None}, even though AccountPayloadSerializer "
            "configures DecimalQuotaField(allow_null=True)."
        ),
        "ground_truth": {
            "core/coercion.py::normalize_numeric_primitive",
            "core/fields.py::BaseField.run_parse",
        },
        "hops": 3,
    })

    # Scenario 2: Stack trace exception in query compiler caused by upstream AST dialect visitor (3 hops)
    repo2_files = {
        "orm/session.py": '''
from orm.compiler import SQLQueryCompiler

class DatabaseSession:
    """Executes compiled ORM queries against the backend connection."""
    def execute_select(self, query_ast):
        compiler = SQLQueryCompiler()
        sql, params = compiler.compile_query(query_ast)
        return self._run_raw(sql, params)

    def _run_raw(self, sql, params):
        return (sql, params)
''',
        "orm/compiler.py": '''
from orm.expressions import WindowFrameNode

class SQLQueryCompiler:
    """Translates high-level query AST into parameterized SQL strings."""
    def compile_query(self, query_ast):
        frame = WindowFrameNode(query_ast)
        clause = frame.render_partition_clause()
        return f"SELECT * OVER ({clause})", frame.bound_params
''',
        "orm/expressions.py": '''
from orm.dialect_escape import quote_identifier_token

class BaseExpressionNode:
    """Base AST expression node with parameter binding."""
    def __init__(self, spec):
        self.spec = spec
        self.bound_params = []

    def resolve_column_ref(self, col_name):
        return quote_identifier_token(col_name)

class WindowFrameNode(BaseExpressionNode):
    """Represents SQL OVER (PARTITION BY ...) window specification."""
    def render_partition_clause(self):
        col = self.resolve_column_ref(self.spec.get("partition_by"))
        return f"PARTITION BY {col}"
''',
        "orm/dialect_escape.py": '''
def quote_identifier_token(identifier):
    """Escapes schema.table.column identifiers for ANSI SQL dialect."""
    # BUG: splits on '.' without handling tuple/list composite partition keys
    parts = identifier.split(".")
    return ".".join(f'"{p}"' for p in parts)
''',
        "orm/pool.py": '''
class ConnectionPool:
    """Manages SQLite connection pooling."""
    def acquire_connection(self):
        return True
''',
    }
    scenarios.append({
        "id": "SWE-HOP-02",
        "category": "Stack Trace AST Dialect Propagation (3-Hop)",
        "files": repo2_files,
        "query": (
            "Traceback (most recent call last):\n"
            "  File \"orm/session.py\", line 7, in execute_select\n"
            "    sql, params = compiler.compile_query(query_ast)\n"
            "AttributeError: 'list' object has no attribute 'split' when passing composite "
            "partition_by keys to DatabaseSession.execute_select."
        ),
        "ground_truth": {
            "orm/dialect_escape.py::quote_identifier_token",
            "orm/expressions.py::BaseExpressionNode.resolve_column_ref",
        },
        "hops": 3,
    })

    # Scenario 3: Cache invalidation stale read across decorator -> registry -> key canonicalizer (3 hops)
    repo3_files = {
        "services/catalog.py": '''
from cache.decorators import memoize_endpoint

class ProductCatalogService:
    """Fetches pricing tiers for catalog items."""
    @memoize_endpoint(ttl=300)
    def get_item_pricing(self, sku_id, currency="USD", include_tax=False):
        return {"sku": sku_id, "currency": currency, "tax": include_tax}
''',
        "cache/decorators.py": '''
from cache.backend import InMemoryStore

def memoize_endpoint(ttl=60):
    """Decorator that caches method outputs in InMemoryStore."""
    store = InMemoryStore(ttl=ttl)
    def wrapper(fn):
        def inner(*args, **kwargs):
            return store.get_or_compute(fn, args, kwargs)
        return inner
    return wrapper
''',
        "cache/backend.py": '''
from cache.key_builder import derive_digest_key

class InMemoryStore:
    """Thread-local dictionary store with TTL expiration."""
    def __init__(self, ttl=60):
        self.ttl = ttl
        self.bucket = {}

    def get_or_compute(self, fn, args, kwargs):
        key = derive_digest_key(fn.__name__, args, kwargs)
        if key not in self.bucket:
            self.bucket[key] = fn(*args, **kwargs)
        return self.bucket[key]
''',
        "cache/key_builder.py": '''
def derive_digest_key(fn_name, positional_args, keyword_args):
    """Builds deterministic cache key from function arguments."""
    # BUG: ignores keyword_args values, only hashing positional_args and kwarg keys!
    pos_part = ":".join(str(a) for a in positional_args[1:])
    kw_part = ",".join(sorted(keyword_args.keys()))
    return f"{fn_name}|{pos_part}|{kw_part}"
''',
        "services/discounts.py": '''
def compute_seasonal_discount(price):
    """Applies flat 10% promotional discount."""
    return price * 0.9
''',
    }
    scenarios.append({
        "id": "SWE-HOP-03",
        "category": "Decorator & Key Canonicalization (3-Hop)",
        "files": repo3_files,
        "query": (
            "Bug Report: Calling ProductCatalogService.get_item_pricing('SKU1', include_tax=True) "
            "followed by get_item_pricing('SKU1', include_tax=False) returns the cached tax=True "
            "response instead of recomputing."
        ),
        "ground_truth": {
            "cache/key_builder.py::derive_digest_key",
            "cache/backend.py::InMemoryStore.get_or_compute",
        },
        "hops": 3,
    })

    # Scenario 4: Event bus async retry loop dropping exception context via middleware policy (3 hops)
    repo4_files = {
        "workers/dispatcher.py": '''
from workers.pipeline import MiddlewareChain

class WebhookDispatcher:
    """Dispatches outbound webhook events through configured middleware."""
    def dispatch_event(self, payload):
        chain = MiddlewareChain()
        return chain.execute_with_policy(payload)
''',
        "workers/pipeline.py": '''
from workers.retry_policy import ExponentialBackoffPolicy

class MiddlewareChain:
    """Runs pre-hooks and delegates execution to retry policy."""
    def __init__(self):
        self.policy = ExponentialBackoffPolicy(max_attempts=3)

    def execute_with_policy(self, payload):
        return self.policy.run_attempt_loop(payload)
''',
        "workers/retry_policy.py": '''
from workers.classifier import is_transient_failure

class BaseRetryPolicy:
    """Abstract retry policy contract."""
    def should_retry(self, exc):
        return is_transient_failure(exc)

class ExponentialBackoffPolicy(BaseRetryPolicy):
    """Retries transient transport errors with exponential backoff."""
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def run_attempt_loop(self, payload):
        for attempt in range(self.max_attempts):
            try:
                return self._send(payload)
            except Exception as exc:
                if not self.should_retry(exc):
                    raise
        return None

    def _send(self, payload):
        raise TimeoutError("Gateway 504 timeout")
''',
        "workers/classifier.py": '''
def is_transient_failure(exc):
    """Determines whether an exception represents a retryable network condition."""
    # BUG: checks ConnectionResetError only, missing TimeoutError subclass hierarchy
    return isinstance(exc, ConnectionResetError)
''',
        "workers/metrics.py": '''
def record_webhook_latency(ms):
    return ms
''',
    }
    scenarios.append({
        "id": "SWE-HOP-04",
        "category": "Exception Hierarchy & Retry Policy (3-Hop)",
        "files": repo4_files,
        "query": (
            "Bug Report: WebhookDispatcher.dispatch_event immediately aborts on the first "
            "504 Gateway TimeoutError without performing any retries, even though max_attempts=3 "
            "is configured in MiddlewareChain."
        ),
        "ground_truth": {
            "workers/classifier.py::is_transient_failure",
            "workers/retry_policy.py::BaseRetryPolicy.should_retry",
        },
        "hops": 3,
    })

    # Scenario 5: Tensor shape broadcasting bug in attention mask prepare (2-Hop)
    repo5_files = {
        "modeling/decoder.py": '''
from modeling.attention import CausalSelfAttention

class TransformerBlock:
    """Single decoder layer combining causal attention and MLP."""
    def __init__(self, d_model):
        self.attn = CausalSelfAttention(d_model)

    def forward_block(self, hidden_states, padding_mask):
        return self.attn.compute_attention(hidden_states, padding_mask)
''',
        "modeling/attention.py": '''
from modeling.masking import expand_causal_padding_mask

class CausalSelfAttention:
    """Multi-head causal self-attention module."""
    def __init__(self, d_model):
        self.d_model = d_model

    def compute_attention(self, x, padding_mask):
        mask_4d = expand_causal_padding_mask(padding_mask, seq_len=len(x))
        return mask_4d
''',
        "modeling/masking.py": '''
def expand_causal_padding_mask(mask_2d, seq_len):
    """Converts 2D [B, S] boolean mask into 4D [B, 1, S, S] additive attention bias."""
    # BUG: inverts boolean mask convention (0.0 for masked tokens, -inf for valid tokens)
    return [[-1e9 if token_valid else 0.0 for token_valid in row] for row in mask_2d]
''',
        "modeling/rope.py": '''
def apply_rotary_pos_emb(q, k):
    """Applies RoPE embeddings to query and key tensors."""
    return q, k
''',
    }
    scenarios.append({
        "id": "SWE-HOP-05",
        "category": "Tensor Mask Polarity Inversion (2-Hop)",
        "files": repo5_files,
        "query": (
            "Bug Report: TransformerBlock.forward_block attends exclusively to PAD tokens "
            "and masks out all non-padding tokens when a boolean padding_mask is supplied."
        ),
        "ground_truth": {
            "modeling/masking.py::expand_causal_padding_mask",
            "modeling/attention.py::CausalSelfAttention.compute_attention",
        },
        "hops": 2,
    })

    # Scenario 6: CLI config merge overriding CLI flags with default env file (3-Hop)
    repo6_files = {
        "cli/entrypoint.py": '''
from config.loader import RuntimeConfigLoader

class AgentCLI:
    """Main command-line interface for launching the local agent."""
    def run_cli(self, cli_Flags):
        loader = RuntimeConfigLoader()
        cfg = loader.resolve_effective_config(cli_Flags)
        return cfg
''',
        "config/loader.py": '''
from config.merger import deep_merge_dicts

class RuntimeConfigLoader:
    """Loads defaults, environment variables, and user CLI overrides."""
    def resolve_effective_config(self, cli_overrides):
        defaults = {"temperature": 0.7, "max_tokens": 2048}
        return deep_merge_dicts(cli_overrides, defaults)
''',
        "config/merger.py": '''
def deep_merge_dicts(base_dict, override_dict):
    """Recursively merges override_dict on top of base_dict."""
    merged = dict(base_dict)
    for k, v in override_dict.items():
        if v is not None:
            merged[k] = v
    return merged
''',
        "cli/formatter.py": '''
def format_banner(version):
    return f"Agent v{version}"
''',
    }
    scenarios.append({
        "id": "SWE-HOP-06",
        "category": "Precedence Inversion in Config Loader (2-Hop)",
        "files": repo6_files,
        "query": (
            "Bug Report: Passing --temperature 0.0 to AgentCLI.run_cli has no effect; "
            "the resolved config always reverts to the default temperature of 0.7."
        ),
        "ground_truth": {
            "config/loader.py::RuntimeConfigLoader.resolve_effective_config",
            "config/merger.py::deep_merge_dicts",
        },
        "hops": 2,
    })

    # Scenario 7: Graph serializer cycle infinite recursion (3-Hop)
    repo7_files = {
        "export/bundle.py": '''
from export.serializer import GraphJSONExporter

class WorkspaceBundler:
    """Packages workspace dependency graph for offline persistence."""
    def create_archive(self, root_entity):
        exporter = GraphJSONExporter()
        return exporter.serialize_root(root_entity)
''',
        "export/serializer.py": '''
from export.traversal import collect_reachable_nodes

class GraphJSONExporter:
    """Serializes entity reference graphs into JSON dictionaries."""
    def serialize_root(self, root_entity):
        nodes = collect_reachable_nodes(root_entity)
        return [n.name for n in nodes]
''',
        "export/traversal.py": '''
def collect_reachable_nodes(start_node, visited=None):
    """Depth-first traversal over cyclic entity references."""
    # BUG: appends children before adding current node to visited set, causing RecursionError on self-loops
    if visited is None:
        visited = set()
    result = [start_node]
    for child in getattr(start_node, "children", []):
        if child not in visited:
            result.extend(collect_reachable_nodes(child, visited))
    visited.add(start_node)
    return result
''',
        "export/compression.py": '''
def gzip_bytes(data):
    return data
''',
    }
    scenarios.append({
        "id": "SWE-HOP-07",
        "category": "DFS Cycle Visited-Set Ordering (2-Hop)",
        "files": repo7_files,
        "query": (
            "Traceback (most recent call last):\n"
            "  File \"export/bundle.py\", line 7, in create_archive\n"
            "    return exporter.serialize_root(root_entity)\n"
            "RecursionError: maximum recursion depth exceeded when WorkspaceBundler.create_archive "
            "encounters bidirectional parent-child entity links."
        ),
        "ground_truth": {
            "export/traversal.py::collect_reachable_nodes",
            "export/serializer.py::GraphJSONExporter.serialize_root",
        },
        "hops": 2,
    })

    # Scenario 8: OAuth token refresh clock-skew boundary failure (3-Hop)
    repo8_files = {
        "auth/client.py": '''
from auth.session_manager import TokenSessionManager

class AuthenticatedHTTPClient:
    """Executes API calls with automatic bearer token renewal."""
    def __init__(self):
        self.mgr = TokenSessionManager()

    def request_resource(self, url, now_ts):
        token = self.mgr.ensure_active_token(now_ts)
        return f"Bearer {token}"
''',
        "auth/session_manager.py": '''
from auth.validators import is_jwt_expired

class BaseSessionManager:
    """Base credential lifecycle handler."""
    def check_needs_refresh(self, expiry_ts, now_ts):
        return is_jwt_expired(expiry_ts, now_ts)

class TokenSessionManager(BaseSessionManager):
    """Manages OAuth2 access token caching and refresh."""
    def __init__(self):
        self.cached_token = "tok_initial"
        self.expiry_ts = 1000

    def ensure_active_token(self, now_ts):
        if self.check_needs_refresh(self.expiry_ts, now_ts):
            self.cached_token = "tok_refreshed"
        return self.cached_token
''',
        "auth/validators.py": '''
def is_jwt_expired(expiry_ts, current_ts, leeway_seconds=30):
    """Checks whether current timestamp is within leeway window of token expiration."""
    # BUG: adds leeway_seconds to expiry_ts instead of subtracting it, delaying refresh!
    return current_ts >= (expiry_ts + leeway_seconds)
''',
        "auth/crypto.py": '''
def verify_hmac_signature(payload, secret):
    return True
''',
    }
    scenarios.append({
        "id": "SWE-HOP-08",
        "category": "Leeway Sign Error via Base Class (3-Hop)",
        "files": repo8_files,
        "query": (
            "Bug Report: AuthenticatedHTTPClient.request_resource continues sending expired "
            "bearer tokens for 30 seconds after expiration instead of proactively refreshing "
            "30 seconds before expiry."
        ),
        "ground_truth": {
            "auth/validators.py::is_jwt_expired",
            "auth/session_manager.py::BaseSessionManager.check_needs_refresh",
        },
        "hops": 3,
    })

    # Scenario 9: Streaming chunk UTF-8 multi-byte split corruption (3-Hop)
    repo9_files = {
        "streaming/consumer.py": '''
from streaming.parser import SSEEventParser

class StreamResponseReader:
    """Reads server-sent events from chunked byte streams."""
    def consume_stream(self, byte_chunks):
        parser = SSEEventParser()
        return parser.feed_chunks(byte_chunks)
''',
        "streaming/parser.py": '''
from streaming.decoder import IncrementalUTF8Decoder

class SSEEventParser:
    """Parses raw byte chunks into SSE text frames."""
    def __init__(self):
        self.decoder = IncrementalUTF8Decoder()

    def feed_chunks(self, chunks):
        out = []
        for ch in chunks:
            out.append(self.decoder.decode_frame(ch))
        return "".join(out)
''',
        "streaming/decoder.py": '''
class IncrementalUTF8Decoder:
    """Decodes arbitrary byte slices into UTF-8 strings."""
    def decode_frame(self, raw_bytes):
        # BUG: decodes each chunk independently without buffering incomplete multi-byte sequences
        return raw_bytes.decode("utf-8", errors="strict")
''',
        "streaming/heartbeat.py": '''
def is_ping_frame(line):
    return line.startswith(": ping")
''',
    }
    scenarios.append({
        "id": "SWE-HOP-09",
        "category": "Streaming Multi-Byte Boundary Split (2-Hop)",
        "files": repo9_files,
        "query": (
            "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0 in position 63: "
            "unexpected end of data when StreamResponseReader.consume_stream receives "
            "emoji or CJK characters split across TCP packet chunks."
        ),
        "ground_truth": {
            "streaming/decoder.py::IncrementalUTF8Decoder.decode_frame",
            "streaming/parser.py::SSEEventParser.feed_chunks",
        },
        "hops": 2,
    })

    # Scenario 10: Git diff hunk line offset drift across multiple hunks (3-Hop)
    repo10_files = {
        "patcher/workspace_editor.py": '''
from patcher.applier import UnifiedDiffApplier

class WorkspacePatchService:
    """Applies LLM-generated unified diffs to workspace files."""
    def apply_patch_to_file(self, original_text, hunks):
        applier = UnifiedDiffApplier()
        return applier.apply_hunks(original_text, hunks)
''',
        "patcher/applier.py": '''
from patcher.splicer import splice_line_range

class UnifiedDiffApplier:
    """Iterates over diff hunks and splices replacements into file lines."""
    def apply_hunks(self, text, hunks):
        lines = text.splitlines()
        for start_line, delete_count, new_lines in hunks:
            lines = splice_line_range(lines, start_line, delete_count, new_lines)
        return "\\n".join(lines)
''',
        "patcher/splicer.py": '''
def splice_line_range(lines, start_1based, delete_count, replacement_lines):
    """Replaces lines[start_1based-1 : start_1based-1+delete_count] with replacement_lines."""
    idx = start_1based - 1
    return lines[:idx] + list(replacement_lines) + lines[idx + delete_count:]
''',
        "patcher/linter.py": '''
def check_trailing_whitespace(text):
    return True
''',
    }
    scenarios.append({
        "id": "SWE-HOP-10",
        "category": "Multi-Hunk Offset Drift in Patch Applier (2-Hop)",
        "files": repo10_files,
        "query": (
            "Bug Report: WorkspacePatchService.apply_patch_to_file corrupts files whenever "
            "a diff contains two or more hunks and the first hunk changes the total line count."
        ),
        "ground_truth": {
            "patcher/applier.py::UnifiedDiffApplier.apply_hunks",
            "patcher/splicer.py::splice_line_range",
        },
        "hops": 2,
    })

    # Scenario 11: Pagination cursor off-by-one duplicate record across BasePaginator (3-Hop)
    repo11_files = {
        "api/endpoints.py": '''
from pagination.cursor import KeysetPaginator

class AuditLogController:
    """Lists organization audit logs using keyset pagination."""
    def list_logs(self, records, after_id, limit):
        paginator = KeysetPaginator(limit=limit)
        return paginator.fetch_page(records, after_id)
''',
        "pagination/cursor.py": '''
from pagination.slicer import slice_after_cursor

class BasePaginator:
    """Abstract cursor pagination helper."""
    def apply_cursor_filter(self, items, cursor_id):
        return slice_after_cursor(items, cursor_id)

class KeysetPaginator(BasePaginator):
    """Keyset paginator ordered by monotonically increasing record ID."""
    def __init__(self, limit=20):
        self.limit = limit

    def fetch_page(self, records, after_id):
        filtered = self.apply_cursor_filter(records, after_id)
        return filtered[:self.limit]
''',
        "pagination/slicer.py": '''
def slice_after_cursor(sorted_records, cursor_id):
    """Returns all records strictly following cursor_id."""
    if cursor_id is None:
        return sorted_records
    # BUG: uses >= instead of >, duplicating the boundary cursor item on the next page
    return [r for r in sorted_records if r["id"] >= cursor_id]
''',
        "api/serializers.py": '''
def serialize_log_entry(entry):
    return entry
''',
    }
    scenarios.append({
        "id": "SWE-HOP-11",
        "category": "Keyset Cursor Boundary Duplication (3-Hop)",
        "files": repo11_files,
        "query": (
            "Bug Report: AuditLogController.list_logs repeats the last item of page N "
            "as the first item of page N+1 when passing after_id."
        ),
        "ground_truth": {
            "pagination/slicer.py::slice_after_cursor",
            "pagination/cursor.py::BasePaginator.apply_cursor_filter",
        },
        "hops": 3,
    })

    # Scenario 12: LoRA adapter gradient scaling missing alpha/r factor (3-Hop)
    repo12_files = {
        "trainer/loop.py": '''
from peft.linear import LoRALinearLayer

class LocalFineTuner:
    """Runs on-device LoRA fine-tuning step for Gemma blocks."""
    def forward_step(self, x, base_w, lora_a, lora_b, rank, alpha):
        layer = LoRALinearLayer(base_w, lora_a, lora_b, rank=rank, alpha=alpha)
        return layer.compute_projection(x)
''',
        "peft/linear.py": '''
from peft.scaling import compute_adapter_delta

class BaseAdapterLayer:
    """Base parameter-efficient adapter interface."""
    def eval_delta(self, x, lora_a, lora_b, rank, alpha):
        return compute_adapter_delta(x, lora_a, lora_b, rank, alpha)

class LoRALinearLayer(BaseAdapterLayer):
    """Low-Rank Adaptation wrapper around frozen linear weights."""
    def __init__(self, base_w, lora_a, lora_b, rank=8, alpha=16):
        self.base_w = base_w
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.rank = rank
        self.alpha = alpha

    def compute_projection(self, x):
        base_out = x * self.base_w
        delta = self.eval_delta(x, self.lora_a, self.lora_b, self.rank, self.alpha)
        return base_out + delta
''',
        "peft/scaling.py": '''
def compute_adapter_delta(x, lora_a, lora_b, rank, alpha):
    """Computes scaled low-rank update (alpha / rank) * (x @ A @ B)."""
    # BUG: divides by alpha instead of multiplying by (alpha / rank)
    scale = 1.0 / max(1.0, float(alpha))
    return x * lora_a * lora_b * scale
''',
        "trainer/lr_scheduler.py": '''
def cosine_warmup_step(step, total_steps):
    return 1e-4
''',
    }
    scenarios.append({
        "id": "SWE-HOP-12",
        "category": "Adapter Scaling Factor Inversion (3-Hop)",
        "files": repo12_files,
        "query": (
            "Bug Report: Increasing lora_alpha from 16 to 64 in LocalFineTuner.forward_step "
            "causes the adapter update magnitude to shrink by 4x instead of increasing by 4x."
        ),
        "ground_truth": {
            "peft/scaling.py::compute_adapter_delta",
            "peft/linear.py::BaseAdapterLayer.eval_delta",
        },
        "hops": 3,
    })

    return scenarios


def rank_bm25_only(graph: HeterogeneousCodeGraph, query: str) -> List[str]:
    s0, _ = graph.compute_seed_distribution(query)
    return sorted(s0.keys(), key=lambda k: s0[k], reverse=True)


def rank_cosine_rag(graph: HeterogeneousCodeGraph, query: str) -> List[str]:
    q_tokens = set(graph.tokenize_identifier(query))
    scores: Dict[str, float] = {}
    for nid, node in graph.nodes.items():
        d_tokens = set(
            graph.tokenize_identifier(f"{node.symbol_name} {node.signature} {node.docstring} {node.body_text}")
        )
        if not q_tokens or not d_tokens:
            scores[nid] = 0.0
        else:
            scores[nid] = len(q_tokens & d_tokens) / math.sqrt(len(q_tokens) * len(d_tokens))
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


def rank_uniform_pagerank(graph: HeterogeneousCodeGraph, query: str, damping: float = 0.85) -> List[str]:
    """Uniform undirected PageRank without relation-specific conductance or strong query teleportation."""
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    s0, _ = graph.compute_seed_distribution(query)
    # Standard RepoMap uses weak personalization (15% teleportation) and uniform edge weights
    p = {u: 1.0 / n for u in node_ids}
    for _ in range(30):
        next_p = {u: (1.0 - damping) * (0.5 * (1.0 / n) + 0.5 * s0[u]) for u in node_ids}
        for u in node_ids:
            nbrs = graph.adj.get(u, [])
            if nbrs:
                share = damping * p[u] / len(nbrs)
                for v, _, _ in nbrs:
                    next_p[v] += share
        p = next_p
    return sorted(p.keys(), key=lambda k: p[k], reverse=True)


def rank_prism_rwppr(graph: HeterogeneousCodeGraph, query: str) -> List[str]:
    ppr = graph.run_rw_ppr(query, damping=0.72)
    # Filter module header pseudo-nodes below concrete symbol definitions
    return sorted(
        ppr.keys(),
        key=lambda k: (0 if graph.nodes[k].node_type == "module" else 1, ppr[k]),
        reverse=True,
    )


def compute_metrics(ranked: List[str], gt: Set[str], root_cause_id: str) -> Dict[str, float]:
    # Filter out module pseudo-nodes for fair symbol-level comparison across all methods
    symbol_ranked = [nid for nid in ranked if not nid.endswith("::<module>")]
    hits_at_1 = len(set(symbol_ranked[:1]) & gt) / len(gt)
    hits_at_3 = len(set(symbol_ranked[:3]) & gt) / len(gt)
    hits_at_5 = len(set(symbol_ranked[:5]) & gt) / len(gt)

    rr = 0.0
    for idx, nid in enumerate(symbol_ranked):
        if nid in gt:
            rr = 1.0 / (idx + 1)
            break

    root_in_top3 = 1.0 if root_cause_id in symbol_ranked[:3] else 0.0
    root_rank = (symbol_ranked.index(root_cause_id) + 1) if root_cause_id in symbol_ranked else len(symbol_ranked)
    return {
        "recall@1": hits_at_1,
        "recall@3": hits_at_3,
        "recall@5": hits_at_5,
        "mrr": rr,
        "root_cause_recall@3": root_in_top3,
        "root_cause_rank": float(root_rank),
    }


def evaluate_kv_cache_prefix_reuse(scenarios: List[Dict]) -> Dict[str, float]:
    """
    Simulates a 5-turn iterative debugging & patch refinement session on consumer hardware.
    Across turns t=1..5, the agent refines its query/focus around the downstream buggy function
    and edits the target node body.
    Measures the fraction of prompt context tokens in the static prefix that remain byte-identical
    (KV-cache hit rate) between consecutive turns under:
      1. Relevance-Descending Order (highest score first -> mutable target at index 0 -> invalidates entire KV cache!)
      2. Alphabetical File Order
      3. PRISM Topo-KV Order (upstream contracts & base classes first, mutable edit target at suffix)
    """
    turn_queries_suffix = [
        "",
        " inspect intermediate caller arguments",
        " check None coercion and boundary condition",
        " draft candidate fix in target function",
        " verify post-patch return contract",
    ]

    reuse_scores = {"relevance_desc": [], "alphabetical": [], "prism_topo_kv": []}

    for sc in scenarios:
        g = HeterogeneousCodeGraph()
        g.build_from_virtual_files(sc["files"])
        root_target = sorted(sc["ground_truth"])[0]

        for policy in reuse_scores:
            prev_tokens: List[str] = []
            turn_hits: List[float] = []
            for turn_idx, suffix in enumerate(turn_queries_suffix):
                q = sc["query"] + suffix
                ppr = g.run_rw_ppr(q)
                chosen = [
                    nid for nid in sorted(ppr.keys(), key=lambda k: ppr[k], reverse=True)
                    if g.nodes[nid].node_type != "module"
                ][:5]

                if policy == "relevance_desc":
                    ordered = sorted(chosen, key=lambda k: ppr[k], reverse=True)
                elif policy == "alphabetical":
                    ordered = sorted(chosen)
                else:
                    # PRISM Topo-KV: upstream dependencies first, edited root_target at suffix
                    ordered = g._topological_prefix_sort(chosen, ppr, set())
                    if root_target in ordered:
                        ordered.remove(root_target)
                        ordered.append(root_target)

                blocks = []
                for nid in ordered:
                    body = g.nodes[nid].body_text
                    if nid == root_target and turn_idx >= 2:
                        # Simulate iterative patch edits to the target function across turns 2..4
                        body = body + f"\n    # candidate_patch_revision_{turn_idx}"
                    blocks.append(f"### {nid}\n{body}")

                full_prompt_tokens = "\n".join(blocks).split()
                if prev_tokens:
                    common_prefix = 0
                    for a, b in zip(prev_tokens, full_prompt_tokens):
                        if a == b:
                            common_prefix += 1
                        else:
                            break
                    hit_rate = common_prefix / max(1, len(full_prompt_tokens))
                    turn_hits.append(hit_rate)
                prev_tokens = full_prompt_tokens

            reuse_scores[policy].append(sum(turn_hits) / len(turn_hits))

    return {
        k: round(100.0 * sum(v) / len(v), 2)
        for k, v in reuse_scores.items()
    }


def evaluate_grpo_reward_shaping(scenarios: List[Dict]) -> Dict[str, float]:
    """
    Evaluates 6 sampled rollouts per scenario (mimicking GRPO group size G=6 on Gemma 4 9B):
      - Rollout 1: Cyclic grep on entry file, syntax error patch, test fails
      - Rollout 2: Stays at 2-hop symptom node, valid AST patch, test fails
      - Rollout 3: Reaches 1-hop intermediate caller, valid AST patch, test fails
      - Rollout 4: Reaches 0-hop root cause node, syntax error patch, test fails
      - Rollout 5: Reaches 0-hop root cause node, valid AST partial fix, test fails
      - Rollout 6: Reaches 0-hop root cause node, valid AST fix, test passes
    Compares Sparse Binary Outcome Reward vs. PRISM Graph-Proximity Reward Shaping (GPRS).
    """
    sparse_non_zero_groups = 0
    gprs_non_zero_groups = 0
    corr_with_oracle_quality = []

    for sc in scenarios:
        g = HeterogeneousCodeGraph()
        g.build_from_virtual_files(sc["files"])
        gt = sc["ground_truth"]
        root_node = sorted(gt)[0]
        all_nodes = [k for k in g.nodes if not k.endswith("::<module>")]
        entry_node = all_nodes[0]

        # Find a 1-hop neighbor of gt
        one_hop = entry_node
        for cand in all_nodes:
            if g.shortest_path_distance(cand, gt) == 1:
                one_hop = cand
                break

        rollouts = [
            {"nodes": [entry_node, entry_node, entry_node], "code": "def broken(:", "pass": False, "oracle_rank": 1},
            {"nodes": [entry_node], "code": "def valid():\n    return 1", "pass": False, "oracle_rank": 2},
            {"nodes": [entry_node, one_hop], "code": "def valid():\n    return 2", "pass": False, "oracle_rank": 3},
            {"nodes": [entry_node, one_hop, root_node], "code": "def broken(:", "pass": False, "oracle_rank": 4},
            {"nodes": [entry_node, one_hop, root_node], "code": "def valid():\n    return None", "pass": False, "oracle_rank": 5},
            {"nodes": [entry_node, one_hop, root_node], "code": "def valid():\n    return None", "pass": True, "oracle_rank": 6},
        ]

        # Consider hard regime where all 6 rollouts fail unit tests (common early in RL on 4B/9B models)
        hard_sparse = [0.0 for _ in rollouts[:5]]
        hard_gprs = [
            g.compute_grpo_trajectory_reward(r["nodes"], gt, r["code"], False)["r_total"]
            for r in rollouts[:5]
        ]

        if max(hard_sparse) - min(hard_sparse) > 1e-5:
            sparse_non_zero_groups += 1
        if max(hard_gprs) - min(hard_gprs) > 1e-5:
            gprs_non_zero_groups += 1

        # Rank correlation with oracle trajectory quality across the 6 rollouts
        full_gprs = [
            g.compute_grpo_trajectory_reward(r["nodes"], gt, r["code"], r["pass"])["r_total"]
            for r in rollouts
        ]
        # Check monotonic alignment
        pairs = 0
        concordant = 0
        for i in range(len(full_gprs)):
            for j in range(i + 1, len(full_gprs)):
                pairs += 1
                if full_gprs[j] > full_gprs[i]:
                    concordant += 1
        corr_with_oracle_quality.append(concordant / pairs)

    return {
        "hard_regime_informative_gradient_pct_sparse": round(100.0 * sparse_non_zero_groups / len(scenarios), 1),
        "hard_regime_informative_gradient_pct_gprs": round(100.0 * gprs_non_zero_groups / len(scenarios), 1),
        "kendall_concordance_with_oracle_progress": round(sum(corr_with_oracle_quality) / len(corr_with_oracle_quality), 4),
    }


def run_ablation_variant(
    graph: HeterogeneousCodeGraph,
    query: str,
    use_non_backtracking: bool = True,
    use_typed_conductance: bool = True,
    use_sink_boost: bool = True,
    max_hops: int = 3,
) -> List[str]:
    s0, regime = graph.compute_seed_distribution(query)
    edge_weights = (
        graph.DEFAULT_EDGE_WEIGHTS[regime]
        if use_typed_conductance
        else {k: 1.0 for k in graph.DEFAULT_EDGE_WEIGHTS[regime]}
    )
    symbol_ids = [nid for nid, n in graph.nodes.items() if n.node_type != "module"]
    sym_set = set(symbol_ids)
    sym_total = sum(s0[u] for u in symbol_ids) or 1.0
    s_sym = {u: s0[u] / sym_total for u in symbol_ids}

    allowed_types = (
        {"CALLS", "INHERITS", "CONTAINS", "TESTS"}
        if use_non_backtracking
        else set(edge_weights.keys())
    )
    transitions: Dict[str, List[Tuple[str, float]]] = {}
    out_causal_calls: Dict[str, int] = {}

    for u in symbol_ids:
        neighbors = [
            (v, etype, base_w)
            for v, etype, base_w in graph.adj.get(u, [])
            if v in sym_set and etype in allowed_types
        ]
        out_causal_calls[u] = sum(1 for _, et, _ in neighbors if et in ("CALLS", "INHERITS"))
        total_w = sum(base_w * edge_weights.get(et, 1.0) for _, et, base_w in neighbors)
        transitions[u] = (
            [(v, (base_w * edge_weights.get(et, 1.0)) / total_w) for v, et, base_w in neighbors]
            if total_w > 0
            else []
        )

    base_weights = [0.22, 0.24, 0.28, 0.26][: max_hops + 1]
    w_sum = sum(base_weights)
    hop_weights = [w / w_sum for w in base_weights]

    accumulated = {u: hop_weights[0] * s_sym[u] for u in symbol_ids}
    frontier = dict(s_sym)
    prev_frontier = {u: 0.0 for u in symbol_ids}

    for hop_idx in range(1, len(hop_weights)):
        next_frontier = {u: 0.0 for u in symbol_ids}
        hw = hop_weights[hop_idx]
        for u, mass in frontier.items():
            if mass <= 1e-9:
                continue
            for v, prob in transitions[u]:
                bt = (
                    0.25
                    if (use_non_backtracking and prev_frontier.get(v, 0.0) > frontier.get(u, 0.0))
                    else 1.0
                )
                next_frontier[v] += mass * prob * bt
        wave_sum = sum(next_frontier.values())
        if wave_sum > 1e-9:
            for u in symbol_ids:
                accumulated[u] += hw * (next_frontier[u] / wave_sum)
        prev_frontier = frontier
        frontier = next_frontier

    final_scores: Dict[str, float] = {}
    for nid, node in graph.nodes.items():
        if node.node_type == "module":
            final_scores[nid] = 1e-6
            continue
        score = accumulated.get(nid, 1e-6)
        if use_sink_boost:
            if node.node_type in ("function", "method"):
                score *= 1.45 if out_causal_calls.get(nid, 0) == 0 else 1.15
            elif node.node_type == "class":
                score *= 0.65
        final_scores[nid] = score

    return sorted(
        final_scores.keys(),
        key=lambda k: (0 if graph.nodes[k].node_type == "module" else 1, final_scores[k]),
        reverse=True,
    )


def main() -> None:
    scenarios = build_benchmark_repositories()
    methods = {
        "BM25 / Lexical Grep": rank_bm25_only,
        "Dense Token Cosine RAG": rank_cosine_rag,
        "Uniform RepoMap (PageRank)": rank_uniform_pagerank,
        "PRISM-Gemma (Full NBD-PPR)": rank_prism_rwppr,
        "Ablation: w/o Non-Backtracking (Undirected Walk)": lambda g, q: run_ablation_variant(
            g, q, use_non_backtracking=False, use_typed_conductance=True, use_sink_boost=True, max_hops=3
        ),
        "Ablation: w/o Typed Edge Conductance": lambda g, q: run_ablation_variant(
            g, q, use_non_backtracking=True, use_typed_conductance=False, use_sink_boost=True, max_hops=3
        ),
        "Ablation: w/o Causal Sink Amplification": lambda g, q: run_ablation_variant(
            g, q, use_non_backtracking=True, use_typed_conductance=True, use_sink_boost=False, max_hops=3
        ),
        "Ablation: 1-Hop Neighborhood Only (K=1)": lambda g, q: run_ablation_variant(
            g, q, use_non_backtracking=True, use_typed_conductance=True, use_sink_boost=True, max_hops=1
        ),
        "Ablation: 2-Hop Neighborhood Only (K=2)": lambda g, q: run_ablation_variant(
            g, q, use_non_backtracking=True, use_typed_conductance=True, use_sink_boost=True, max_hops=2
        ),
    }

    agg: Dict[str, Dict[str, List[float]]] = {
        m: defaultdict(list) for m in methods
    }

    per_scenario_log = []
    for sc in scenarios:
        g = HeterogeneousCodeGraph()
        g.build_from_virtual_files(sc["files"])
        root_cause_id = sorted(sc["ground_truth"])[0]

        sc_entry = {"id": sc["id"], "category": sc["category"], "results": {}}
        for m_name, fn in methods.items():
            ranked = fn(g, sc["query"])
            metrics = compute_metrics(ranked, sc["ground_truth"], root_cause_id)
            sc_entry["results"][m_name] = metrics
            for k, v in metrics.items():
                agg[m_name][k].append(v)
        per_scenario_log.append(sc_entry)

    summary = {}
    for m_name, m_dict in agg.items():
        summary[m_name] = {
            k: round(sum(vals) / len(vals), 4) for k, vals in m_dict.items()
        }

    kv_stats = evaluate_kv_cache_prefix_reuse(scenarios)
    grpo_stats = evaluate_grpo_reward_shaping(scenarios)

    report = {
        "num_scenarios": len(scenarios),
        "localization_summary": summary,
        "kv_cache_prefix_hit_rate_pct": kv_stats,
        "grpo_reward_shaping_analysis": grpo_stats,
        "per_scenario": per_scenario_log,
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== PRISM-Gemma Benchmark Results ===")
    print(json.dumps(report["localization_summary"], indent=2))
    print("\n=== Multi-Turn KV-Cache Prefix Hit Rate (%) ===")
    print(json.dumps(report["kv_cache_prefix_hit_rate_pct"], indent=2))
    print("\n=== GRPO Reward Shaping Analysis ===")
    print(json.dumps(report["grpo_reward_shaping_analysis"], indent=2))


if __name__ == "__main__":
    main()
