import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class ExecutionSpan:
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, metadata: Optional[Dict[str, Any]] = None):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if metadata:
            self.metadata.update(metadata)

class LLMOpsTracer:
    """
    Lightweight, high-precision LLMOps execution and latency tracer.
    Measures retrieval waterfall, TTFT (Time to First Token), tokens, and cost.
    """
    def __init__(self, query: str = ""):
        self.query = query
        self.start_wall_time = time.time()
        self.start_perf_time = time.perf_counter()
        self.spans: List[ExecutionSpan] = []
        self.ttft_ms: Optional[float] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.model_name: str = ""

    def start_span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> ExecutionSpan:
        span = ExecutionSpan(name=name, start_time=time.perf_counter(), metadata=metadata or {})
        self.spans.append(span)
        return span

    def record_first_token(self):
        """Record Time To First Token (TTFT) for streaming."""
        if self.ttft_ms is None:
            self.ttft_ms = round((time.perf_counter() - self.start_perf_time) * 1000, 2)

    def set_tokens(self, prompt_tokens: int, completion_tokens: int, model_name: str = ""):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model_name = model_name

    def calculate_cost_estimate(self) -> float:
        """Estimate inference cost in USD based on model pricing per million tokens."""
        # Groq models are exceptionally affordable (often < $0.59 / M tokens or free tier)
        price_per_m_in = 0.59
        price_per_m_out = 0.79
        if "20b" in self.model_name.lower():
            price_per_m_in = 0.20
            price_per_m_out = 0.30
        
        in_cost = (self.prompt_tokens / 1_000_000.0) * price_per_m_in
        out_cost = (self.completion_tokens / 1_000_000.0) * price_per_m_out
        return round(in_cost + out_cost, 6)

    def get_summary(self) -> Dict[str, Any]:
        total_latency_ms = round((time.perf_counter() - self.start_perf_time) * 1000, 2)
        spans_summary = []
        for s in self.spans:
            spans_summary.append({
                "name": s.name,
                "duration_ms": s.duration_ms,
                "pct_of_total": round((s.duration_ms / max(total_latency_ms, 0.001)) * 100, 1),
                "metadata": s.metadata
            })

        return {
            "total_latency_ms": total_latency_ms,
            "ttft_ms": self.ttft_ms or total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": self.calculate_cost_estimate(),
            "model_name": self.model_name,
            "spans": spans_summary
        }
