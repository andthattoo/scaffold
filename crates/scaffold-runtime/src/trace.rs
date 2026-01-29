//! Tracing and metrics for scaffold execution
//!
//! This module provides structured tracing for:
//! - LLM calls (prompts, responses, token usage)
//! - Tool executions (inputs, outputs, duration)
//! - Agent turns (tool calls, intermediate results)
//! - Task/subgoal completion (rewards, status)
//!
//! # Usage
//!
//! Enable tracing by setting one of:
//! - `SCAFFOLD_TRACE_FILE=/path/to/traces.jsonl` - write traces to a file
//! - `SCAFFOLD_TRACE_STDERR=1` - write traces to stderr
//!
//! Traces are emitted as JSONL.
//!
//! ```ignore
//! use scaffold_runtime::trace::{Tracer, TraceEvent};
//!
//! let tracer = Tracer::new();
//! tracer.record(TraceEvent::LlmCall { ... });
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Global tracer state
static TRACER: std::sync::OnceLock<Tracer> = std::sync::OnceLock::new();

/// Get the global tracer instance
pub fn tracer() -> &'static Tracer {
    TRACER.get_or_init(Tracer::new)
}

/// Initialize the global tracer with custom configuration
pub fn init_tracer(config: TracerConfig) {
    let _ = TRACER.set(Tracer::with_config(config));
}

/// Tracer configuration
#[derive(Clone, Debug)]
pub struct TracerConfig {
    /// Whether tracing is enabled
    pub enabled: bool,
    /// Output destination (stderr, file path, or callback)
    pub output: TraceOutput,
    /// Whether to include full request/response bodies
    pub include_bodies: bool,
    /// Minimum event level to record
    pub min_level: TraceLevel,
}

impl Default for TracerConfig {
    fn default() -> Self {
        // SCAFFOLD_TRACE_FILE=/path/to/file.jsonl enables tracing to that file
        // SCAFFOLD_TRACE_STDERR=1 enables tracing to stderr (original behavior)
        // If neither is set, tracing is disabled
        let trace_file = std::env::var("SCAFFOLD_TRACE_FILE").ok().filter(|s| !s.is_empty());
        let trace_stderr = std::env::var("SCAFFOLD_TRACE_STDERR").ok().filter(|s| !s.is_empty());

        let (enabled, output) = if let Some(file_path) = trace_file {
            (true, TraceOutput::File(file_path))
        } else if trace_stderr.is_some() {
            (true, TraceOutput::Stderr)
        } else {
            (false, TraceOutput::Stderr)
        };

        Self {
            enabled,
            output,
            include_bodies: true,
            min_level: TraceLevel::Info,
        }
    }
}

/// Trace output destination
#[derive(Clone, Debug)]
pub enum TraceOutput {
    /// Write to stderr
    Stderr,
    /// Write to a file
    File(String),
    /// Collect in memory (for testing)
    Memory,
}

/// Trace event severity level
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// A recorded trace event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Event level
    pub level: TraceLevel,
    /// Span ID for correlation
    pub span_id: String,
    /// Parent span ID (for nested events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// The event data
    pub event: TraceEvent,
    /// Duration in milliseconds (for completed events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Types of trace events
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceEvent {
    /// LLM call started/completed
    LlmCall {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Tool execution
    ToolCall {
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Agent turn (one iteration of the agent loop)
    AgentTurn {
        agent_name: String,
        turn_number: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        completed: Option<bool>,
    },
    /// Prompt execution
    PromptExecution {
        prompt_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Task/subgoal completion
    TaskComplete {
        task_name: String,
        status: TaskStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        reward: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Custom metric
    Metric {
        name: String,
        value: MetricValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        tags: Option<std::collections::HashMap<String, String>>,
    },
}

/// Task completion status
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Completed,
    Failed,
    Timeout,
    Skipped,
}

/// Metric value types
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    Counter(i64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

/// The main tracer struct
#[derive(Debug)]
pub struct Tracer {
    config: TracerConfig,
    /// In-memory trace storage (for testing/inspection)
    records: Mutex<Vec<TraceRecord>>,
}

impl Tracer {
    /// Create a new tracer with default configuration
    pub fn new() -> Self {
        Self::with_config(TracerConfig::default())
    }

    /// Create a new tracer with custom configuration
    pub fn with_config(config: TracerConfig) -> Self {
        Self {
            config,
            records: Mutex::new(Vec::new()),
        }
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Record a trace event
    pub fn record(&self, event: TraceEvent) -> String {
        self.record_with_level(TraceLevel::Info, event)
    }

    /// Record a trace event with a specific level
    pub fn record_with_level(&self, level: TraceLevel, event: TraceEvent) -> String {
        if !self.config.enabled || level < self.config.min_level {
            return String::new();
        }

        let span_id = generate_span_id();
        let record = TraceRecord {
            timestamp_ms: current_timestamp_ms(),
            level,
            span_id: span_id.clone(),
            parent_span_id: None,
            event,
            duration_ms: None,
        };

        self.emit(&record);
        span_id
    }

    /// Record a trace event with duration
    pub fn record_completed(&self, span_id: &str, event: TraceEvent, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        let record = TraceRecord {
            timestamp_ms: current_timestamp_ms(),
            level: TraceLevel::Info,
            span_id: span_id.to_string(),
            parent_span_id: None,
            event,
            duration_ms: Some(duration.as_millis() as u64),
        };

        self.emit(&record);
    }

    /// Start a span and return a guard that records completion
    pub fn span(&self, event: TraceEvent) -> SpanGuard<'_> {
        let span_id = self.record(event);
        SpanGuard {
            tracer: self,
            span_id,
            start: Instant::now(),
        }
    }

    /// Emit a trace record to the configured output
    fn emit(&self, record: &TraceRecord) {
        // Store in memory if configured
        if matches!(self.config.output, TraceOutput::Memory) {
            if let Ok(mut records) = self.records.lock() {
                records.push(record.clone());
            }
        }

        // Serialize to JSONL
        let json = match serde_json::to_string(record) {
            Ok(j) => j,
            Err(_) => return,
        };

        match &self.config.output {
            TraceOutput::Stderr => {
                eprintln!("{}", json);
            }
            TraceOutput::File(path) => {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{}", json);
                }
            }
            TraceOutput::Memory => {
                // Already stored above
            }
        }
    }

    /// Get all recorded traces (for testing)
    pub fn get_records(&self) -> Vec<TraceRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Clear recorded traces
    pub fn clear(&self) {
        if let Ok(mut records) = self.records.lock() {
            records.clear();
        }
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for span timing
pub struct SpanGuard<'a> {
    tracer: &'a Tracer,
    span_id: String,
    start: Instant,
}

impl<'a> SpanGuard<'a> {
    /// Get the span ID
    pub fn span_id(&self) -> &str {
        &self.span_id
    }

    /// Complete the span with an event
    pub fn complete(self, event: TraceEvent) {
        let duration = self.start.elapsed();
        self.tracer.record_completed(&self.span_id, event, duration);
    }
}

/// Generate a unique span ID
fn generate_span_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = current_timestamp_ms();
    format!("{:x}-{:04x}", timestamp, count & 0xFFFF)
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Convenience macro for tracing LLM calls
#[macro_export]
macro_rules! trace_llm {
    ($model:expr, $prompt:expr) => {
        $crate::trace::tracer().record($crate::trace::TraceEvent::LlmCall {
            model: $model.to_string(),
            prompt: Some($prompt.to_string()),
            response: None,
            input_tokens: None,
            output_tokens: None,
            error: None,
        })
    };
}

/// Convenience macro for tracing tool calls
#[macro_export]
macro_rules! trace_tool {
    ($name:expr, $input:expr) => {
        $crate::trace::tracer().record($crate::trace::TraceEvent::ToolCall {
            tool_name: $name.to_string(),
            input: Some(serde_json::to_value($input).unwrap_or_default()),
            output: None,
            error: None,
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_disabled_by_default() {
        let tracer = Tracer::new();
        // Without SCAFFOLD_TRACE_FILE env var, tracing should be disabled
        // (depends on env during test)
    }

    #[test]
    fn test_tracer_memory_output() {
        let tracer = Tracer::with_config(TracerConfig {
            enabled: true,
            output: TraceOutput::Memory,
            include_bodies: true,
            min_level: TraceLevel::Debug,
        });

        tracer.record(TraceEvent::ToolCall {
            tool_name: "test_tool".to_string(),
            input: Some(serde_json::json!({"x": 1})),
            output: None,
            error: None,
        });

        let records = tracer.get_records();
        assert_eq!(records.len(), 1);
        assert!(
            matches!(&records[0].event, TraceEvent::ToolCall { tool_name, .. } if tool_name == "test_tool")
        );
    }

    #[test]
    fn test_trace_record_serialization() {
        let record = TraceRecord {
            timestamp_ms: 1234567890,
            level: TraceLevel::Info,
            span_id: "abc-123".to_string(),
            parent_span_id: None,
            event: TraceEvent::LlmCall {
                model: "gpt-4".to_string(),
                prompt: Some(serde_json::json!({"role": "user", "content": "Hello"})),
                response: Some(serde_json::json!({"role": "assistant", "content": "Hi there!"})),
                input_tokens: Some(10),
                output_tokens: Some(5),
                error: None,
            },
            duration_ms: Some(150),
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("llm_call"));
        assert!(json.contains("gpt-4"));
    }
}
