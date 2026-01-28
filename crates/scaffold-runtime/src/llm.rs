//! LLM (Language Model) integration via rig.rs
//!
//! This module provides a unified interface for calling LLMs from scaffold.
//!
//! Supported providers:
//! - OpenAI (gpt-4, gpt-4o, gpt-4o-mini, o1, o3, etc.)
//! - Anthropic (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
//!
//! # Configuration
//!
//! API keys can be set via:
//! - Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY
//! - Config file: ~/.scaffold/config.toml
//!
//! # Usage
//!
//! ```ignore
//! use scaffold_runtime::llm::{query, query_with_model, AgentBuilder};
//!
//! // Simple query with default model
//! let response = query("What is 2+2?").await?;
//!
//! // Query with specific model
//! let response = query_with_model("claude-3-sonnet", "Explain quantum computing").await?;
//!
//! // Build a custom agent
//! let agent = AgentBuilder::new("gpt-4o")
//!     .system_prompt("You are a helpful coding assistant.")
//!     .temperature(0.7)
//!     .build();
//! let response = agent.prompt("Write a Python hello world").await?;
//! ```

use crate::config::{config, parse_model_id};
use crate::error::{Error, Result};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{AssistantContent, CompletionModel};
use rig::providers::{anthropic, openai};
use serde::{Deserialize, Serialize};

// ============================================================================
// OpenAI-compatible Chat Types (for native tool calling)
// ============================================================================

/// A chat message in OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }
}

/// Chat message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A tool call requested by the assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// The function being called
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool definition for the API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

impl ChatToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Response from chat completion with tools
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

impl ChatResponse {
    /// Check if the response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }

    /// Get the tool calls if any
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.message.tool_calls.as_deref()
    }

    /// Get the content if any
    pub fn content(&self) -> Option<&str> {
        self.message.content.as_deref()
    }
}

/// Configuration for LLM calls
#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    /// Model to use (e.g., "gpt-4", "claude-3-sonnet")
    pub model: Option<String>,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// System prompt to prepend
    pub system_prompt: Option<String>,
}

impl LlmConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// Query an LLM with a prompt using the default model
pub async fn query(prompt: &str) -> Result<String> {
    let model = &config().default_model;
    query_with_model(model, prompt).await
}

/// Query an LLM with a specific model
pub async fn query_with_model(model: &str, prompt: &str) -> Result<String> {
    let config = LlmConfig::new().with_model(model);
    query_with_config(prompt, &config).await
}

/// Query an LLM with full configuration
pub async fn query_with_config(prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    // Check for mock mode (for testing)
    if std::env::var("SCAFFOLD_LLM_MOCK").is_ok() {
        return Ok(format!("Mock LLM response for: {}", prompt));
    }

    let model = llm_config
        .model
        .as_deref()
        .unwrap_or(&config().default_model);
    let (provider, model_name) = parse_model_id(model);

    match provider {
        "openai" => query_openai(model_name, prompt, llm_config).await,
        "anthropic" => query_anthropic(model_name, prompt, llm_config).await,
        other => Err(Error::ConfigError(format!(
            "Unknown LLM provider: {}. Supported: openai, anthropic",
            other
        ))),
    }
}

/// Extract text from AssistantContent
fn extract_text(content: AssistantContent) -> String {
    match content {
        AssistantContent::Text(text) => text.text,
        _ => String::new(),
    }
}

/// Query OpenAI models
async fn query_openai(model: &str, prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let cfg = config();
    let api_key = cfg.get_api_key("openai").ok_or_else(|| {
        Error::ConfigError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or add to ~/.scaffold/config.toml"
                .to_string(),
        )
    })?;

    // Set env vars for rig to pick up (supports custom base_url for OpenRouter, etc.)
    std::env::set_var("OPENAI_API_KEY", api_key);
    if let Some(base_url) = cfg.get_base_url("openai") {
        std::env::set_var("OPENAI_BASE_URL", base_url);
    }

    let client: openai::Client = openai::Client::from_env();
    let completion_model = client.completion_model(model);

    // Build the completion request
    let mut request = completion_model.completion_request(prompt);

    if let Some(ref system) = llm_config.system_prompt {
        request = request.preamble(system.clone());
    }

    if let Some(temp) = llm_config.temperature {
        request = request.temperature(temp as f64);
    }

    if let Some(max_tokens) = llm_config.max_tokens {
        request = request.max_tokens(max_tokens as u64);
    }

    let response = request
        .send()
        .await
        .map_err(|e| Error::Runtime(format!("OpenAI API error: {}", e)))?;

    // Extract text from the first choice
    Ok(extract_text(response.choice.first()))
}

/// Query Anthropic models
async fn query_anthropic(model: &str, prompt: &str, llm_config: &LlmConfig) -> Result<String> {
    let cfg = config();
    let api_key = cfg.get_api_key("anthropic").ok_or_else(|| {
        Error::ConfigError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or add to ~/.scaffold/config.toml"
                .to_string(),
        )
    })?;

    // Set env vars for rig to pick up
    std::env::set_var("ANTHROPIC_API_KEY", api_key);
    if let Some(base_url) = cfg.get_base_url("anthropic") {
        std::env::set_var("ANTHROPIC_BASE_URL", base_url);
    }

    let client: anthropic::Client = anthropic::Client::from_env();
    let completion_model = client.completion_model(model);

    // Build the completion request
    let mut request = completion_model.completion_request(prompt);

    if let Some(ref system) = llm_config.system_prompt {
        request = request.preamble(system.clone());
    }

    if let Some(temp) = llm_config.temperature {
        request = request.temperature(temp as f64);
    }

    if let Some(max_tokens) = llm_config.max_tokens {
        request = request.max_tokens(max_tokens as u64);
    }

    let response = request
        .send()
        .await
        .map_err(|e| Error::Runtime(format!("Anthropic API error: {}", e)))?;

    // Extract text from the first choice
    Ok(extract_text(response.choice.first()))
}

/// Query and parse response as JSON (typed)
pub async fn query_json<T: serde::de::DeserializeOwned>(prompt: &str) -> Result<T> {
    let response = query(prompt).await?;
    serde_json::from_str(&response)
        .map_err(|e| Error::Runtime(format!("Failed to parse LLM response as JSON: {}", e)))
}

/// Query with structured JSON output
///
/// Wraps the prompt with instructions to return JSON matching the given schema,
/// and parses the response into a Value.
///
/// # Example schema
/// ```text
/// { "answer": "string", "confidence": "high|medium|low", "reasoning": "string" }
/// ```
pub async fn query_structured(prompt: &str, schema: &str) -> Result<crate::Value> {
    let structured_prompt = format!(
        "{}\n\nRespond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}",
        prompt, schema
    );

    let response = query(&structured_prompt).await?;

    // Try to extract JSON if wrapped in markdown code blocks
    let json_str = extract_json(&response);

    // Parse into serde_json::Value first
    let json_value: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        Error::Runtime(format!(
            "Failed to parse LLM response as JSON: {}. Response was: {}",
            e, response
        ))
    })?;

    // Convert to our Value type
    Ok(json_to_value(json_value))
}

/// Extract JSON from a response that might be wrapped in markdown code blocks
fn extract_json(response: &str) -> &str {
    let trimmed = response.trim();

    // Check for ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // Check for ``` ... ``` blocks
    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        // Skip optional language identifier on same line
        let newline_pos = trimmed[content_start..].find('\n').unwrap_or(0);
        let actual_start = content_start + newline_pos;
        if let Some(end) = trimmed[actual_start..].find("```") {
            return trimmed[actual_start..actual_start + end].trim();
        }
    }

    // Return as-is if no code blocks found
    trimmed
}

/// Convert serde_json::Value to our Value type
fn json_to_value(v: serde_json::Value) -> crate::Value {
    match v {
        serde_json::Value::Null => crate::Value::Null,
        serde_json::Value::Bool(b) => crate::Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                crate::Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                crate::Value::Float(f)
            } else {
                crate::Value::Null
            }
        }
        serde_json::Value::String(s) => crate::Value::String(s),
        serde_json::Value::Array(arr) => {
            crate::Value::List(arr.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(obj) => crate::Value::Map(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect(),
        ),
    }
}

/// Trait for custom LLM implementations
pub trait LlmBackend: Send + Sync {
    /// Query the LLM with a prompt
    fn query(&self, prompt: &str) -> impl std::future::Future<Output = Result<String>> + Send;

    /// Query with configuration
    fn query_with_config(
        &self,
        prompt: &str,
        config: &LlmConfig,
    ) -> impl std::future::Future<Output = Result<String>> + Send {
        async move {
            let _ = config;
            self.query(prompt).await
        }
    }
}

/// Agent builder for creating configured LLM agents
#[derive(Debug, Clone)]
pub struct AgentBuilder {
    model: String,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl AgentBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Build the agent configuration
    pub fn build_config(&self) -> LlmConfig {
        LlmConfig {
            model: Some(self.model.clone()),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt.clone(),
        }
    }

    /// Create an agent that can be used for multiple queries
    pub fn build(self) -> Agent {
        Agent {
            config: self.build_config(),
        }
    }
}

/// A configured agent for making LLM queries
#[derive(Debug, Clone)]
pub struct Agent {
    config: LlmConfig,
}

impl Agent {
    /// Create a new agent with the specified model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: LlmConfig::new().with_model(model),
        }
    }

    /// Query the agent with a prompt
    pub async fn prompt(&self, prompt: &str) -> Result<String> {
        query_with_config(prompt, &self.config).await
    }

    /// Query and parse as JSON
    pub async fn prompt_json<T: serde::de::DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let response = self.prompt(prompt).await?;
        serde_json::from_str(&response)
            .map_err(|e| Error::Runtime(format!("Failed to parse response as JSON: {}", e)))
    }

    /// Get the model being used
    pub fn model(&self) -> Option<&str> {
        self.config.model.as_deref()
    }

    /// Get the system prompt
    pub fn system_prompt(&self) -> Option<&str> {
        self.config.system_prompt.as_deref()
    }
}

// ============================================================================
// Chat Completions with Native Tool Calling
// ============================================================================

/// Make a chat completion request with tool calling support
///
/// This function calls the OpenAI-compatible chat completions API directly,
/// supporting native tool calling without text markers.
pub async fn chat_with_tools(
    messages: &[ChatMessage],
    tools: &[ChatToolDefinition],
) -> Result<ChatResponse> {
    let model = &config().default_model;
    chat_with_tools_and_model(model, messages, tools).await
}

/// Make a chat completion request with a specific model
pub async fn chat_with_tools_and_model(
    model: &str,
    messages: &[ChatMessage],
    tools: &[ChatToolDefinition],
) -> Result<ChatResponse> {
    // Check for mock mode
    if std::env::var("SCAFFOLD_LLM_MOCK").is_ok() {
        return Ok(ChatResponse {
            message: ChatMessage::assistant("Mock response"),
            finish_reason: Some("stop".to_string()),
        });
    }

    let cfg = config();
    let (provider, model_name) = parse_model_id(model);

    // If a custom OpenAI base URL is set, assume OpenAI-compatible API
    // This allows using proxies (LiteLLM, OpenRouter, etc.) for any model
    let has_custom_base_url = cfg
        .get_base_url("openai")
        .map(|url| url != "https://api.openai.com/v1")
        .unwrap_or(false);

    // When using a proxy (custom base URL), pass the original model string
    // When using direct OpenAI, pass just the model name
    let effective_model = if has_custom_base_url { model } else { model_name };

    match provider {
        "openai" => chat_openai(effective_model, messages, tools, cfg).await,
        "anthropic" if has_custom_base_url => {
            // Custom base URL set - assume OpenAI-compatible proxy
            chat_openai(model, messages, tools, cfg).await
        }
        "anthropic" => {
            Err(Error::ConfigError(
                "Native tool calling not yet supported for Anthropic. Use OpenAI-compatible proxy with custom base_url, or use 'openai/model-name' prefix."
                    .to_string(),
            ))
        }
        _ if has_custom_base_url => {
            // Unknown provider but custom base URL - try OpenAI-compatible
            chat_openai(model, messages, tools, cfg).await
        }
        other => Err(Error::ConfigError(format!(
            "Unknown LLM provider: {}. Supported: openai (or use custom base_url for compatible APIs)",
            other
        ))),
    }
}

/// OpenAI chat completions with tools
async fn chat_openai(
    model: &str,
    messages: &[ChatMessage],
    tools: &[ChatToolDefinition],
    cfg: &crate::config::Config,
) -> Result<ChatResponse> {
    let api_key = cfg.get_api_key("openai").ok_or_else(|| {
        Error::ConfigError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or add to ~/.scaffold/config.toml"
                .to_string(),
        )
    })?;

    let base_url = cfg
        .get_base_url("openai")
        .unwrap_or("https://api.openai.com/v1");

    // Build request body
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
    });

    // Add tools if any
    if !tools.is_empty() {
        body["tools"] = serde_json::to_value(tools)
            .map_err(|e| Error::Runtime(format!("Failed to serialize tools: {}", e)))?;
    }

    // Make HTTP request with timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| Error::Runtime(format!("Failed to create HTTP client: {}", e)))?;
    let response = client
        .post(format!("{}/chat/completions", base_url))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| Error::Runtime(format!("HTTP request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(Error::Runtime(format!(
            "OpenAI API error: {} - {}",
            status, text
        )));
    }

    // Parse response
    let response_json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| Error::Runtime(format!("Failed to parse response: {}", e)))?;

    // Extract the first choice
    let choice = response_json["choices"]
        .get(0)
        .ok_or_else(|| Error::Runtime("No choices in response".to_string()))?;

    let finish_reason = choice["finish_reason"].as_str().map(String::from);

    // Parse the message
    let msg = &choice["message"];
    let role = match msg["role"].as_str() {
        Some("assistant") => ChatRole::Assistant,
        Some("system") => ChatRole::System,
        Some("user") => ChatRole::User,
        Some("tool") => ChatRole::Tool,
        _ => ChatRole::Assistant,
    };

    let content = msg["content"].as_str().map(String::from);

    // Parse tool calls if present
    let tool_calls = if let Some(tc_array) = msg["tool_calls"].as_array() {
        let calls: Vec<ToolCall> = tc_array
            .iter()
            .filter_map(|tc| {
                Some(ToolCall {
                    id: tc["id"].as_str()?.to_string(),
                    call_type: tc["type"].as_str().unwrap_or("function").to_string(),
                    function: FunctionCall {
                        name: tc["function"]["name"].as_str()?.to_string(),
                        arguments: tc["function"]["arguments"].as_str()?.to_string(),
                    },
                })
            })
            .collect();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    } else {
        None
    };

    Ok(ChatResponse {
        message: ChatMessage {
            role,
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_query() {
        std::env::set_var("SCAFFOLD_LLM_MOCK", "1");
        let result = query("test prompt").await.unwrap();
        assert!(result.contains("Mock LLM response"));
        std::env::remove_var("SCAFFOLD_LLM_MOCK");
    }

    #[test]
    fn test_agent_builder() {
        let agent = AgentBuilder::new("gpt-4o")
            .system_prompt("You are helpful.")
            .temperature(0.7)
            .max_tokens(1000)
            .build();

        assert_eq!(agent.model(), Some("gpt-4o"));
        assert_eq!(agent.system_prompt(), Some("You are helpful."));
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::new()
            .with_model("claude-3-sonnet")
            .with_temperature(0.5)
            .with_max_tokens(2000);

        assert_eq!(config.model, Some("claude-3-sonnet".to_string()));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(2000));
    }
}
