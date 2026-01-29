//! Execution engine for scaffold IR
//!
//! Executes tasks and tools by interpreting the IR.

use crate::error::{InterpreterError, Result};
use crate::foreign::ForeignRegistry;
use scaffold_ir::{
    AgentIR, ExprIR, LiteralIR, PipelineCallIR, PipelineIR, PromptIR, StringOrFileIR, ToolExprIR,
    ToolIR, ToolImplIR, TypeDefIR, TypeIR,
};
use scaffold_runtime::trace::{tracer, TraceEvent};
use scaffold_runtime::{PromptManager, Value};
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

/// Tool executor - executes tool implementations
pub struct ToolExecutor {
    /// Cached tool results (for pure tools)
    #[allow(dead_code)]
    cache: HashMap<String, Value>,
    /// Registered tools for ToolCall lookups
    tools: HashMap<String, ToolIR>,
    /// Registered prompts for prompt calls
    prompts_ir: HashMap<String, PromptIR>,
    /// Registered agents
    agents: HashMap<String, AgentIR>,
    /// Registered pipelines
    pipelines: HashMap<String, PipelineIR>,
    /// Registered type definitions (for resolving Named types)
    types: HashMap<String, TypeDefIR>,
    /// Base path for file() references
    base_path: Option<std::path::PathBuf>,
    /// Foreign function registry
    foreign_registry: ForeignRegistry,
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            tools: HashMap::new(),
            prompts_ir: HashMap::new(),
            agents: HashMap::new(),
            pipelines: HashMap::new(),
            types: HashMap::new(),
            base_path: None,
            foreign_registry: ForeignRegistry::new(),
        }
    }

    /// Set base path for file() references
    pub fn with_base_path(mut self, path: std::path::PathBuf) -> Self {
        self.base_path = Some(path);
        self
    }

    /// Set the foreign function registry
    pub fn with_foreign_registry(mut self, registry: ForeignRegistry) -> Self {
        self.foreign_registry = registry;
        self
    }

    /// Get a mutable reference to the foreign registry for registration
    pub fn foreign_registry_mut(&mut self) -> &mut ForeignRegistry {
        &mut self.foreign_registry
    }

    /// Get a reference to the foreign registry
    pub fn foreign_registry(&self) -> &ForeignRegistry {
        &self.foreign_registry
    }

    /// Register tools for ToolCall lookups
    pub fn register_tools(&mut self, tools: &[ToolIR]) {
        for tool in tools {
            self.tools.insert(tool.name.clone(), tool.clone());
        }
    }

    /// Register prompts
    pub fn register_prompts(&mut self, prompts: &[PromptIR]) {
        for prompt in prompts {
            self.prompts_ir.insert(prompt.name.clone(), prompt.clone());
        }
    }

    /// Register agents
    pub fn register_agents(&mut self, agents: &[AgentIR]) {
        for agent in agents {
            self.agents.insert(agent.name.clone(), agent.clone());
        }
    }

    /// Register pipelines
    pub fn register_pipelines(&mut self, pipelines: &[PipelineIR]) {
        for pipeline in pipelines {
            self.pipelines
                .insert(pipeline.name.clone(), pipeline.clone());
        }
    }

    /// Register type definitions
    pub fn register_types(&mut self, types: &[TypeDefIR]) {
        for type_def in types {
            self.types.insert(type_def.name.clone(), type_def.clone());
        }
    }

    /// Resolve a type, looking up Named types in the types registry
    fn resolve_type<'a>(&'a self, ty: &'a TypeIR) -> &'a TypeIR {
        match ty {
            TypeIR::Named { name } => {
                if let Some(type_def) = self.types.get(name) {
                    &type_def.definition
                } else {
                    ty
                }
            }
            _ => ty,
        }
    }

    /// Get a registered tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolIR> {
        self.tools.get(name)
    }

    /// Get a registered prompt by name
    pub fn get_prompt(&self, name: &str) -> Option<&PromptIR> {
        self.prompts_ir.get(name)
    }

    /// Get a registered agent by name
    pub fn get_agent(&self, name: &str) -> Option<&AgentIR> {
        self.agents.get(name)
    }

    /// Get a registered pipeline by name
    pub fn get_pipeline(&self, name: &str) -> Option<&PipelineIR> {
        self.pipelines.get(name)
    }

    /// Resolve a StringOrFileIR to actual content
    fn resolve_string_or_file(&self, sof: &StringOrFileIR) -> Result<String> {
        match sof {
            StringOrFileIR::Literal { value } => Ok(value.clone()),
            StringOrFileIR::File { path } => {
                let full_path = if let Some(ref base) = self.base_path {
                    base.join(path)
                } else {
                    Path::new(path).to_path_buf()
                };
                std::fs::read_to_string(&full_path).map_err(|e| {
                    InterpreterError::Runtime(format!(
                        "Failed to read file '{}': {}",
                        full_path.display(),
                        e
                    ))
                })
            }
        }
    }

    /// Generate example JSON from TypeIR (for prompts)
    fn generate_example_json(&self, ty: &TypeIR) -> String {
        match ty {
            TypeIR::Bool => "true".to_string(),
            TypeIR::Int => "42".to_string(),
            TypeIR::Float => "3.14".to_string(),
            TypeIR::String => "\"your text here\"".to_string(),
            TypeIR::Any => "null".to_string(),
            TypeIR::Bytes => "\"base64data\"".to_string(),
            TypeIR::List { element } => {
                format!("[{}]", self.generate_example_json(element))
            }
            TypeIR::Map { key: _, value } => {
                format!("{{\"key\": {}}}", self.generate_example_json(value))
            }
            TypeIR::Option { inner } => self.generate_example_json(inner),
            TypeIR::Result { ok, .. } => self.generate_example_json(ok),
            TypeIR::Struct { fields } => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, self.generate_example_json(v)))
                    .collect();
                format!("{{{}}}", field_strs.join(", "))
            }
            TypeIR::Named { name } => {
                if let Some(type_def) = self.types.get(name) {
                    self.generate_example_json(&type_def.definition)
                } else {
                    "{}".to_string()
                }
            }
        }
    }

    /// Generate JSON schema from TypeIR
    fn type_to_json_schema(&self, ty: &TypeIR) -> String {
        match ty {
            TypeIR::Bool => "boolean".to_string(),
            TypeIR::Int => "integer".to_string(),
            TypeIR::Float => "number".to_string(),
            TypeIR::String => "\"string\"".to_string(),
            TypeIR::Any => "\"any\"".to_string(),
            TypeIR::Bytes => "\"string\"".to_string(),
            TypeIR::List { element } => {
                format!("[{}]", self.type_to_json_schema(element))
            }
            TypeIR::Map { key: _, value } => {
                format!("{{ \"key\": {} }}", self.type_to_json_schema(value))
            }
            TypeIR::Option { inner } => self.type_to_json_schema(inner),
            TypeIR::Result { ok, .. } => self.type_to_json_schema(ok),
            TypeIR::Struct { fields } => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, self.type_to_json_schema(v)))
                    .collect();
                format!("{{ {} }}", field_strs.join(", "))
            }
            TypeIR::Named { name } => format!("\"{}\"", name),
        }
    }

    /// Execute a tool with the given input
    pub async fn execute(
        &mut self,
        tool: &ToolIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        let start = std::time::Instant::now();
        let input_json = serde_json::to_value(&input).ok();

        // Check preconditions if spec exists
        if let Some(ref spec) = tool.spec {
            for pre in &spec.preconditions {
                let result = self.eval_expr(pre, &input)?;
                if !result.as_bool().unwrap_or(false) {
                    return Err(InterpreterError::PreconditionFailed(format!("{:?}", pre)));
                }
            }
        }

        // Execute implementation with expected output type
        let result = match &tool.implementation {
            Some(impl_) => {
                self.execute_impl(impl_, &input, prompts, Some(&tool.output))
                    .await?
            }
            None => {
                return Err(InterpreterError::Runtime(format!(
                    "Tool '{}' has no implementation",
                    tool.name
                )));
            }
        };

        // Check postconditions
        if let Some(ref spec) = tool.spec {
            for post in &spec.postconditions {
                // Create context with output available
                let mut ctx = HashMap::new();
                ctx.insert("output".to_string(), result.clone());
                ctx.insert("input".to_string(), input.clone());
                let ctx_value = Value::Map(ctx);

                let check = self.eval_expr(post, &ctx_value)?;
                if !check.as_bool().unwrap_or(false) {
                    return Err(InterpreterError::PostconditionFailed(format!("{:?}", post)));
                }
            }
        }

        // Trace tool execution
        if tracer().is_enabled() {
            tracer().record_completed(
                &format!("tool-{}", tool.name),
                TraceEvent::ToolCall {
                    tool_name: tool.name.clone(),
                    input: input_json,
                    output: serde_json::to_value(&result).ok(),
                    error: None,
                },
                start.elapsed(),
            );
        }

        Ok(result)
    }

    /// Execute a prompt with the given input
    pub async fn execute_prompt(
        &mut self,
        prompt: &PromptIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        // Resolve system prompt if present
        let system = if let Some(ref sys) = prompt.system {
            Some(self.resolve_string_or_file(sys)?)
        } else {
            None
        };

        // Resolve template
        let template = self.resolve_string_or_file(&prompt.template)?;

        // Interpolate template with input
        let rendered = prompts
            .interpolate(&template, &input)
            .map_err(|e| InterpreterError::Runtime(e.to_string()))?;

        // Generate JSON schema from output type
        let schema = self.type_to_json_schema(&prompt.output);

        // Build full prompt with system message if present
        let full_prompt = if let Some(sys) = system {
            format!("{}\n\n{}", sys, rendered)
        } else {
            rendered
        };

        // Query LLM with structured output
        let result = scaffold_runtime::llm::query_structured(&full_prompt, &schema)
            .await
            .map_err(|e| InterpreterError::LlmError(e.to_string()))?;

        Ok(result)
    }

    /// Execute an agent with the given input using native tool calling
    pub async fn execute_agent(
        &mut self,
        agent: &AgentIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        use scaffold_runtime::{chat_with_tools, ChatMessage, ChatToolDefinition};

        // Resolve system prompt
        let system = self.resolve_string_or_file(&agent.system)?;

        // Build system message with output format instructions
        let example_json = self.generate_example_json(&agent.output);
        let system_content = format!(
            "{}\n\n## Output Format\n\nWhen you are done and ready to respond to the user, you MUST output ONLY valid JSON matching this exact structure:\n\n{}\n\nDo NOT include any text before or after the JSON. Do NOT wrap it in markdown code blocks.",
            system, example_json
        );

        // Build tool definitions for native tool calling
        let tool_defs: Vec<ChatToolDefinition> = agent
            .tools
            .iter()
            .filter_map(|tool_name| {
                self.tools.get(tool_name).map(|tool| {
                    let params = self.type_to_json_schema_value(&tool.input);
                    ChatToolDefinition::new(
                        &tool.name,
                        format!("Execute the {} tool", tool.name),
                        params,
                    )
                })
            })
            .collect();

        // Initialize messages with system and user input
        let input_str =
            serde_json::to_string_pretty(&input).unwrap_or_else(|_| format!("{:?}", input));
        let mut messages = vec![
            ChatMessage::system(&system_content),
            ChatMessage::user(format!("Input: {}", input_str)),
        ];

        let max_turns = agent.max_turns.unwrap_or(10);
        let mut turn = 0;

        loop {
            if turn >= max_turns {
                return Err(InterpreterError::Runtime(format!(
                    "Agent exceeded max_turns ({})",
                    max_turns
                )));
            }
            turn += 1;

            // Make chat completion request with tools
            let llm_start = std::time::Instant::now();
            let response = chat_with_tools(&messages, &tool_defs)
                .await
                .map_err(|e| InterpreterError::LlmError(e.to_string()))?;

            // Trace the LLM call with messages
            if tracer().is_enabled() {
                let model_name = scaffold_runtime::config().default_model.clone();
                tracer().record_completed(
                    &format!("llm-{}-turn{}", agent.name, turn),
                    TraceEvent::LlmCall {
                        model: model_name,
                        prompt: serde_json::to_value(&messages).ok(),
                        response: serde_json::to_value(&response.message).ok(),
                        input_tokens: None,
                        output_tokens: None,
                        error: None,
                    },
                    llm_start.elapsed(),
                );
            }

            // Check if response has tool calls
            if response.has_tool_calls() {
                // Add assistant message with tool calls to history
                messages.push(response.message.clone());

                // Execute each tool call
                for tool_call in response.tool_calls().unwrap_or(&[]) {
                    let tool_name = &tool_call.function.name;
                    let args_json = &tool_call.function.arguments;

                    // Parse arguments - fail if LLM provided malformed JSON
                    let args_value: serde_json::Value =
                        serde_json::from_str(args_json).map_err(|e| {
                            InterpreterError::Runtime(format!(
                                "Failed to parse tool '{}' arguments as JSON: {}. Raw arguments: {}",
                                tool_name, e, args_json
                            ))
                        })?;
                    let args = json_to_value(args_value);

                    // Execute tool
                    let result_content = if let Some(tool) = self.tools.get(tool_name).cloned() {
                        match self.execute(&tool, args, prompts).await {
                            Ok(result) => {
                                serde_json::to_string_pretty(&result)
                                    .unwrap_or_else(|_| format!("{:?}", result))
                            }
                            Err(e) => format!("Error: {}", e),
                        }
                    } else {
                        format!("Error: Tool '{}' not found", tool_name)
                    };

                    // Add tool result message
                    messages.push(ChatMessage::tool_result(&tool_call.id, result_content));
                }
            } else if let Some(content) = response.content() {
                // No tool calls - this should be the final response
                // Check if content is empty (LLM can return empty string with tool_calls)
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    return Err(InterpreterError::Runtime(
                        "Agent returned empty response".to_string(),
                    ));
                }
                // Try to parse as JSON matching our output schema
                let json_str = extract_json(trimmed);
                let json_value: serde_json::Value =
                    serde_json::from_str(json_str).map_err(|e| {
                        InterpreterError::Runtime(format!(
                            "Failed to parse agent response as JSON: {}. Response was: {}",
                            e, content
                        ))
                    })?;
                return Ok(json_to_value(json_value));
            } else {
                // Empty response - continue or error
                return Err(InterpreterError::Runtime(
                    "Agent returned empty response".to_string(),
                ));
            }
        }
    }

    /// Convert TypeIR to JSON schema as serde_json::Value (for tool definitions)
    fn type_to_json_schema_value(&self, ty: &TypeIR) -> serde_json::Value {
        match ty {
            TypeIR::Bool => serde_json::json!({"type": "boolean"}),
            TypeIR::Int => serde_json::json!({"type": "integer"}),
            TypeIR::Float => serde_json::json!({"type": "number"}),
            TypeIR::String => serde_json::json!({"type": "string"}),
            TypeIR::Bytes => serde_json::json!({"type": "string", "format": "byte"}),
            TypeIR::Any => serde_json::json!({}),
            TypeIR::List { element } => {
                serde_json::json!({
                    "type": "array",
                    "items": self.type_to_json_schema_value(element)
                })
            }
            TypeIR::Map { key: _, value } => {
                serde_json::json!({
                    "type": "object",
                    "additionalProperties": self.type_to_json_schema_value(value)
                })
            }
            TypeIR::Option { inner } => {
                // Represent Option<T> as a union with null using standard JSON Schema
                let inner_schema = self.type_to_json_schema_value(inner);
                match inner_schema {
                    serde_json::Value::Object(mut obj) => {
                        match obj.get("type").cloned() {
                            Some(serde_json::Value::String(t)) => {
                                // Convert "type": "T" to "type": ["null", "T"]
                                let types = serde_json::Value::Array(vec![
                                    serde_json::Value::String("null".to_string()),
                                    serde_json::Value::String(t),
                                ]);
                                obj.insert("type".to_string(), types);
                                serde_json::Value::Object(obj)
                            }
                            Some(serde_json::Value::Array(mut arr)) => {
                                // Ensure "null" is included in an existing type array
                                let null_val = serde_json::Value::String("null".to_string());
                                if !arr.contains(&null_val) {
                                    arr.insert(0, null_val);
                                    obj.insert("type".to_string(), serde_json::Value::Array(arr));
                                }
                                serde_json::Value::Object(obj)
                            }
                            _ => {
                                // Fallback: wrap the entire schema in an anyOf with null
                                serde_json::json!({
                                    "anyOf": [
                                        { "type": "null" },
                                        serde_json::Value::Object(obj)
                                    ]
                                })
                            }
                        }
                    }
                    other => {
                        // Non-object schemas: wrap in an anyOf with null
                        serde_json::json!({
                            "anyOf": [
                                { "type": "null" },
                                other
                            ]
                        })
                    }
                }
            }
            TypeIR::Result { ok, err: _ } => self.type_to_json_schema_value(ok),
            TypeIR::Struct { fields } => {
                let properties: serde_json::Map<String, serde_json::Value> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.type_to_json_schema_value(v)))
                    .collect();
                // Only include non-Option fields in required array
                let required: Vec<String> = fields
                    .iter()
                    .filter(|(_, v)| !matches!(v, TypeIR::Option { .. }))
                    .map(|(k, _)| k.clone())
                    .collect();
                serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required
                })
            }
            TypeIR::Named { name } => {
                // Try to resolve from type definitions
                if let Some(type_def) = self.types.get(name) {
                    self.type_to_json_schema_value(&type_def.definition)
                } else {
                    serde_json::json!({"type": "object"})
                }
            }
        }
    }

    /// Execute a pipeline with the given input
    pub async fn execute_pipeline(
        &mut self,
        pipeline: &PipelineIR,
        input: Value,
        prompts: &PromptManager,
    ) -> Result<Value> {
        let mut bindings: HashMap<String, Value> = HashMap::new();

        // Add input to bindings
        if let Value::Map(m) = &input {
            bindings.extend(m.clone());
        }
        bindings.insert("input".to_string(), input.clone());

        let mut last_value = Value::Null;

        for step in &pipeline.steps {
            let ctx = Value::Map(bindings.clone());

            let result = match &step.call {
                PipelineCallIR::Prompt { name, args } => {
                    // Get the prompt
                    let prompt = self.prompts_ir.get(name).cloned().ok_or_else(|| {
                        InterpreterError::Runtime(format!("Prompt '{}' not found", name))
                    })?;

                    // Build input from args
                    let mut input_map = HashMap::new();
                    let field_names: Vec<String> = match &prompt.input {
                        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
                        _ => Vec::new(),
                    };

                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, &ctx, prompts, None).await?;
                        let key = field_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }

                    self.execute_prompt(&prompt, Value::Map(input_map), prompts)
                        .await?
                }
                PipelineCallIR::Tool { name, args } => {
                    // Get the tool
                    let tool = self
                        .tools
                        .get(name)
                        .cloned()
                        .ok_or_else(|| InterpreterError::ToolNotFound(name.clone()))?;

                    // Build input from args
                    let mut input_map = HashMap::new();
                    let field_names: Vec<String> = match &tool.input {
                        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
                        _ => Vec::new(),
                    };

                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, &ctx, prompts, None).await?;
                        let key = field_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }

                    self.execute(&tool, Value::Map(input_map), prompts).await?
                }
                PipelineCallIR::Expr { expr } => {
                    // Evaluate expression in current bindings context
                    self.execute_tool_expr(expr, &ctx, prompts, None).await?
                }
            };

            if let Some(ref binding_name) = step.binding {
                bindings.insert(binding_name.clone(), result.clone());
            }
            last_value = result;
        }

        // If pipeline declares a struct output, synthesize from bindings
        // Resolve Named types to get the actual struct fields
        let resolved_output = self.resolve_type(&pipeline.output);
        if let TypeIR::Struct { fields } = resolved_output {
            let mut out_map: HashMap<String, Value> = HashMap::new();
            for (k, _) in fields {
                if let Some(v) = bindings.get(k).cloned() {
                    out_map.insert(k.clone(), v);
                }
            }
            if !out_map.is_empty() {
                return Ok(Value::Map(out_map));
            }
        }
        Ok(last_value)
    }

    /// Execute a tool implementation
    ///
    /// Returns a boxed future to handle recursive async calls
    fn execute_impl<'a>(
        &'a mut self,
        impl_: &'a ToolImplIR,
        input: &'a Value,
        prompts: &'a PromptManager,
        expected: Option<&'a TypeIR>,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match impl_ {
                ToolImplIR::Expr { expr } => {
                    self.execute_tool_expr(expr, input, prompts, expected).await
                }
                ToolImplIR::Sequence { statements } => {
                    let mut last_value = Value::Null;
                    let mut bindings: HashMap<String, Value> = HashMap::new();

                    // Add input to bindings
                    if let Value::Map(m) = input {
                        bindings.extend(m.clone());
                    }
                    bindings.insert("input".to_string(), input.clone());

                    for stmt in statements {
                        let ctx = Value::Map(bindings.clone());
                        let value = self
                            .execute_tool_expr(&stmt.expr, &ctx, prompts, None)
                            .await?;

                        if let Some(ref name) = stmt.binding {
                            bindings.insert(name.clone(), value.clone());
                        }
                        last_value = value;
                    }

                    // Synthesize struct output if expected
                    if let Some(TypeIR::Struct { fields }) = expected {
                        let mut out_map: HashMap<String, Value> = HashMap::new();
                        for (k, _) in fields {
                            if let Some(v) = bindings.get(k).cloned() {
                                out_map.insert(k.clone(), v);
                            }
                        }
                        if !out_map.is_empty() {
                            return Ok(Value::Map(out_map));
                        }
                    }
                    Ok(last_value)
                }
                ToolImplIR::Parallel { statements } => {
                    // For now, execute sequentially (TODO: true parallel with tokio::join!)
                    let mut last_value = Value::Null;
                    let mut bindings: HashMap<String, Value> = HashMap::new();

                    // Add input to bindings
                    if let Value::Map(m) = input {
                        bindings.extend(m.clone());
                    }
                    bindings.insert("input".to_string(), input.clone());

                    for stmt in statements {
                        let ctx = Value::Map(bindings.clone());
                        let value = self
                            .execute_tool_expr(&stmt.expr, &ctx, prompts, None)
                            .await?;

                        if let Some(ref name) = stmt.binding {
                            bindings.insert(name.clone(), value.clone());
                        }
                        last_value = value;
                    }

                    if let Some(TypeIR::Struct { fields }) = expected {
                        let mut out_map: HashMap<String, Value> = HashMap::new();
                        for (k, _) in fields {
                            if let Some(v) = bindings.get(k).cloned() {
                                out_map.insert(k.clone(), v);
                            }
                        }
                        if !out_map.is_empty() {
                            return Ok(Value::Map(out_map));
                        }
                    }
                    Ok(last_value)
                }
            }
        })
    }

    /// Execute a tool expression
    ///
    /// Returns a boxed future to handle recursive async calls
    fn execute_tool_expr<'a>(
        &'a mut self,
        expr: &'a ToolExprIR,
        ctx: &'a Value,
        prompts: &'a PromptManager,
        expected: Option<&'a TypeIR>,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match expr {
                ToolExprIR::Ident { name } => self.get_value(ctx, name),
                ToolExprIR::FieldAccess { base, field } => {
                    let base_val = self.execute_tool_expr(base, ctx, prompts, None).await?;
                    self.get_field(&base_val, field)
                }
                ToolExprIR::Literal { value } => Ok(self.literal_to_value(value)),
                ToolExprIR::Shell { command } => {
                    // Interpolate variables in command
                    let cmd = prompts
                        .interpolate(command, ctx)
                        .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                    if let Some(exp) = expected {
                        match exp {
                            TypeIR::Bytes => {
                                let bytes = scaffold_runtime::shell::execute_bytes(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::Bytes(bytes))
                            }
                            TypeIR::Int => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let i = scaffold_runtime::parse::parse_i64(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Int(i))
                            }
                            TypeIR::Float => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let f = scaffold_runtime::parse::parse_f64(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Float(f))
                            }
                            TypeIR::Bool => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let b = scaffold_runtime::parse::parse_bool(s.trim())
                                    .map_err(|e| InterpreterError::Runtime(e.to_string()))?;
                                Ok(Value::Bool(b))
                            }
                            TypeIR::String => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::String(s.trim().to_string()))
                            }
                            TypeIR::Struct { fields } => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                let st = s.trim();
                                if fields.len() == 1 {
                                    let (fname, fty) = fields.iter().next().unwrap();
                                    let inner = match fty {
                                        TypeIR::Int => Value::Int(
                                            scaffold_runtime::parse::parse_i64(st).map_err(
                                                |e| InterpreterError::Runtime(e.to_string()),
                                            )?,
                                        ),
                                        TypeIR::Float => Value::Float(
                                            scaffold_runtime::parse::parse_f64(st).map_err(
                                                |e| InterpreterError::Runtime(e.to_string()),
                                            )?,
                                        ),
                                        TypeIR::Bool => Value::Bool(
                                            scaffold_runtime::parse::parse_bool(st).map_err(
                                                |e| InterpreterError::Runtime(e.to_string()),
                                            )?,
                                        ),
                                        TypeIR::String => Value::String(st.to_string()),
                                        _ => match serde_json::from_str::<serde_json::Value>(st) {
                                            Ok(j) => json_to_value(j),
                                            Err(_) => Value::String(st.to_string()),
                                        },
                                    };
                                    let mut m = HashMap::new();
                                    m.insert(fname.clone(), inner);
                                    Ok(Value::Map(m))
                                } else {
                                    match serde_json::from_str::<serde_json::Value>(st) {
                                        Ok(j) => Ok(json_to_value(j)),
                                        Err(_) => Ok(Value::String(s)),
                                    }
                                }
                            }
                            _ => {
                                let s = scaffold_runtime::shell::execute(&cmd)
                                    .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                                Ok(Value::String(s))
                            }
                        }
                    } else {
                        let output = scaffold_runtime::shell::execute(&cmd)
                            .map_err(|e| InterpreterError::ShellError(e.to_string()))?;
                        Ok(Value::String(output))
                    }
                }
                ToolExprIR::ForeignCall {
                    module,
                    function,
                    args,
                } => {
                    // Evaluate arguments
                    let mut arg_values = Vec::new();
                    for arg in args {
                        let val = self.execute_tool_expr(arg, ctx, prompts, None).await?;
                        arg_values.push(val);
                    }

                    // Call the foreign function through the registry
                    self.foreign_registry.call(module, function, arg_values)
                }
                ToolExprIR::ToolCall { tool, args } => {
                    // Look up the tool first to get input field names
                    let tool_ir = self
                        .get_tool(tool)
                        .cloned()
                        .ok_or_else(|| InterpreterError::ToolNotFound(tool.clone()))?;

                    // Get field names from tool's input type
                    let field_names: Vec<String> = match &tool_ir.input {
                        scaffold_ir::TypeIR::Struct { fields } => fields.keys().cloned().collect(),
                        _ => Vec::new(),
                    };

                    // Build input from args, using field names if available
                    let mut input_map = HashMap::new();
                    for (i, arg) in args.iter().enumerate() {
                        let val = self.execute_tool_expr(arg, ctx, prompts, None).await?;
                        let key = field_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("arg{}", i));
                        input_map.insert(key, val);
                    }
                    let input = Value::Map(input_map);

                    // Execute the tool
                    self.execute(&tool_ir, input, prompts).await
                }
                ToolExprIR::Pipe { left, right } => {
                    let left_val = self.execute_tool_expr(left, ctx, prompts, None).await?;
                    // Use left result as input to right
                    self.execute_tool_expr(right, &left_val, prompts, expected)
                        .await
                }
                ToolExprIR::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    let cond = self.eval_expr(condition, ctx)?;
                    if cond.as_bool().unwrap_or(false) {
                        self.execute_impl(then_branch, ctx, prompts, expected).await
                    } else if let Some(else_) = else_branch {
                        self.execute_impl(else_, ctx, prompts, expected).await
                    } else {
                        Ok(Value::Null)
                    }
                }
                ToolExprIR::Match { scrutinee, arms } => {
                    let scrutinee_val = self
                        .execute_tool_expr(scrutinee, ctx, prompts, None)
                        .await?;

                    for arm in arms {
                        // Simple pattern matching - check equality
                        let pattern_val = self.eval_expr(&arm.pattern, ctx)?;
                        if scrutinee_val == pattern_val {
                            return self.execute_impl(&arm.body, ctx, prompts, expected).await;
                        }
                    }

                    // No match - return null
                    Ok(Value::Null)
                }
                ToolExprIR::For {
                    variable,
                    iterable,
                    body,
                } => {
                    let iterable_val = self.execute_tool_expr(iterable, ctx, prompts, None).await?;

                    // Get list to iterate over
                    let items = match &iterable_val {
                        Value::List(items) => items.clone(),
                        _ => {
                            return Err(InterpreterError::TypeMismatch {
                                expected: "list".to_string(),
                                actual: iterable_val.type_name().to_string(),
                            })
                        }
                    };

                    let mut last_result = Value::Null;

                    // Create a mutable context with the loop variable
                    for item in items {
                        let loop_ctx = match ctx {
                            Value::Map(m) => {
                                let mut new_map = m.clone();
                                new_map.insert(variable.clone(), item);
                                Value::Map(new_map)
                            }
                            _ => {
                                let mut new_map = HashMap::new();
                                new_map.insert(variable.clone(), item);
                                Value::Map(new_map)
                            }
                        };

                        match self.execute_impl(body, &loop_ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::While { condition, body } => {
                    let mut last_result = Value::Null;

                    loop {
                        let cond = self.eval_expr(condition, ctx)?;
                        if !cond.as_bool().unwrap_or(false) {
                            break;
                        }

                        match self.execute_impl(body, ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::Loop { body } => {
                    let mut last_result = Value::Null;

                    loop {
                        match self.execute_impl(body, ctx, prompts, expected).await {
                            Ok(val) => last_result = val,
                            Err(InterpreterError::Break) => break,
                            Err(InterpreterError::Continue) => continue,
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(last_result)
                }
                ToolExprIR::Break => Err(InterpreterError::Break),
                ToolExprIR::Continue => Err(InterpreterError::Continue),
                ToolExprIR::Expr { expr } => {
                    // Evaluate a general expression (arithmetic, comparisons, etc.)
                    self.eval_expr(expr, &ctx)
                }
            }
        })
    }

    /// Evaluate a simple expression
    fn eval_expr(&self, expr: &ExprIR, ctx: &Value) -> Result<Value> {
        match expr {
            ExprIR::Literal { value } => Ok(self.literal_to_value(value)),
            ExprIR::Ident { name } => self.get_value(ctx, name),
            ExprIR::FieldAccess { base, field } => {
                let base_val = self.eval_expr(base, ctx)?;
                self.get_field(&base_val, field)
            }
            ExprIR::Binary { left, op, right } => {
                let l = self.eval_expr(left, ctx)?;
                let r = self.eval_expr(right, ctx)?;
                self.eval_binary_op(&l, op, &r)
            }
            ExprIR::Call { function, args } => {
                // Built-in functions
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a, ctx))
                    .collect::<Result<Vec<_>>>()?;

                self.eval_builtin(function, &arg_vals)
            }
            ExprIR::ForeignCall {
                module,
                function,
                args,
            } => {
                // Evaluate arguments and call foreign function
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a, ctx))
                    .collect::<Result<Vec<_>>>()?;

                self.foreign_registry.call(module, function, arg_vals)
            }
        }
    }

    /// Convert literal to value
    fn literal_to_value(&self, lit: &LiteralIR) -> Value {
        match lit {
            LiteralIR::Int { value } => Value::Int(*value),
            LiteralIR::Float { value } => Value::Float(*value),
            LiteralIR::String { value } => Value::String(value.clone()),
            LiteralIR::Bool { value } => Value::Bool(*value),
            LiteralIR::Null => Value::Null,
        }
    }

    /// Get a value from context
    fn get_value(&self, ctx: &Value, name: &str) -> Result<Value> {
        // Special identifier: 'input' refers to the entire current context
        if name == "input" {
            return Ok(ctx.clone());
        }
        match ctx {
            Value::Map(m) => m
                .get(name)
                .cloned()
                .ok_or_else(|| InterpreterError::VariableNotFound(name.to_string())),
            Value::Struct { fields, .. } => fields
                .get(name)
                .cloned()
                .ok_or_else(|| InterpreterError::VariableNotFound(name.to_string())),
            _ => Err(InterpreterError::VariableNotFound(name.to_string())),
        }
    }

    /// Get a field from a value
    fn get_field(&self, val: &Value, field: &str) -> Result<Value> {
        match val {
            Value::Map(m) => m
                .get(field)
                .cloned()
                .ok_or_else(|| InterpreterError::FieldNotFound {
                    type_name: "map".to_string(),
                    field: field.to_string(),
                }),
            Value::Struct { type_name, fields } => {
                fields
                    .get(field)
                    .cloned()
                    .ok_or_else(|| InterpreterError::FieldNotFound {
                        type_name: type_name.clone(),
                        field: field.to_string(),
                    })
            }
            _ => Err(InterpreterError::FieldNotFound {
                type_name: val.type_name().to_string(),
                field: field.to_string(),
            }),
        }
    }

    /// Evaluate binary operation
    fn eval_binary_op(&self, left: &Value, op: &str, right: &Value) -> Result<Value> {
        match op {
            "==" => Ok(Value::Bool(left == right)),
            "!=" => Ok(Value::Bool(left != right)),
            "<" => match (left.as_int(), right.as_int()) {
                (Some(l), Some(r)) => Ok(Value::Bool(l < r)),
                _ => match (left.as_float(), right.as_float()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l < r)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                },
            },
            "<=" => match (left.as_int(), right.as_int()) {
                (Some(l), Some(r)) => Ok(Value::Bool(l <= r)),
                _ => match (left.as_float(), right.as_float()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l <= r)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                },
            },
            ">" => match (left.as_int(), right.as_int()) {
                (Some(l), Some(r)) => Ok(Value::Bool(l > r)),
                _ => match (left.as_float(), right.as_float()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l > r)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                },
            },
            ">=" => match (left.as_int(), right.as_int()) {
                (Some(l), Some(r)) => Ok(Value::Bool(l >= r)),
                _ => match (left.as_float(), right.as_float()) {
                    (Some(l), Some(r)) => Ok(Value::Bool(l >= r)),
                    _ => Err(InterpreterError::TypeMismatch {
                        expected: "number".to_string(),
                        actual: format!("{}, {}", left.type_name(), right.type_name()),
                    }),
                },
            },
            "+" => match (left, right) {
                (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l + r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
                (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l + *r as f64)),
                (Value::String(l), Value::String(r)) => Ok(Value::String(format!("{}{}", l, r))),
                _ => Err(InterpreterError::TypeMismatch {
                    expected: "number or string".to_string(),
                    actual: format!("{}, {}", left.type_name(), right.type_name()),
                }),
            },
            "-" => match (left, right) {
                (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l - r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
                (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l - *r as f64)),
                _ => Err(InterpreterError::TypeMismatch {
                    expected: "number".to_string(),
                    actual: format!("{}, {}", left.type_name(), right.type_name()),
                }),
            },
            "*" => match (left, right) {
                (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l * r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
                (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l * *r as f64)),
                _ => Err(InterpreterError::TypeMismatch {
                    expected: "number".to_string(),
                    actual: format!("{}, {}", left.type_name(), right.type_name()),
                }),
            },
            "/" => match (left, right) {
                (Value::Int(l), Value::Int(r)) if *r != 0 => Ok(Value::Int(l / r)),
                (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l / r)),
                (Value::Int(l), Value::Float(r)) => Ok(Value::Float(*l as f64 / r)),
                (Value::Float(l), Value::Int(r)) => Ok(Value::Float(l / *r as f64)),
                _ => Err(InterpreterError::Runtime("Division error".to_string())),
            },
            "&&" | "and" => {
                let l = left.as_bool().unwrap_or(false);
                let r = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(l && r))
            }
            "||" | "or" => {
                let l = left.as_bool().unwrap_or(false);
                let r = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(l || r))
            }
            _ => Err(InterpreterError::Runtime(format!(
                "Unknown operator: {}",
                op
            ))),
        }
    }

    /// Evaluate built-in function
    fn eval_builtin(&self, name: &str, args: &[Value]) -> Result<Value> {
        match name {
            "len" | "length" => {
                if let Some(v) = args.first() {
                    match v {
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::List(l) => Ok(Value::Int(l.len() as i64)),
                        Value::Bytes(b) => Ok(Value::Int(b.len() as i64)),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            "is_some" => Ok(Value::Bool(!matches!(
                args.first(),
                Some(Value::Null) | None
            ))),
            "is_none" => Ok(Value::Bool(matches!(
                args.first(),
                Some(Value::Null) | None
            ))),
            "not" => {
                let b = args.first().and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(Value::Bool(!b))
            }
            "abs" => {
                if let Some(v) = args.first() {
                    match v {
                        Value::Int(i) => Ok(Value::Int(i.abs())),
                        Value::Float(f) => Ok(Value::Float(f.abs())),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            _ => Err(InterpreterError::Runtime(format!(
                "Unknown function: {}",
                name
            ))),
        }
    }
}

/// Extract JSON from a string that might have markdown code blocks
fn extract_json(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("```json") {
        let start = s.find('\n').map(|i| i + 1).unwrap_or(7);
        let end = s.rfind("```").unwrap_or(s.len());
        return s[start..end].trim();
    }
    if s.starts_with("```") {
        let start = s.find('\n').map(|i| i + 1).unwrap_or(3);
        let end = s.rfind("```").unwrap_or(s.len());
        return s[start..end].trim();
    }
    s
}

/// Convert serde_json::Value to scaffold_runtime::Value
fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => Value::List(arr.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, Value> = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect();
            Value::Map(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_literal() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Literal {
            value: LiteralIR::Int { value: 42 },
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_eval_binary() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 10 },
            }),
            op: "+".to_string(),
            right: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 5 },
            }),
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_eval_comparison() {
        let executor = ToolExecutor::new();
        let ctx = Value::Map(HashMap::new());

        let expr = ExprIR::Binary {
            left: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 10 },
            }),
            op: ">".to_string(),
            right: Box::new(ExprIR::Literal {
                value: LiteralIR::Int { value: 5 },
            }),
        };
        let result = executor.eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_get_field() {
        let executor = ToolExecutor::new();

        let mut fields = HashMap::new();
        fields.insert("x".to_string(), Value::Int(10));
        fields.insert("y".to_string(), Value::Int(20));
        let ctx = Value::Struct {
            type_name: "Point".to_string(),
            fields,
        };

        let expr = ExprIR::FieldAccess {
            base: Box::new(ExprIR::Ident {
                name: "input".to_string(),
            }),
            field: "x".to_string(),
        };

        // Wrap in map for lookup
        let mut wrapper = HashMap::new();
        wrapper.insert("input".to_string(), ctx);
        let result = executor.eval_expr(&expr, &Value::Map(wrapper)).unwrap();
        assert_eq!(result, Value::Int(10));
    }
}
