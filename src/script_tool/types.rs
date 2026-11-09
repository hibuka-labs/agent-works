use std::collections::BTreeMap;
use std::path::PathBuf;

use agent_base::ToolExposure;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tool_provider::ToolIdentity;

use super::DEFAULT_TIMEOUT_MS;
use super::law::{ScriptLawError, check_run_law};

/// Parameter-table entry (the author-facing side of plugin manifest / tools.json): scalars only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    String,
    Number,
    Boolean,
}

/// A single parameter: `{ "type": "string", "description": "…" }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamSpec {
    #[serde(rename = "type")]
    pub ty: ParamType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Tool-table budget tier (`ToolExposure` has no serde; the author-facing side uses this string form).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureSpec {
    #[default]
    Direct,
    Deferred,
    Hidden,
}

impl From<ExposureSpec> for ToolExposure {
    fn from(value: ExposureSpec) -> Self {
        match value {
            ExposureSpec::Direct => ToolExposure::Direct,
            ExposureSpec::Deferred => ToolExposure::Deferred,
            ExposureSpec::Hidden => ToolExposure::Hidden,
        }
    }
}

/// A script tool entry (one row of tools.json / one row of the plugin manifest `script` form).
///
/// JSON form (the design's "global script tool entry"):
/// ```json
/// {
///   "name": "deploy",
///   "description": "部署当前项目到 staging",
///   "parameters": { "env": { "type": "string", "description": "目标环境" } },
///   "run": ["/Users/me/proj/deploy.sh", "--env", "${env}"],
///   "timeout_secs": 120
/// }
/// ```
///
/// serde parsing enforces the path law and the name law (fail-closed: failing to parse a legal entry = parse failure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ScriptEntryRaw", into = "ScriptEntryRaw")]
pub struct ScriptToolEntry {
    /// Name (`script.{name}` / the plugin's tool segment). `[a-z0-9_-]{1,32}`, not a reserved word.
    pub name: String,
    /// Model-visible description.
    pub description: String,
    /// Parameter table (scalars only).
    pub parameters: BTreeMap<String, ParamSpec>,
    /// argv template: `${param}` substitution per element, no shell. The path law has already been enforced.
    pub run: Vec<String>,
    /// Hard timeout (milliseconds). `None` = [`DEFAULT_TIMEOUT_MS`].
    pub timeout_ms: Option<u64>,
    /// Environment variable overrides (layered on top of the inherited process env).
    pub env: BTreeMap<String, String>,
    /// Working directory (the current project root; pinned at product-layer registration, the framework does not sniff paths).
    /// Not stored in tools.json (machine-specific path, injected at product-layer load time).
    pub cwd: Option<PathBuf>,
    /// Tool-table budget tier, defaults to direct.
    pub exposure: ExposureSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScriptEntryRaw {
    name: String,
    description: String,
    #[serde(default)]
    parameters: BTreeMap<String, ParamSpec>,
    run: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "is_default_exposure")]
    exposure: ExposureSpec,
}

fn is_default_exposure(e: &ExposureSpec) -> bool {
    *e == ExposureSpec::default()
}

impl TryFrom<ScriptEntryRaw> for ScriptToolEntry {
    type Error = ScriptLawError;

    fn try_from(raw: ScriptEntryRaw) -> Result<Self, Self::Error> {
        let entry = Self {
            name: raw.name,
            description: raw.description,
            parameters: raw.parameters,
            run: raw.run,
            timeout_ms: raw.timeout_secs.map(|s| s.saturating_mul(1_000)),
            env: raw.env,
            cwd: None,
            exposure: raw.exposure,
        };
        entry.check_laws()?;
        Ok(entry)
    }
}

impl From<ScriptToolEntry> for ScriptEntryRaw {
    fn from(entry: ScriptToolEntry) -> Self {
        Self {
            name: entry.name,
            description: entry.description,
            parameters: entry.parameters,
            run: entry.run,
            timeout_secs: entry.timeout_ms.map(|ms| ms.div_ceil(1_000)),
            env: entry.env,
            exposure: entry.exposure,
        }
    }
}

impl ScriptToolEntry {
    /// Construct + enforce the laws (name law + path law). Manual construction also goes through here; never assemble the struct bare.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        run: Vec<String>,
    ) -> Result<Self, ScriptLawError> {
        let entry = Self {
            name: name.into(),
            description: description.into(),
            parameters: BTreeMap::new(),
            run,
            timeout_ms: None,
            env: BTreeMap::new(),
            cwd: None,
            exposure: ExposureSpec::default(),
        };
        entry.check_laws()?;
        Ok(entry)
    }

    /// Name law + path law (called at definition/install time; at execution time only check whether the script file is still there).
    pub fn check_laws(&self) -> Result<(), ScriptLawError> {
        ToolIdentity::script(&self.name).map_err(|_| ScriptLawError::BadName {
            name: self.name.clone(),
        })?;
        check_run_law(&self.run)
    }

    /// Hard timeout (milliseconds): the entry's value, defaulting to [`DEFAULT_TIMEOUT_MS`].
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS)
    }

    /// LLM parameter envelope (MCP `inputSchema` form: object + properties + required=all).
    pub fn parameters_envelope(&self) -> Value {
        parameters_envelope(&self.parameters)
    }

    /// Identity key (global-script form `script.{name}`). The plugin form is assembled by the plugin provider itself.
    pub fn identity(&self) -> ToolIdentity {
        // check_laws already guarantees validity; construction failure = programming error.
        ToolIdentity::script(&self.name).expect("entry name validated at construction")
    }
}

/// Parameter table → JSON Schema envelope: `type: object` + properties + required=all +
/// `additionalProperties: false` (in v1 all parameters are de facto required, see contract §appendix 3).
pub fn parameters_envelope(params: &BTreeMap<String, ParamSpec>) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, spec) in params {
        let mut schema = serde_json::Map::new();
        let ty = match spec.ty {
            ParamType::String => "string",
            ParamType::Number => "number",
            ParamType::Boolean => "boolean",
        };
        schema.insert("type".into(), json!(ty));
        if let Some(desc) = &spec.description {
            schema.insert("description".into(), json!(desc));
        }
        properties.insert(name.clone(), Value::Object(schema));
        required.push(json!(name));
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}
