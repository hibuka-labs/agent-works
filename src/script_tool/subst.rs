use agent_base::{AgentError, AgentResult};
use serde_json::Value;

/// `${param}` substitution per element (argv, no shell).
///
/// - Only the `${name}` form is recognized; bare text such as `$FOO` or anything outside the
///   `${…}` form is left untouched (`${ENV:…}` has already been resolved to a literal at
///   product-layer registration, so none should remain by this point);
/// - Parameter values are inserted verbatim, **no re-expansion**;
/// - Missing, non-scalar (null/array/object), or unclosed `${` → [`AgentError::ToolArgsInvalid`]
///   (fail-closed, tool error for that call).
pub fn substitute_params(tool_key: &str, run: &[String], args: &Value) -> AgentResult<Vec<String>> {
    let Some(obj) = args.as_object() else {
        return Err(AgentError::ToolArgsInvalid {
            name: tool_key.to_string(),
            raw: args.to_string(),
        });
    };
    run.iter()
        .map(|element| substitute_element(tool_key, element, obj, args))
        .collect()
}

fn invalid_args(tool_key: &str, raw_args: &Value) -> AgentError {
    AgentError::ToolArgsInvalid {
        name: tool_key.to_string(),
        raw: raw_args.to_string(),
    }
}

fn substitute_element(
    tool_key: &str,
    element: &str,
    obj: &serde_json::Map<String, Value>,
    raw_args: &Value,
) -> AgentResult<String> {
    let mut out = String::with_capacity(element.len());
    let mut rest = element;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| invalid_args(tool_key, raw_args))?;
        let name = &after[..end];
        let value = obj
            .get(name)
            .ok_or_else(|| invalid_args(tool_key, raw_args))?;
        let scalar = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => return Err(invalid_args(tool_key, raw_args)), // null/array/object are non-scalar
        };
        out.push_str(&scalar); // inserted verbatim, no re-expansion
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}
