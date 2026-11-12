// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Streaming withhold for the arch-0 Mistral v3 `[TOOL_CALLS]` span.
//!
//! The embedded chat template renders available tools as `[AVAILABLE_TOOLS]`
//! and replays an assistant call as `[TOOL_CALLS]` plus a JSON array of
//! `{name, arguments, id?}`. The model emits that same span and then eos.
//! Eos sits after the closing `]`. This parser reads that span. It does not
//! accept Hermes `<tool_call>` or Qwen XML.

use crate::prompt_frame::ToolCall;

/// Marker the Mistral v3 template writes in front of the call array.
pub const LLAMA_TOOL_CALLS_MARKER: &str = "[TOOL_CALLS]";

/// Why a `[TOOL_CALLS]` span cannot be released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaToolSpanError {
    /// Eos arrived before the JSON array closed.
    Unclosed,
    /// The array closed, or a non-array byte arrived, and the body is not calls.
    Malformed(&'static str),
}

impl std::fmt::Display for LlamaToolSpanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unclosed => f.write_str("unclosed [TOOL_CALLS] span"),
            Self::Malformed(reason) => write!(f, "malformed [TOOL_CALLS] span: {reason}"),
        }
    }
}

impl std::error::Error for LlamaToolSpanError {}

/// How the llama decode turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaToolEnd {
    /// An eos / terminator token ended the turn.
    Stop,
    /// `max_tokens` (or the equivalent cap) ended the turn.
    Length,
}

/// Legacy terminal for a llama turn that may have parsed tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaToolTerminal {
    pub finish_reason: &'static str,
    /// Stage and emit calls only when this is true. Length never releases.
    pub release_calls: bool,
}

/// `finish_reason` and whether parsed calls may leave the process.
///
/// Length releases nothing, including calls whose span already closed.
/// Stop releases calls only when at least one parsed.
pub fn llama_tool_terminal(end: LlamaToolEnd, call_count: usize) -> LlamaToolTerminal {
    match end {
        LlamaToolEnd::Length => LlamaToolTerminal {
            finish_reason: "length",
            release_calls: false,
        },
        LlamaToolEnd::Stop if call_count > 0 => LlamaToolTerminal {
            finish_reason: "tool_calls",
            release_calls: true,
        },
        LlamaToolEnd::Stop => LlamaToolTerminal {
            finish_reason: "stop",
            release_calls: false,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prose,
    Body,
}

/// Withholds `[TOOL_CALLS]` arrays from content while a tools request is live.
///
/// Disabled, `push` returns its input unchanged and records no calls.
pub struct LlamaToolSplitter {
    enabled: bool,
    phase: Phase,
    pending: String,
    calls: Vec<ToolCall>,
}

impl LlamaToolSplitter {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            phase: Phase::Prose,
            pending: String::new(),
            calls: Vec::new(),
        }
    }

    /// Feed decode text in order. Returns content that is safe to stream.
    /// A trailing prefix of `[TOOL_CALLS]` is held so a marker split across
    /// tokens is still recognized. A closed body that is not an array of
    /// calls is an error.
    pub fn push(&mut self, text: &str) -> Result<String, LlamaToolSpanError> {
        if !self.enabled {
            return Ok(text.to_owned());
        }
        self.pending.push_str(text);
        let mut visible = String::new();
        loop {
            match self.phase {
                Phase::Prose => {
                    if let Some(at) = self.pending.find(LLAMA_TOOL_CALLS_MARKER) {
                        visible.push_str(&self.pending[..at]);
                        self.pending.drain(..at + LLAMA_TOOL_CALLS_MARKER.len());
                        self.phase = Phase::Body;
                    } else {
                        let hold = marker_prefix_suffix_len(&self.pending);
                        let release = self.pending.len() - hold;
                        visible.push_str(&self.pending[..release]);
                        self.pending.drain(..release);
                        break;
                    }
                }
                Phase::Body => match completed_array_end(&self.pending)? {
                    None => break,
                    Some(end) => {
                        let parsed = parse_call_array(&self.pending[..end])?;
                        self.calls.extend(parsed);
                        self.pending.drain(..end);
                        self.phase = Phase::Prose;
                    }
                },
            }
        }
        Ok(visible)
    }

    /// End the turn. Stop with an open array is [`LlamaToolSpanError::Unclosed`].
    /// Length drops the open body and every call parsed earlier in the turn,
    /// and returns any held prose prefix. A held marker prefix that never
    /// became `[TOOL_CALLS]` is prose.
    pub fn finish(&mut self, end: LlamaToolEnd) -> Result<String, LlamaToolSpanError> {
        if !self.enabled {
            return Ok(String::new());
        }
        match end {
            LlamaToolEnd::Length => {
                let visible = match self.phase {
                    Phase::Prose => std::mem::take(&mut self.pending),
                    Phase::Body => {
                        self.pending.clear();
                        String::new()
                    }
                };
                self.calls.clear();
                self.phase = Phase::Prose;
                Ok(visible)
            }
            LlamaToolEnd::Stop => match self.phase {
                Phase::Prose => Ok(std::mem::take(&mut self.pending)),
                Phase::Body => Err(LlamaToolSpanError::Unclosed),
            },
        }
    }

    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// True while a `[TOOL_CALLS]` array is open. Those bytes are not prose.
    pub fn in_body(&self) -> bool {
        self.phase == Phase::Body
    }

    /// Prose prefix held back so a split `[TOOL_CALLS]` marker can still match.
    /// Empty while a body is open. The caller that classifies the following
    /// prose takes this back; otherwise the prefix is stranded until finish.
    pub fn take_prose_hold(&mut self) -> String {
        if self.in_body() {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

fn marker_prefix_suffix_len(text: &str) -> usize {
    (1..LLAMA_TOOL_CALLS_MARKER.len())
        .rev()
        .find(|&len| {
            len <= text.len()
                && text.is_char_boundary(text.len() - len)
                && LLAMA_TOOL_CALLS_MARKER.starts_with(&text[text.len() - len..])
        })
        .unwrap_or(0)
}

/// Byte index just past a completed JSON array, leading whitespace allowed.
/// `Ok(None)` means the array has not closed yet.
fn completed_array_end(body: &str) -> Result<Option<usize>, LlamaToolSpanError> {
    let trimmed = body.trim_start();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let start = body.len() - trimmed.len();
    let bytes = trimmed.as_bytes();
    if bytes[0] != b'[' {
        return Err(LlamaToolSpanError::Malformed(
            "tool call body is not a JSON array",
        ));
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &byte) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth < 0 {
                    return Err(LlamaToolSpanError::Malformed(
                        "unbalanced JSON in tool call",
                    ));
                }
                if depth == 0 {
                    if byte != b']' {
                        return Err(LlamaToolSpanError::Malformed(
                            "tool call body closed on an object, not an array",
                        ));
                    }
                    return Ok(Some(start + i + 1));
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn parse_call_array(body: &str) -> Result<Vec<ToolCall>, LlamaToolSpanError> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|_| LlamaToolSpanError::Malformed("tool call array is not valid JSON"))?;
    let Some(items) = value.as_array() else {
        return Err(LlamaToolSpanError::Malformed(
            "tool call body is not a JSON array",
        ));
    };
    items.iter().map(parse_one_call).collect()
}

fn parse_one_call(value: &serde_json::Value) -> Result<ToolCall, LlamaToolSpanError> {
    let Some(obj) = value.as_object() else {
        return Err(LlamaToolSpanError::Malformed("tool call is not an object"));
    };
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|name| !name.is_empty())
        .ok_or(LlamaToolSpanError::Malformed("tool call is missing a name"))?;
    let arguments = match obj.get("arguments") {
        None => serde_json::json!({}),
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map.clone()),
        Some(serde_json::Value::String(text)) => {
            let parsed: serde_json::Value = serde_json::from_str(text).map_err(|_| {
                LlamaToolSpanError::Malformed("tool call arguments string is not JSON")
            })?;
            if !parsed.is_object() {
                return Err(LlamaToolSpanError::Malformed(
                    "tool call arguments string is not an object",
                ));
            }
            parsed
        }
        Some(_) => {
            return Err(LlamaToolSpanError::Malformed(
                "tool call arguments are not an object",
            ));
        }
    };
    let id = match obj.get("id") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(id)) => Some(id.clone()),
        Some(_) => {
            return Err(LlamaToolSpanError::Malformed(
                "tool call id is not a string",
            ));
        }
    };
    Ok(ToolCall {
        id,
        name: name.to_owned(),
        arguments,
        rendered_body: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        llama_tool_terminal, LlamaToolEnd, LlamaToolSpanError, LlamaToolSplitter, LlamaToolTerminal,
    };
    use serde_json::json;

    fn push_chunks(
        splitter: &mut LlamaToolSplitter,
        text: &str,
        chunk: usize,
    ) -> Result<String, LlamaToolSpanError> {
        let mut visible = String::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut end = chunk.min(rest.len());
            while end > 0 && !rest.is_char_boundary(end) {
                end -= 1;
            }
            if end == 0 {
                end = rest
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(rest.len());
            }
            visible.push_str(&splitter.push(&rest[..end])?);
            rest = &rest[end..];
        }
        Ok(visible)
    }

    #[test]
    fn completed_call_is_withheld_and_parsed() {
        let raw = r#"I'll check. [TOOL_CALLS][{"name":"get_weather","arguments":{"city":"SF"},"id":"abc123xyz"}] Done."#;
        let mut splitter = LlamaToolSplitter::new(true);
        let mut visible = push_chunks(&mut splitter, raw, 1).unwrap();
        visible.push_str(&splitter.finish(LlamaToolEnd::Stop).unwrap());
        assert_eq!(visible, "I'll check.  Done.");
        assert!(!visible.contains("[TOOL_CALLS]"));
        assert_eq!(splitter.calls().len(), 1);
        let call = &splitter.calls()[0];
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, json!({"city": "SF"}));
        assert_eq!(call.id.as_deref(), Some("abc123xyz"));
        assert_eq!(
            llama_tool_terminal(LlamaToolEnd::Stop, splitter.calls().len()),
            LlamaToolTerminal {
                finish_reason: "tool_calls",
                release_calls: true,
            }
        );
    }

    #[test]
    fn prose_is_not_a_tool_call() {
        let mut splitter = LlamaToolSplitter::new(true);
        let text = "The weather looks fine. Use tools if you need them.";
        let mut visible = splitter.push(text).unwrap();
        visible.push_str(&splitter.finish(LlamaToolEnd::Stop).unwrap());
        assert_eq!(visible, text);
        assert!(splitter.calls().is_empty());
        assert_eq!(
            llama_tool_terminal(LlamaToolEnd::Stop, splitter.calls().len()),
            LlamaToolTerminal {
                finish_reason: "stop",
                release_calls: false,
            }
        );
    }

    #[test]
    fn hermes_markup_stays_content() {
        let text =
            r#"prelude<tool_call>{"name":"read","arguments":{"path":"/etc/hostname"}}</tool_call>"#;
        let mut splitter = LlamaToolSplitter::new(true);
        let mut visible = splitter.push(text).unwrap();
        visible.push_str(&splitter.finish(LlamaToolEnd::Stop).unwrap());
        assert_eq!(visible, text);
        assert!(splitter.calls().is_empty());
    }

    #[test]
    fn marker_inside_argument_string_does_not_split_the_array() {
        let raw = r#"[TOOL_CALLS][{"name":"echo","arguments":{"text":"see [TOOL_CALLS] later"}}]"#;
        let mut splitter = LlamaToolSplitter::new(true);
        let visible = push_chunks(&mut splitter, raw, 3).unwrap();
        assert!(splitter.finish(LlamaToolEnd::Stop).unwrap().is_empty());
        assert!(visible.is_empty());
        assert_eq!(splitter.calls().len(), 1);
        assert_eq!(
            splitter.calls()[0].arguments["text"],
            "see [TOOL_CALLS] later"
        );
    }

    #[test]
    fn im_end_inside_argument_string_is_kept() {
        let raw = r#"[TOOL_CALLS][{"name":"echo","arguments":{"text":"a <|im_end|> b"}}]"#;
        let mut splitter = LlamaToolSplitter::new(true);
        assert!(splitter.push(raw).unwrap().is_empty());
        assert!(splitter.finish(LlamaToolEnd::Stop).unwrap().is_empty());
        assert_eq!(splitter.calls()[0].arguments["text"], "a <|im_end|> b");
    }

    #[test]
    fn two_spans_and_argument_forms() {
        let raw = r#"ok [TOOL_CALLS][{"name":"a","arguments":{"x":1}},{"name":"b","arguments":"{\"y\":\"z\"}"}] and [TOOL_CALLS][{"name":"ping"}] end"#;
        let mut splitter = LlamaToolSplitter::new(true);
        let mut visible = splitter.push(raw).unwrap();
        visible.push_str(&splitter.finish(LlamaToolEnd::Stop).unwrap());
        assert_eq!(visible, "ok  and  end");
        assert_eq!(splitter.calls().len(), 3);
        assert_eq!(splitter.calls()[0].name, "a");
        assert_eq!(splitter.calls()[0].arguments, json!({"x": 1}));
        assert_eq!(splitter.calls()[1].name, "b");
        assert_eq!(splitter.calls()[1].arguments, json!({"y": "z"}));
        assert_eq!(splitter.calls()[2].name, "ping");
        assert_eq!(splitter.calls()[2].arguments, json!({}));
        assert!(splitter.calls()[2].id.is_none());
    }

    #[test]
    fn partial_marker_is_held_until_it_resolves() {
        let mut splitter = LlamaToolSplitter::new(true);
        assert_eq!(splitter.push("hello [TOOL").unwrap(), "hello ");
        assert!(splitter.calls().is_empty());
        assert!(splitter
            .push(r#"_CALLS][{"name":"ping","id":null}]"#)
            .unwrap()
            .is_empty());
        assert!(splitter.finish(LlamaToolEnd::Stop).unwrap().is_empty());
        assert_eq!(splitter.calls().len(), 1);
        assert_eq!(splitter.calls()[0].name, "ping");
        assert!(splitter.calls()[0].id.is_none());
    }

    #[test]
    fn held_bracket_that_is_not_a_marker_is_prose() {
        let mut splitter = LlamaToolSplitter::new(true);
        assert_eq!(splitter.push("hello [").unwrap(), "hello ");
        let tail = splitter.finish(LlamaToolEnd::Stop).unwrap();
        assert_eq!(tail, "[");
        assert!(splitter.calls().is_empty());
    }

    #[test]
    fn unclosed_span_at_stop_is_an_error() {
        let mut splitter = LlamaToolSplitter::new(true);
        assert!(splitter
            .push(r#"[TOOL_CALLS][{"name":"get_weather""#)
            .unwrap()
            .is_empty());
        assert_eq!(
            splitter.finish(LlamaToolEnd::Stop).unwrap_err(),
            LlamaToolSpanError::Unclosed
        );
    }

    #[test]
    fn length_drops_an_open_span_and_completed_calls() {
        let mut splitter = LlamaToolSplitter::new(true);
        let visible = splitter
            .push(r#"Note. [TOOL_CALLS][{"name":"get_weather","arguments":{"city":"SF"}}] [TOOL_CALLS][{"name":"ping""#)
            .unwrap();
        assert_eq!(visible, "Note.  ");
        assert_eq!(
            splitter.calls().len(),
            1,
            "the closed span parses before eos"
        );
        let tail = splitter.finish(LlamaToolEnd::Length).unwrap();
        assert!(tail.is_empty());
        assert!(splitter.calls().is_empty(), "length releases no calls");
        assert_eq!(
            llama_tool_terminal(LlamaToolEnd::Length, 1),
            LlamaToolTerminal {
                finish_reason: "length",
                release_calls: false,
            }
        );
    }

    #[test]
    fn closed_malformed_body_errors_immediately() {
        let mut splitter = LlamaToolSplitter::new(true);
        let error = splitter
            .push(r#"[TOOL_CALLS][{"name":"","arguments":{}}]"#)
            .unwrap_err();
        assert_eq!(
            error,
            LlamaToolSpanError::Malformed("tool call is missing a name")
        );
        let mut splitter = LlamaToolSplitter::new(true);
        let error = splitter.push(r#"[TOOL_CALLS]{"name":"ping"}"#).unwrap_err();
        assert_eq!(
            error,
            LlamaToolSpanError::Malformed("tool call body is not a JSON array")
        );
        let mut splitter = LlamaToolSplitter::new(true);
        let error = splitter
            .push(r#"[TOOL_CALLS][{"name":"ping","id":1}]"#)
            .unwrap_err();
        assert_eq!(
            error,
            LlamaToolSpanError::Malformed("tool call id is not a string")
        );
    }

    #[test]
    fn disabled_splitter_passes_the_marker_through() {
        let mut splitter = LlamaToolSplitter::new(false);
        let raw = r#"[TOOL_CALLS][{"name":"ping"}]"#;
        assert_eq!(splitter.push(raw).unwrap(), raw);
        assert!(splitter.finish(LlamaToolEnd::Stop).unwrap().is_empty());
        assert!(splitter.calls().is_empty());
    }
}
