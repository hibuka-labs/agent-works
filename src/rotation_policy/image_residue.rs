//! Eviction residue — what a rotated-out picture leaves behind.
//!
//! At [`crate::rotation_policy::TokenBudgetAction::Reset`] the outgoing
//! window is archived verbatim (pictures included); the fresh window gets
//! these lines instead of the pixels. Pure text, no model call: the reset
//! point sits on the turn's critical path and this strategy stays mechanical
//! (no LLM calls, no I/O — see the module doc on [`super`]). Recovery is
//! `read_image(path)` on a vision model.
//!
//! The `path="..."` fields come from the `<image name=N path="...">` labels
//! the paste pipeline writes next to the picture (phi-agent-tui
//! `run::image_label_lines`); pre-label transcripts degrade to an honest
//! `<path not recorded>` stub instead of guessing.

use agent_base::ChatMessage;

/// One residue line per picture in the outgoing window: says a picture
/// existed, where its bytes live, and how to see it again.
pub fn image_residue_lines(old_messages: &[ChatMessage]) -> Vec<String> {
    let mut out = Vec::new();
    for msg in old_messages {
        let (content, images) = match msg {
            ChatMessage::User {
                content, images, ..
            }
            | ChatMessage::Tool {
                content, images, ..
            } => (content, images),
            _ => continue,
        };
        if images.is_empty() {
            continue;
        }
        let path =
            extract_image_label_path(content).unwrap_or_else(|| "<path not recorded>".to_string());
        out.push(format!(
            "[an image attached earlier was rotated out of context — \
             path={path}; call read_image(path) to re-inspect]"
        ));
    }
    out
}

/// Pull `path="..."` out of a `<image name=N path="...">` label line —
/// the label format `image_label_lines` writes (phi-agent-tui).
fn extract_image_label_path(content: &str) -> Option<String> {
    let start = content.find("path=\"")? + "path=\"".len();
    let end = start + content[start..].find('"')?;
    Some(content[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::ImageAttachment;

    fn png_msg(content: &str) -> ChatMessage {
        ChatMessage::user_with_images(
            content,
            vec![ImageAttachment::Base64 {
                data: "aGVsbG8=".into(),
                media_type: Some("image/png".into()),
                detail: None,
            }],
        )
    }

    /// protects: the residue names the capture path from the label, so the
    /// fresh window can route a later question to `read_image`. fails_when:
    /// the path is lost, or the line forgets the recovery verb.
    #[test]
    fn residue_names_the_path_from_the_label() {
        let old = vec![png_msg(
            "<image name=1 path=\"/s/shot.png\">\nwhat is broken?",
        )];
        let lines = image_residue_lines(&old);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains("/s/shot.png"),
            "the path must survive: {}",
            lines[0]
        );
        assert!(
            lines[0].contains("read_image"),
            "the recovery verb must be named: {}",
            lines[0]
        );
    }

    /// protects: a picture from a pre-label transcript still leaves a stub —
    /// honest about the missing path instead of silent or invented.
    /// fails_when: no-label pictures vanish without residue, or the stub
    /// claims a path it does not have.
    #[test]
    fn residue_without_a_label_says_so() {
        let old = vec![png_msg("what is broken?")];
        let lines = image_residue_lines(&old);
        assert!(lines[0].contains("<path not recorded>"));
    }

    /// protects: word-only messages leave no residue — the fresh window is
    /// not padded with stubs for things that were never pictures.
    /// fails_when: text messages gain image stubs.
    #[test]
    fn text_only_messages_leave_no_residue() {
        let old = vec![ChatMessage::user("plain question")];
        assert!(image_residue_lines(&old).is_empty());
    }

    /// protects: the residue is words only — the pixels stay in the archive,
    /// never duplicated into the fresh window. fails_when: any base64 body
    /// rides the residue.
    #[test]
    fn residue_never_carries_pixels() {
        let old = vec![png_msg("<image name=1 path=\"/s/shot.png\">")];
        for line in image_residue_lines(&old) {
            assert!(!line.contains("aGVsbG8="));
        }
    }

    // ── integration: the hook inside `assemble_window` ──

    /// protects: a reset with a labeled picture in the outgoing window leaves
    /// a residue line naming the capture path and the recovery verb — and no
    /// pixels ride the fresh window. fails_when: the fresh window is silent
    /// about the evicted picture, or picture attachments leak into it.
    /// no_claim: the residue's exact wording beyond the two routing markers.
    #[test]
    fn reset_window_carries_image_residue_but_not_pixels() {
        use crate::rotation_policy::{HintSlot, TokenBudgetConfig, TokenBudgetCore};
        let core = TokenBudgetCore::new(TokenBudgetConfig::default(), Some("You are helpful."));
        let old = vec![
            ChatMessage::system("You are helpful."),
            png_msg("<image name=1 path=\"/s/shot.png\">\nwhat is broken?"),
        ];
        let window = core.build_reset_messages(&old, HintSlot::default(), 0);
        let mut saw_residue = false;
        for msg in &window {
            let (content, images): (&str, &[ImageAttachment]) = match msg {
                ChatMessage::System { content, .. } => (content.as_str(), &[]),
                ChatMessage::User {
                    content, images, ..
                }
                | ChatMessage::Tool {
                    content, images, ..
                } => (content.as_str(), images.as_slice()),
                _ => ("", &[]),
            };
            assert!(
                !content.contains("aGVsbG8="),
                "pixels must not ride the fresh window"
            );
            assert!(
                images.is_empty(),
                "a fresh window carries no picture attachments"
            );
            if content.contains("rotated out of context") {
                saw_residue = true;
                assert!(content.contains("/s/shot.png"), "the path must survive");
                assert!(
                    content.contains("read_image"),
                    "the recovery verb must be named"
                );
            }
        }
        assert!(saw_residue, "the fresh window must carry the residue line");
    }
}
