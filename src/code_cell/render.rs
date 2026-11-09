//! Observation → tool content for the `code_cell` tool: the text stream
//! (with the effect tally and progress trail of knives ① and ③), the image
//! channel and the M3c spill-to-store capping — the model-facing surface of
//! one engine step.

use base64::Engine as _;

use agent_base::Content;
use codexec::{ArtifactHub, artifact_line};
use phi_ns::{ImageRef, ImageSource};

use super::{CellEffects, CellEvent, Observation};
use crate::script_tool::truncate_output;

/// Observation → tool content. The text stream — with its per-cell
/// `namespace:` snapshot lines and image `alt` floors — is what the model
/// reads; errors and events ride as their own lines at the tail so nothing
/// hides in the noise. Engine failures are content, not tool errors: the model
/// must see them and act (re-observe, rebuild references, try again).
///
/// One text block, self-truncated to `max_chars` with the family convention
/// (`truncate_output`: head 1/3 + `...[output truncated]` + tail 2/3 — error
/// and event lines land on the tail side and survive). The engine's hard
/// reject (`Tool 'code_cell' output exceeds the …-char limit`) must never fire
/// for a big print: the code already ran, and its result is the model's only
/// way forward. Images ride their own content and don't count against the cap.
/// Text is joined the same way `content_text` flattens tool output, so the
/// model-visible bytes are the multi-block shape's, cut when over budget.
///
/// M3c (action-executor-design §6.1): a body cut here does not die — the whole
/// joined body spills to the engine's artifact store and the block keeps the
/// preview plus the summary+reference line (both fitted inside `max_chars`).
/// Spilled bodies also get a typed [`Content::Artifact`] face alongside the
/// text (the line is the model face; the typed item is for UI/logs).
pub(super) fn render(
    obs: Observation,
    max_chars: Option<usize>,
    hub: Option<&ArtifactHub>,
) -> Vec<Content> {
    let mut blocks: Vec<String> = Vec::new();
    let mut artifacts: Vec<Content> = Vec::new();
    if !obs.stdout.is_empty() {
        blocks.push(obs.stdout);
    }
    if !obs.stderr.is_empty() {
        blocks.push(format!("stderr:\n{}", obs.stderr));
    }
    // Knife③: what the run looked like while it ran. The trail is a tail-side
    // line like everything else summary-shaped — a cut keeps it via the family
    // preview, a spill keeps it byte for byte.
    if !obs.progress.is_empty() {
        blocks.push(format!("progress:\n{}", obs.progress.join("\n")));
    }
    // Knife①: what the world actually did, per cell — the model must not have
    // to infer effects from what the code chose to print.
    for tally in &obs.effects {
        blocks.push(effects_line(tally));
    }
    if let Some(err) = &obs.error {
        blocks.push(format!("error: {err}"));
    }
    let mut images = Vec::new();
    for event in &obs.events {
        match event {
            CellEvent::Image(image) => images.push(image_content(image)),
            CellEvent::TodoChanged { id, status } => {
                blocks.push(format!("todo {id}: {status}"));
            }
            CellEvent::Notice { text } => blocks.push(format!("notice: {text}")),
            // A spilled body's standing reference (T13): summary + fetch
            // instruction in the text stream, typed face alongside.
            CellEvent::Artifact {
                capture_id,
                kind,
                bytes,
                summary,
            } => {
                blocks.push(artifact_line(capture_id, kind, *bytes, summary));
                artifacts.push(Content::artifact(
                    capture_id.clone(),
                    kind.clone(),
                    summary.clone(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    if blocks.is_empty() {
        if images.is_empty() {
            out.push(Content::text("(no output)"));
        }
    } else {
        out.push(Content::text(fit_to_cap(
            &blocks.join("\n"),
            max_chars,
            hub,
            &mut artifacts,
        )));
    }
    out.extend(artifacts);
    out.extend(images);
    out
}

/// One cell's mediated-effect tally, one line: `effects: cell 0: 2 read_fs,
/// 1 write_fs (1 denied)`. Counts only — the journal stays the full record.
fn effects_line(tally: &CellEffects) -> String {
    let counts = tally
        .counts
        .iter()
        .map(|(cap, n)| format!("{n} {cap}"))
        .collect::<Vec<_>>()
        .join(", ");
    let denied = if tally.denied > 0 {
        format!(" ({} denied)", tally.denied)
    } else {
        String::new()
    };
    if counts.is_empty() {
        return format!(
            "effects: cell {}: {} denied",
            tally.cell_index, tally.denied
        );
    }
    format!("effects: cell {}: {}{denied}", tally.cell_index, counts)
}

/// Cap one text block: over budget, spill the whole body to the store (M3c)/// and keep the family preview plus the summary+reference line, fitted inside
/// `max_chars` — the line counts against the budget so the hard reject can
/// never fire on a spill. The spill writes through the store directly (not
/// `hub.spill`): the hub's event queue belongs to the engine's round and this
/// render runs after the fold drained it. The typed face of a spill lands in
/// `artifacts`. No store (or a store failure): the family truncation stands,
/// body lost as before — marked, never silent.
fn fit_to_cap(
    body: &str,
    max_chars: Option<usize>,
    hub: Option<&ArtifactHub>,
    artifacts: &mut Vec<Content>,
) -> String {
    let Some(max_chars) = max_chars else {
        return body.to_string();
    };
    if body.chars().count() <= max_chars {
        return body.to_string();
    }
    let Some(r) = hub.and_then(|h| h.store().spill("observation", body).ok()) else {
        return truncate_output(body, Some(max_chars));
    };
    let line = artifact_line(&r.capture_id, &r.kind, r.bytes, &r.summary);
    if line.chars().count() + 1 >= max_chars {
        // The reference line cannot fit — the cap is the hard contract (the
        // engine rejects the whole call over it), so plain truncation wins
        // and no typed face points at a reference the model cannot see.
        return truncate_output(body, Some(max_chars));
    }
    artifacts.push(Content::artifact(
        r.capture_id.clone(),
        r.kind.clone(),
        r.summary.clone(),
    ));
    let budget = max_chars - line.chars().count() - 1;
    format!("{}\n{}", truncate_output(body, Some(budget)), line)
}

/// One image event → one `Content::Image`. Bytes that cannot be produced are
/// reported as text — the model must never lose the sentence (the `alt` floor
/// is already in `stdout`; this covers the bytes themselves).
fn image_content(image: &ImageRef) -> Content {
    let bytes = match &image.source {
        ImageSource::Inline(bytes) => Ok(bytes.clone()),
        ImageSource::Path(path) => std::fs::read(path),
    };
    match bytes {
        Ok(bytes) => {
            let (bytes, mime) = wire_image_bytes(bytes, image.mime);
            Content::image(
                base64::engine::general_purpose::STANDARD.encode(bytes),
                mime,
            )
        }
        Err(e) => Content::text(format!("image \"{}\": bytes unavailable ({e})", image.alt)),
    }
}

/// Raw PNG screenshots are ~800 KB each and base64 inflates them by a third;
/// a request carrying a few of them tripped the gateway's body limit (HTTP
/// 413, session 20261030_862ddf5c). Large pictures are re-encoded as JPEG
/// **at unchanged pixel dimensions** — the model does coordinate math off the
/// image's geometry, so only the codec may change, never the layout. The
/// on-disk artifact keeps full fidelity; only the model-facing copy is
/// budgeted. Anything that fails to decode, or that re-encodes no smaller,
/// goes out byte-for-byte as it came in.
const IMAGE_TRANSCODE_OVER_BYTES: usize = 100 * 1024;
const IMAGE_JPEG_QUALITY: u8 = 85;

/// Re-encode an oversized picture for the wire; returns the bytes and the
/// mime that truthfully describes them.
fn wire_image_bytes(bytes: Vec<u8>, mime: &'static str) -> (Vec<u8>, &'static str) {
    if bytes.len() <= IMAGE_TRANSCODE_OVER_BYTES {
        return (bytes, mime);
    }
    let Ok(img) = image::load_from_memory(&bytes) else {
        return (bytes, mime);
    };
    // JPEG carries no alpha; a screenshot's transparency is meaningless to
    // the model, and RGB is what every screenshot decodes to anyway.
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, IMAGE_JPEG_QUALITY);
    if img.to_rgb8().write_with_encoder(encoder).is_err() || out.len() >= bytes.len() {
        return (bytes, mime);
    }
    (out, "image/jpeg")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script_tool::TRUNCATION_MARKER;
    use codexec::observation::ExecError;
    use phi_cap::Capability;

    #[test]
    fn inline_image_bytes_ride_content() {
        let image = ImageRef::inline("image/png", vec![1, 2, 3], "three bytes");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/png");
                assert_eq!(
                    data,
                    base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3])
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn a_failed_image_read_becomes_an_honest_sentence() {
        let image = ImageRef::from_path("image/png", "/nonexistent/never.png", "a shot of nothing");
        match image_content(&image) {
            Content::Text { text } => {
                assert!(
                    text.contains("a shot of nothing"),
                    "alt must survive: {text}"
                );
                assert!(text.contains("bytes unavailable"), "{text}");
            }
            other => panic!("expected honest text, got {other:?}"),
        }
    }

    /// A deterministic noisy picture: a flat colour compresses to almost
    /// nothing, so the fixture needs entropy to land over the transcode
    /// threshold the way a real screenshot does.
    fn noisy_png(width: u32, height: u32) -> Vec<u8> {
        let mut rgb = image::RgbImage::new(width, height);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for p in rgb.pixels_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = image::Rgb([
                (state >> 33) as u8,
                (state >> 41) as u8,
                (state >> 49) as u8,
            ]);
        }
        let mut buf = Vec::new();
        rgb.write_with_encoder(image::codecs::png::PngEncoder::new(&mut buf))
            .unwrap();
        buf
    }

    #[test]
    fn a_large_png_goes_on_the_wire_as_a_smaller_jpeg_of_the_same_size() {
        let png = noisy_png(900, 600);
        assert!(
            png.len() > IMAGE_TRANSCODE_OVER_BYTES,
            "fixture must exceed the threshold, got {} bytes",
            png.len()
        );
        let image = ImageRef::inline("image/png", png.clone(), "a noisy screenshot");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/jpeg", "the wire copy is a JPEG");
                let jpeg = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .unwrap();
                assert!(
                    jpeg.len() < png.len(),
                    "transcode must shrink: {} vs {} original bytes",
                    jpeg.len(),
                    png.len()
                );
                let decoded = image::load_from_memory(&jpeg).unwrap();
                assert_eq!(
                    (decoded.width(), decoded.height()),
                    (900, 600),
                    "geometry is what the model's coordinate math relies on — only the codec may change"
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn a_small_picture_keeps_its_exact_bytes() {
        let mut buf = Vec::new();
        image::RgbImage::new(16, 16)
            .write_with_encoder(image::codecs::png::PngEncoder::new(&mut buf))
            .unwrap();
        assert!(buf.len() <= IMAGE_TRANSCODE_OVER_BYTES);
        let image = ImageRef::inline("image/png", buf.clone(), "a tiny icon");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/png", "under the budget: untouched");
                assert_eq!(data, base64::engine::general_purpose::STANDARD.encode(buf));
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn undecodable_oversized_bytes_still_reach_the_model() {
        let garbage = vec![0xABu8; IMAGE_TRANSCODE_OVER_BYTES + 1];
        let image = ImageRef::inline("image/png", garbage.clone(), "a corrupt capture");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(
                    mime_type, "image/png",
                    "no silent re-label when nothing was re-encoded"
                );
                assert_eq!(
                    data,
                    base64::engine::general_purpose::STANDARD.encode(garbage),
                    "a picture the codec cannot parse is still better than a silent loss"
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_observation_says_so() {
        let out = render(Observation::empty(), None, None);
        assert_eq!(out.len(), 1);
        assert_eq!(content_text(&out), "(no output)");
    }

    #[test]
    fn an_oversized_observation_is_truncated_not_rejected() {
        // Regression: a big print used to come back as
        // `Tool 'code_cell' output exceeds the 16000-char limit` — the engine
        // rejected the whole call, so the model lost the result of code that
        // had already run (session 20261029_30f097df).
        let obs = Observation {
            stdout: format!("HEAD\n{}\nTAIL", "x".repeat(30_000)),
            stderr: "a warning from the tail side".to_string(),
            error: Some(ExecError::CellError {
                index: 0,
                kind: "ValueError".to_string(),
                message: "boom".to_string(),
            }),
            events: vec![CellEvent::TodoChanged {
                id: "1".to_string(),
                status: "done".to_string(),
            }],
            ..Observation::empty()
        };
        let out = render(obs, Some(16_000), None);
        let text = content_text(&out);
        assert!(
            text.chars().count() <= 16_000,
            "must fit the cap, got {} chars",
            text.chars().count()
        );
        assert!(text.contains(TRUNCATION_MARKER), "{text:.60}");
        assert!(text.starts_with("HEAD"), "{text:.60}");
        for tail_piece in [
            "TAIL",
            "a warning from the tail side",
            "error in block #0 (ValueError): boom",
            "todo 1: done",
        ] {
            assert!(text.contains(tail_piece), "lost {tail_piece:?}: {text:.60}");
        }
    }

    #[test]
    fn a_small_observation_rides_whole() {
        let obs = Observation {
            stdout: "42".to_string(),
            ..Observation::empty()
        };
        let out = render(obs, Some(16_000), None);
        assert_eq!(content_text(&out), "42");
    }

    /// Knives ① + ③: the effect tally and the progress trail are part of the
    /// observation — the model sees what the world did and how the run went,
    /// not only what the code chose to print.
    #[test]
    fn the_effect_tally_and_progress_trail_reach_the_model() {
        let obs = Observation {
            progress: vec![
                "[0.3s] downloaded 40%".to_string(),
                "[1.2s] still running".to_string(),
            ],
            effects: vec![
                CellEffects {
                    cell_index: 0,
                    counts: vec![(Capability::ReadFs, 2), (Capability::WriteFs, 1)],
                    denied: 1,
                },
                CellEffects {
                    cell_index: 1,
                    counts: vec![(Capability::Net, 1)],
                    denied: 0,
                },
            ],
            ..Observation::empty()
        };
        let text = content_text(&render(obs, None, None));
        assert!(
            text.contains("progress:\n[0.3s] downloaded 40%\n[1.2s] still running"),
            "the trail rides in tick order: {text}"
        );
        assert!(
            text.contains("effects: cell 0: 2 read_fs, 1 write_fs (1 denied)"),
            "counts per capability, denials marked: {text}"
        );
        assert!(
            text.contains("effects: cell 1: 1 net"),
            "a clean cell shows no denial rider: {text}"
        );
    }

    #[test]
    fn a_denied_only_cell_says_so_in_one_line() {
        let obs = Observation {
            effects: vec![CellEffects {
                cell_index: 2,
                counts: Vec::new(),
                denied: 3,
            }],
            ..Observation::empty()
        };
        let text = content_text(&render(obs, None, None));
        assert!(
            text.contains("effects: cell 2: 3 denied"),
            "no dangling separator when nothing ran: {text}"
        );
    }

    #[test]
    fn images_survive_the_truncation() {
        let obs = Observation {
            stdout: "x".repeat(30_000),
            events: vec![CellEvent::Image(ImageRef::inline(
                "image/png",
                vec![1, 2, 3],
                "three bytes",
            ))],
            ..Observation::empty()
        };
        let out = render(obs, Some(500), None);
        assert!(
            out.iter().any(|c| matches!(c, Content::Image { .. })),
            "the typed channel must keep the image"
        );
        assert!(content_text(&out).chars().count() <= 500);
    }

    /// M3c (action-executor-design §6.1): over cap with a store mounted — the
    /// body is fetchable byte for byte, and the pane keeps the family preview
    /// plus the summary+reference line, fitted inside the cap (the hard
    /// reject must never fire) with a typed face alongside.
    #[test]
    fn an_oversized_observation_spills_and_keeps_a_reference_line() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hub = ArtifactHub::new(tmp.path().join("captures"));
        let body = format!("HEAD\n{}\nTAIL", "x".repeat(30_000));
        let obs = Observation {
            stdout: body.clone(),
            ..Observation::empty()
        };
        let out = render(obs, Some(16_000), Some(&hub));
        let text = content_text(&out);
        assert!(
            text.chars().count() <= 16_000,
            "must fit the cap, got {} chars",
            text.chars().count()
        );
        assert!(text.starts_with("HEAD"), "{text:.80}");
        assert!(
            text.contains("TAIL"),
            "the tail preview must survive: {text:.120}"
        );
        assert!(
            text.contains("obs.fetch('art_0001') for the full body"),
            "the way back must be on the pane: {text:.200}"
        );
        // 账-物对账: the whole joined body landed in the store, byte for byte.
        assert_eq!(
            hub.store().fetch("art_0001").as_deref(),
            Some(body.as_str()),
            "the spill must be the full body"
        );
        assert!(
            out.iter().any(
                |c| matches!(c, Content::Artifact { capture_id, .. } if capture_id == "art_0001")
            ),
            "the typed face rides alongside: {out:?}"
        );
    }

    /// A spilled body's event renders its standing reference — the line is the
    /// model face, the typed item marks it for UI/logs (content_text skips it).
    #[test]
    fn a_spilled_reference_event_renders_line_and_typed_face() {
        let obs = Observation {
            events: vec![CellEvent::Artifact {
                capture_id: "art_0001".to_string(),
                kind: "stdout".to_string(),
                bytes: 11,
                summary: "first body".to_string(),
            }],
            ..Observation::empty()
        };
        let out = render(obs, None, None);
        let text = content_text(&out);
        assert!(
            text.contains(
                "artifact art_0001 (stdout, 11 bytes): first body — obs.fetch('art_0001') for the full body"
            ),
            "{text}"
        );
        assert!(
            out.iter().any(|c| matches!(c, Content::Artifact { .. })),
            "{out:?}"
        );
        // The typed item must not double the line in the model face.
        assert_eq!(text.matches("first body").count(), 1, "{text}");
    }

    fn content_text(contents: &[Content]) -> String {
        contents
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
