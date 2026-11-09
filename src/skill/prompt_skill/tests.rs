use super::*;

const TEST_PROMPT_SKILL: &str = r#"---
name: test-prompt
description: A test prompt skill
category: 测试
tags: [test, example]
version: "1.0"
author: test
---

# Test Prompt Skill

## 角色
你是一个测试助手。

## 指导
- 始终用中文回复
- 保持简洁
"#;

#[test]
fn test_parse_markdown() {
    let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
    assert_eq!(skill.name(), "test-prompt");
    assert_eq!(skill.brief_description(), "A test prompt skill");
    assert_eq!(skill.category(), "测试");
    assert_eq!(skill.version(), "1.0");
    assert_eq!(skill.author(), "test");
    assert_eq!(skill.tags().len(), 2);
    assert!(skill.body().contains("# Test Prompt Skill"));
}

#[test]
fn test_full_content() {
    let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
    let desc = skill.detailed_description();
    assert!(desc.contains("---"));
    assert!(desc.contains("name: test-prompt"));
    assert!(desc.contains("# Test Prompt Skill"));
}

#[test]
fn test_no_frontmatter() {
    let md = "# Just a heading\nSome content";
    assert!(PromptSkill::from_markdown(md).is_err());
}

#[test]
fn test_empty_name() {
    let md = "---\nname: ''\ndescription: test\n---\nContent";
    assert!(PromptSkill::from_markdown(md).is_err());
}

#[test]
fn test_empty_description() {
    let md = "---\nname: test-skill\ndescription: ''\n---\nContent";
    assert!(PromptSkill::from_markdown(md).is_err());
}

#[test]
fn test_invalid_name() {
    let md = "---\nname: invalid name!\ndescription: test\n---\nContent";
    assert!(PromptSkill::from_markdown(md).is_err());
}

#[test]
fn test_minimal_frontmatter() {
    let md = "---\nname: minimal\ndescription: Minimal skill\n---\nBody here";
    let skill = PromptSkill::from_markdown(md).unwrap();
    assert_eq!(skill.name(), "minimal");
    assert_eq!(skill.version(), "1.0"); // default
    assert_eq!(skill.author(), ""); // default
    assert_eq!(skill.category(), ""); // default
    assert!(skill.tags().is_empty());
}

// ── Phase 4: new frontmatter fields ──

const SKILL_WITH_NEW_FIELDS: &str = r#"---
name: advanced-skill
description: An advanced skill with new fields
category: ops
tags: [deploy, production]
version: "2.0"
author: ops-team
allowed-tools: [bash, git, docker]
disallowed-tools: [rm, drop]
model: opus
user-invocable: true
disable-model-invocation: true
arguments:
  - name: branch
    description: Target branch
    required: true
  - name: env
    description: Target environment
context: fork
paths: ["src/**", "deploy/**"]
---

# Advanced Skill

Deploy to $env on branch $branch.
Scripts are in $PHI_SKILL_DIR/scripts/
Full arguments: $ARGUMENTS
"#;

#[test]
fn test_new_frontmatter_fields() {
    let skill = PromptSkill::from_markdown(SKILL_WITH_NEW_FIELDS).unwrap();
    assert_eq!(skill.name(), "advanced-skill");

    // allowed-tools / disallowed-tools
    assert_eq!(skill.allowed_tools(), &["bash", "git", "docker"]);
    assert_eq!(skill.disallowed_tools(), &["rm", "drop"]);

    // model
    assert_eq!(skill.model_override(), Some("opus"));

    // user-invocable
    assert!(skill.is_user_invocable());

    // disable-model-invocation
    assert!(skill.disable_model_invocation());

    // context
    assert_eq!(skill.context_mode(), Some("fork"));

    // paths
    assert_eq!(skill.path_patterns(), &["src/**", "deploy/**"]);

    // Frontmatter introspection
    let fm = skill.frontmatter();
    assert_eq!(fm.arguments.len(), 2);
    assert_eq!(fm.arguments[0].name, "branch");
    assert!(fm.arguments[0].required);
    assert_eq!(fm.arguments[1].name, "env");
    assert!(!fm.arguments[1].required);
}

#[test]
fn test_new_fields_defaults() {
    // Old format without new fields should still parse with defaults
    let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
    assert!(skill.allowed_tools().is_empty());
    assert!(skill.disallowed_tools().is_empty());
    assert_eq!(skill.model_override(), None);
    assert!(skill.is_user_invocable()); // default true
    assert!(!skill.disable_model_invocation()); // default false
    assert_eq!(skill.context_mode(), None);
    assert!(skill.path_patterns().is_empty());
}

#[test]
fn test_model_inherit_treated_as_none() {
    let md = "---\nname: inherit-skill\ndescription: test\nmodel: inherit\n---\nBody";
    let skill = PromptSkill::from_markdown(md).unwrap();
    assert_eq!(skill.model_override(), None);
}

#[test]
fn test_resolve_body() {
    let skill = PromptSkill::from_markdown(SKILL_WITH_NEW_FIELDS).unwrap();

    let mut params = HashMap::new();
    params.insert("branch".to_string(), "main".to_string());
    params.insert("env".to_string(), "staging".to_string());

    let resolved = skill.resolve_body(&params, "--force");

    assert!(resolved.contains("Deploy to staging on branch main."));
    assert!(resolved.contains("Full arguments: --force"));
    // $PHI_SKILL_DIR should NOT be replaced since skill was loaded from markdown (no dir)
    assert!(resolved.contains("$PHI_SKILL_DIR"));
}

#[test]
fn test_skill_dir_none_when_from_markdown() {
    let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
    assert!(skill.skill_dir().is_none());
    assert!(skill.scripts_dir().is_none());
    assert!(skill.references_dir().is_none());
    assert!(skill.templates_dir().is_none());
}

#[test]
fn test_read_reference_no_dir_returns_error() {
    let skill = PromptSkill::from_markdown(TEST_PROMPT_SKILL).unwrap();
    let result = skill.read_reference("guide.md");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no references/"));
}

// ── from_dir / scan_dir tests ──

#[test]
fn test_from_dir_loads_skill() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("test-skill");
    std::fs::create_dir(&skill_dir).unwrap();

    let skill_md = r#"---
name: test-skill
description: A skill from directory
---
# Skill Body
"#;
    std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();
    std::fs::create_dir(skill_dir.join("scripts")).unwrap();
    std::fs::create_dir(skill_dir.join("references")).unwrap();

    let skill = PromptSkill::from_dir(&skill_dir).unwrap();
    assert_eq!(skill.name(), "test-skill");
    assert_eq!(skill.brief_description(), "A skill from directory");
    assert!(skill.skill_dir().is_some());
    assert!(skill.scripts_dir().is_some());
    assert!(skill.references_dir().is_some());
    assert!(skill.templates_dir().is_none());
}

#[test]
fn test_from_dir_name_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("wrong-name");
    std::fs::create_dir(&skill_dir).unwrap();

    let skill_md = r#"---
name: correct-name
description: Test
---
Body
"#;
    std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

    let result = PromptSkill::from_dir(&skill_dir);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not match"));
}

#[test]
fn test_from_dir_missing_skill_md() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("no-skill-md");
    std::fs::create_dir(&skill_dir).unwrap();

    let result = PromptSkill::from_dir(&skill_dir);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("SKILL.md not found"));
}

#[test]
fn test_scan_dir_discovers_skills() {
    let dir = tempfile::tempdir().unwrap();

    // Create skill directories
    for name in &["deploy", "review", "test"] {
        let skill_dir = dir.path().join(name);
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {}\ndescription: {} skill\n---\nBody",
                name, name
            ),
        )
        .unwrap();
    }

    // Create a non-skill directory (no SKILL.md)
    let misc_dir = dir.path().join("misc");
    std::fs::create_dir(&misc_dir).unwrap();

    let skills = PromptSkill::scan_dir(dir.path()).unwrap();
    assert_eq!(skills.len(), 3);

    let names: Vec<&str> = skills.iter().map(|s| s.name()).collect();
    assert_eq!(names, &["deploy", "review", "test"]); // sorted
}

#[test]
fn test_scan_dir_skips_underscore_prefixed_dirs_silently() {
    let dir = tempfile::tempdir().unwrap();

    // A live skill alongside a `_`-parked one whose SKILL.md would fail
    // validation anyway (dir name != skill name). The `_` dir must be
    // skipped without an error and without entering the result.
    for (name, skill_name) in &[("deploy", "deploy"), ("_disabled", "realname")] {
        let skill_dir = dir.path().join(name);
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {skill_name}\ndescription: d\n---\nBody"),
        )
        .unwrap();
    }

    let skills = PromptSkill::scan_dir(dir.path()).unwrap();
    let names: Vec<&str> = skills.iter().map(|s| s.name()).collect();
    assert_eq!(names, &["deploy"]);
}

#[test]
fn test_scan_dir_nonexistent_returns_empty() {
    let skills = PromptSkill::scan_dir("/nonexistent/path/12345").unwrap();
    assert!(skills.is_empty());
}

#[test]
fn test_read_reference_success() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("ref-skill");
    std::fs::create_dir(&skill_dir).unwrap();
    let refs_dir = skill_dir.join("references");
    std::fs::create_dir(&refs_dir).unwrap();

    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: ref-skill\ndescription: test\n---\nBody",
    )
    .unwrap();
    std::fs::write(
        refs_dir.join("guide.md"),
        "# Reference Guide\n\nImportant info.",
    )
    .unwrap();

    let skill = PromptSkill::from_dir(&skill_dir).unwrap();
    let content = skill.read_reference("guide.md").unwrap();
    assert!(content.contains("Reference Guide"));
    assert!(content.contains("Important info"));
}

#[test]
fn test_read_reference_path_traversal_prevented() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("secure-skill");
    std::fs::create_dir(&skill_dir).unwrap();
    let refs_dir = skill_dir.join("references");
    std::fs::create_dir(&refs_dir).unwrap();

    // Create a file INSIDE the skill dir but OUTSIDE references/
    // (e.g. skill_dir/secret.txt, accessible via ../secret.txt from references/)
    std::fs::write(skill_dir.join("secret.txt"), "top secret").unwrap();

    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: secure-skill\ndescription: test\n---\nBody",
    )
    .unwrap();

    let skill = PromptSkill::from_dir(&skill_dir).unwrap();
    let result = skill.read_reference("../secret.txt");
    assert!(
        result.is_err(),
        "expected error for path traversal, got {:?}",
        result.ok()
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("Path traversal") || err.contains("outside references"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn test_resolve_body_with_skill_dir() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("resolve-skill");
    std::fs::create_dir(&skill_dir).unwrap();

    let skill_md = r#"---
name: resolve-skill
description: test
---
Run scripts from $PHI_SKILL_DIR/scripts/
Arguments: $ARGUMENTS
"#;
    std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();

    let skill = PromptSkill::from_dir(&skill_dir).unwrap();
    let params = HashMap::new();
    let resolved = skill.resolve_body(&params, "extra args");

    assert!(resolved.contains(&*skill_dir.to_string_lossy()));
    assert!(resolved.contains("extra args"));
    assert!(!resolved.contains("$PHI_SKILL_DIR"));
    assert!(!resolved.contains("$ARGUMENTS"));
}

#[test]
fn test_resolve_body_prefix_clobbering_prevented() {
    // If $host is substituted before $hostname, "Connect to $hostname"
    // becomes "Connect to Xname" instead of "Connect to Y".
    let md = "---\nname: prefix-skill\ndescription: test\n---\nConnect to $hostname at $host";
    let skill = PromptSkill::from_markdown(md).unwrap();

    let mut params = HashMap::new();
    params.insert("host".to_string(), "192.168.1.1".to_string());
    params.insert("hostname".to_string(), "web-prod".to_string());

    let resolved = skill.resolve_body(&params, "");
    assert!(
        resolved.contains("Connect to web-prod at 192.168.1.1"),
        "expected correct substitution order, got: {}",
        resolved
    );
}

// ── proptest: split_frontmatter / from_markdown ──

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn from_markdown_never_panics(content in ".*") {
            let _ = PromptSkill::from_markdown(&content);
        }

        #[test]
        fn split_frontmatter_valid_input(
            fm_content in r"[a-z: \n]{5,100}",
            body_content in r"[a-zA-Z0-9 .,\n]{0,200}"
        ) {
            // Build valid markdown with frontmatter
            let name = "test-skill";
            let desc = "test description";
            let input = format!("---\nname: {}\ndescription: {}\n{}\n---\n{}", name, desc, fm_content, body_content);
            let result = split_frontmatter(&input);
            if let Ok((fm, body)) = result {
                assert!(fm.contains("test-skill"), "frontmatter should contain name");
                assert!(fm.contains("test description"), "frontmatter should contain description");
                // Body should not contain the closing ---
                assert!(!body.starts_with("---"), "body should not start with ---");
            }
        }

        #[test]
        fn from_markdown_preserves_body(
            body_text in r"[a-zA-Z0-9 .,\n]{1,200}"
        ) {
            let input = format!("---\nname: test-skill\ndescription: test desc\n---\n{}", body_text);
            if let Ok(skill) = PromptSkill::from_markdown(&input) {
                // Body should contain the original text (trimmed)
                let trimmed_body = body_text.trim();
                if !trimmed_body.is_empty() {
                    assert!(skill.body().contains(trimmed_body),
                        "body {:?} should contain {:?}", skill.body(), trimmed_body);
                }
            }
        }
    }
}
