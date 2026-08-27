#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = agent_works::fuzz::prompt_skill::split_frontmatter(data);
});
