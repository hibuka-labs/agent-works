#![no_main]

use libfuzzer_sys::fuzz_target;
use agent_works::multi_agent::path::AgentPath;

fuzz_target!(|data: &str| {
    let result = AgentPath::parse(data);
    // If parse succeeds, to_string().parse() should roundtrip
    if let Some(path) = result {
        let s = path.to_string();
        let reparsed = AgentPath::parse(&s);
        assert_eq!(Some(path), reparsed, "roundtrip failed for {:?}", s);
    }
});
