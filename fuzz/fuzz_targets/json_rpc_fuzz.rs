#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = agent_works::fuzz::mcp_server::parse_json_rpc(data);
});
