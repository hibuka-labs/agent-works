use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use agent_base::{AgentResult, ToolRegistry, AgentError};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{info, warn, error};

use super::client::{McpClient, McpToolAdapter};
use super::types::{McpServerConfig, McpToolInfo};

#[derive(Clone, Debug)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
    Unhealthy(String),
}

struct ServerEntry {
    config: McpServerConfig,
    clients: RwLock<Vec<Arc<McpClient>>>,
    max_connections: usize,
    tools: RwLock<Vec<McpToolInfo>>,
    state: RwLock<ConnectionState>,
    last_health_check: RwLock<Option<Instant>>,
    reconnect_attempts: RwLock<u32>,
    round_robin_idx: AtomicUsize,
}

impl ServerEntry {
    fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            clients: RwLock::new(Vec::new()),
            max_connections: 5,
            tools: RwLock::new(Vec::new()),
            state: RwLock::new(ConnectionState::Disconnected),
            last_health_check: RwLock::new(None),
            reconnect_attempts: RwLock::new(0),
            round_robin_idx: AtomicUsize::new(0),
        }
    }

    async fn create_client(&self) -> AgentResult<Arc<McpClient>> {
        let client = Arc::new(McpClient::new(self.config.transport.clone()).await?);
        Ok(client)
    }

    async fn get_available_client(&self) -> Option<Arc<McpClient>> {
        let clients = self.clients.read().await;
        if clients.is_empty() {
            return None;
        }
        // Round-robin selection across pooled connections
        let idx = self.round_robin_idx.fetch_add(1, Ordering::Relaxed) % clients.len();
        clients.get(idx).cloned()
    }

    async fn add_client(&self) -> AgentResult<()> {
        if self.clients.read().await.len() >= self.max_connections {
            return Err(AgentError::resource_unavailable(
                "Maximum connections reached".to_string(),
            ));
        }

        let client = self.create_client().await?;
        self.clients.write().await.push(client);

        let mut state = self.state.write().await;
        *state = ConnectionState::Connected;

        Ok(())
    }

    async fn reconnect(&self) -> AgentResult<()> {
        let mut attempts = self.reconnect_attempts.write().await;
        *attempts += 1;
        let attempt_count = *attempts;
        drop(attempts);

        // 指数退避
        let delay = Duration::from_millis(std::cmp::min(1000 * (2_u64.pow(attempt_count.min(5))), 30000));
        sleep(delay).await;

        // 清除现有连接
        self.clients.write().await.clear();

        match self.add_client().await {
            Ok(_) => {
                let mut state = self.state.write().await;
                *state = ConnectionState::Connected;

                let mut attempts = self.reconnect_attempts.write().await;
                *attempts = 0;

                info!("Successfully reconnected to MCP server: {}", self.config.name);
                Ok(())
            }
            Err(e) => {
                let mut state = self.state.write().await;
                *state = ConnectionState::Failed(e.to_string());
                error!("Failed to reconnect to {}: {e}", self.config.name);
                Err(e)
            }
        }
    }

    async fn health_check(&self) -> bool {
        if let Some(client) = self.get_available_client().await {
            match client.list_tools().await {
                Ok(_) => {
                    let mut state = self.state.write().await;
                    *state = ConnectionState::Connected;

                    let mut last_check = self.last_health_check.write().await;
                    *last_check = Some(Instant::now());

                    true
                }
                Err(e) => {
                    let mut state = self.state.write().await;
                    *state = ConnectionState::Unhealthy(e.to_string());
                    warn!("Health check failed for {}: {e}", self.config.name);
                    false
                }
            }
        } else {
            false
        }
    }
}

pub struct EnhancedMcpHub {
    servers: HashMap<String, Arc<ServerEntry>>,
    health_check_interval: Duration,
    shutdown: Arc<AtomicBool>,
    health_task: RwLock<Option<JoinHandle<()>>>,
}

impl EnhancedMcpHub {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            shutdown: Arc::new(AtomicBool::new(false)),
            health_task: RwLock::new(None),
        }
    }

    pub fn with_health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = interval;
        self
    }

    pub fn add_server(&mut self, config: McpServerConfig) {
        let entry = Arc::new(ServerEntry::new(config));
        self.servers.insert(entry.config.name.clone(), entry);
    }

    /// Connect to all configured servers. Continues connecting remaining servers
    /// even if one fails. Returns an error only if *all* servers fail to connect.
    pub async fn connect_all(&self) -> AgentResult<()> {
        let mut errors: Vec<(String, AgentError)> = Vec::new();

        for (name, entry) in &self.servers {
            match entry.add_client().await {
                Ok(_) => {
                    info!(server_name = %name, "Connected to MCP server");
                }
                Err(e) => {
                    let mut state = entry.state.write().await;
                    *state = ConnectionState::Failed(e.to_string());
                    error!(server_name = %name, error = %e, "Failed to connect to MCP server");
                    errors.push((name.clone(), e));
                }
            }
        }

        // Start background health check
        self.start_health_check_task().await;

        if errors.is_empty() {
            Ok(())
        } else if errors.len() == self.servers.len() {
            // All servers failed
            let msg = errors.iter()
                .map(|(name, e)| format!("{name}: {e}"))
                .collect::<Vec<_>>()
                .join("; ");
            Err(AgentError::internal(format!("All MCP servers failed to connect: {msg}")))
        } else {
            // Some succeeded, some failed — log but don't error
            let failed: Vec<&str> = errors.iter().map(|(n, _)| n.as_str()).collect();
            warn!("Some MCP servers failed to connect: {}", failed.join(", "));
            Ok(())
        }
    }

    pub async fn discover_all(&self) -> AgentResult<Vec<(String, Vec<McpToolInfo>)>> {
        let mut results = Vec::new();

        for (name, entry) in &self.servers {
            let Some(client) = entry.get_available_client().await else {
                warn!("Server {} not connected", name);
                continue;
            };

            match client.list_tools().await {
                Ok(tools) => {
                    {
                        let mut entry_tools = entry.tools.write().await;
                        *entry_tools = tools.clone();
                    }

                    info!("Discovered {} tools from {}", tools.len(), name);
                    results.push((name.clone(), tools));
                }
                Err(e) => {
                    warn!("Failed to discover tools from {}: {e}", name);
                    let mut state = entry.state.write().await;
                    *state = ConnectionState::Failed(format!("{e}"));
                }
            }
        }

        Ok(results)
    }

    /// Register all discovered tools into the given registry.
    /// This is an async method because it reads from `RwLock`-protected state.
    pub async fn register_all(&self, registry: &mut ToolRegistry) {
        for (name, entry) in &self.servers {
            let tools = entry.tools.read().await.clone();
            for tool_info in &tools {
                if let Some(client) = entry.get_available_client().await {
                    let adapter = McpToolAdapter::new(tool_info.clone(), client, name);
                    registry.register(adapter);
                }
            }
            info!(server_name = %name, tool_count = tools.len(), "Registered tools from MCP server");
        }
    }

    pub async fn disconnect_all(&self) {
        // Signal health check loop to stop
        self.shutdown.store(true, Ordering::SeqCst);

        // Abort the health check task if it exists
        if let Some(handle) = self.health_task.write().await.take() {
            handle.abort();
        }

        for (name, entry) in &self.servers {
            entry.clients.write().await.clear();
            let mut state = entry.state.write().await;
            *state = ConnectionState::Disconnected;
            info!(server_name = %name, "Disconnected from MCP server");
        }
    }

    async fn start_health_check_task(&self) {
        let servers = self.servers.clone();
        let interval = self.health_check_interval;
        let shutdown = self.shutdown.clone();

        // Reset shutdown flag in case connect_all is called again after disconnect_all
        shutdown.store(false, Ordering::SeqCst);

        let handle = tokio::spawn(async move {
            loop {
                sleep(interval).await;

                if shutdown.load(Ordering::SeqCst) {
                    info!("Health check task shutting down");
                    break;
                }

                for (name, entry) in &servers {
                    let is_healthy = entry.health_check().await;

                    if !is_healthy && matches!(*entry.state.read().await, ConnectionState::Connected) {
                        if entry.config.auto_reconnect {
                            if let Err(e) = entry.reconnect().await {
                                error!("Failed to reconnect to {}: {e}", name);
                            }
                        }
                    }
                }
            }
        });

        *self.health_task.write().await = Some(handle);
    }

    pub async fn get_connection_state(&self, server_name: &str) -> Option<ConnectionState> {
        if let Some(entry) = self.servers.get(server_name) {
            Some(entry.state.read().await.clone())
        } else {
            None
        }
    }

    pub async fn get_all_states(&self) -> HashMap<String, ConnectionState> {
        let mut states = HashMap::new();
        for (name, entry) in &self.servers {
            states.insert(name.clone(), entry.state.read().await.clone());
        }
        states
    }
}

impl Drop for EnhancedMcpHub {
    fn drop(&mut self) {
        // Signal the health check loop to stop when the hub is dropped
        self.shutdown.store(true, Ordering::SeqCst);
    }
}
