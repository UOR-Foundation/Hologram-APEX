# hologram-config Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

`hologram-config` provides configuration management for Hologram, supporting environment variables, TOML configuration files, and runtime configuration.

## Purpose

Core responsibilities:
- `.env` file support (environment variables)
- TOML configuration file parsing
- Runtime configuration management
- Environment variable overrides
- Backend selection configuration
- Feature flags and settings

## Public API

### Config Types

```rust
/// Main configuration
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Backend configuration
    pub backend: BackendConfig,

    /// Memory configuration
    pub memory: MemoryConfig,

    /// Execution configuration
    pub execution: ExecutionConfig,

    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Config {
    /// Load from environment variables
    pub fn from_env() -> Result<Self>;

    /// Load from TOML file
    pub fn from_file(path: &Path) -> Result<Self>;

    /// Load with environment overrides
    pub fn load() -> Result<Self>;

    /// Create default configuration
    pub fn default() -> Self;
}
```

### Backend Configuration

```rust
/// Backend selection and settings
#[derive(Debug, Clone, PartialEq)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,

    /// Device ID (for GPU backends)
    pub device_id: Option<u32>,

    /// Enable backend-specific optimizations
    pub optimizations: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    Cuda,
    Metal,
    Wasm,
    WebGpu,
}
```

### Memory Configuration

```rust
/// Memory management settings
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConfig {
    /// Maximum buffer size (bytes)
    pub max_buffer_size: usize,

    /// Buffer pool size
    pub buffer_pool_size: usize,

    /// Enable memory pooling
    pub enable_pooling: bool,

    /// Pool storage capacity (for streaming)
    pub pool_capacity: usize,
}
```

### Execution Configuration

```rust
/// Execution settings
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionConfig {
    /// Number of threads (CPU backend)
    pub num_threads: Option<usize>,

    /// Enable parallel execution
    pub parallel: bool,

    /// Grid/block dimensions
    pub launch_config: Option<LaunchConfigTemplate>,

    /// Timeout (milliseconds)
    pub timeout_ms: Option<u64>,
}
```

### Logging Configuration

```rust
/// Logging settings
#[derive(Debug, Clone, PartialEq)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,

    /// Enable profiling
    pub profiling: bool,

    /// Log file path
    pub log_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}
```

## Environment Variables

### Standard Variables

```bash
# Backend selection
HOLOGRAM_BACKEND=cpu|cuda|metal|wasm|webgpu

# Device selection (GPU backends)
HOLOGRAM_DEVICE_ID=0

# Memory configuration
HOLOGRAM_MAX_BUFFER_SIZE=1073741824  # 1GB
HOLOGRAM_BUFFER_POOL_SIZE=1024
HOLOGRAM_POOL_CAPACITY=4096

# Execution configuration
HOLOGRAM_NUM_THREADS=8
HOLOGRAM_PARALLEL=true
HOLOGRAM_TIMEOUT_MS=30000

# Logging
HOLOGRAM_LOG_LEVEL=info|debug|trace|warn|error
HOLOGRAM_PROFILING=true
HOLOGRAM_LOG_FILE=/path/to/log.txt

# Feature flags
HOLOGRAM_ENABLE_SIMD=true
HOLOGRAM_ENABLE_CACHE=true
```

## TOML Configuration

### File Format

**Default location:** `./hologram.toml` or `~/.config/hologram/config.toml`

```toml
# hologram.toml

[backend]
type = "cpu"  # cpu|cuda|metal|wasm|webgpu
device_id = 0
optimizations = true

[memory]
max_buffer_size = 1073741824  # 1GB
buffer_pool_size = 1024
enable_pooling = true
pool_capacity = 4096

[execution]
num_threads = 8
parallel = true
timeout_ms = 30000

[execution.launch_config]
grid = [16, 16, 1]
block = [32, 32, 1]

[logging]
level = "info"
profiling = false
log_file = "/var/log/hologram.log"
```

## Internal Structure

```
crates/config/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── env.rs              # Environment variable parsing
│   ├── file.rs             # TOML file parsing
│   ├── runtime.rs          # Runtime configuration
│   ├── types.rs            # Configuration types
│   └── error.rs            # Error types
└── tests/
    ├── env_tests.rs
    ├── file_tests.rs
    └── integration_tests.rs
```

## Dependencies

```toml
[dependencies]
# TOML parsing
toml = "0.8"

# Environment variables
dotenvy = "0.15"

# Serialization
serde = { version = "1.0", features = ["derive"] }

# Error handling
thiserror = "1.0"

# Path handling
dirs = "5.0"

# Hot-reload support
notify = "6.1"

# System information
num_cpus = "1.16"

# Async runtime (for config watching)
parking_lot = "0.12"
```

## Implementation Details

### Loading Priority

1. **Defaults** - Hardcoded defaults
2. **Config file** - TOML file (if exists)
3. **Environment** - Environment variables (override config file)
4. **Runtime** - Programmatic overrides (highest priority)

```rust
impl Config {
    pub fn load() -> Result<Self> {
        // 1. Start with defaults
        let mut config = Config::default();

        // 2. Load from file (if exists)
        if let Ok(file_config) = Self::from_file_checked() {
            config.merge(file_config);
        }

        // 3. Override with environment
        config.merge_env()?;

        Ok(config)
    }
}
```

### Environment Parsing

```rust
pub fn parse_env() -> Result<EnvConfig> {
    // Load .env file
    dotenvy::dotenv().ok();

    let backend = env::var("HOLOGRAM_BACKEND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(BackendType::Cpu);

    let device_id = env::var("HOLOGRAM_DEVICE_ID")
        .ok()
        .and_then(|s| s.parse().ok());

    // ... parse remaining variables

    Ok(EnvConfig {
        backend,
        device_id,
        // ...
    })
}
```

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_default_config() {
    let config = Config::default();
    assert_eq!(config.backend.backend_type, BackendType::Cpu);
    assert!(config.execution.parallel);
}

#[test]
fn test_env_override() {
    std::env::set_var("HOLOGRAM_BACKEND", "cuda");
    std::env::set_var("HOLOGRAM_DEVICE_ID", "1");

    let config = Config::from_env().unwrap();
    assert_eq!(config.backend.backend_type, BackendType::Cuda);
    assert_eq!(config.backend.device_id, Some(1));
}

#[test]
fn test_toml_parsing() {
    let toml = r#"
        [backend]
        type = "metal"
        device_id = 0
    "#;

    let config: Config = toml::from_str(toml).unwrap();
    assert_eq!(config.backend.backend_type, BackendType::Metal);
}
```

## Examples

### Basic Usage

```rust
use hologram_config::Config;

// Load configuration (file + env)
let config = Config::load()?;

// Use configuration
println!("Using backend: {:?}", config.backend.backend_type);
```

### Programmatic Configuration

```rust
use hologram_config::{Config, BackendConfig, BackendType};

let mut config = Config::default();
config.backend = BackendConfig {
    backend_type: BackendType::Cuda,
    device_id: Some(1),
    optimizations: true,
};
```

### Custom TOML Path

```rust
use hologram_config::Config;
use std::path::Path;

let config = Config::from_file(Path::new("/etc/hologram/config.toml"))?;
```

## Error Handling

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to load config file: {0}")]
    FileLoadError(#[from] std::io::Error),

    #[error("Failed to parse TOML: {0}")]
    TomlParseError(#[from] toml::de::Error),

    #[error("Invalid backend type: {0}")]
    InvalidBackend(String),

    #[error("Environment variable error: {0}")]
    EnvError(#[from] std::env::VarError),
}

pub type Result<T> = std::result::Result<T, ConfigError>;
```

## Configuration Validation

### Macro-Based Validation

```rust
/// Macro for defining validated configuration fields
macro_rules! validated_config {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident: $ty:ty => $validator:expr
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $name {
            $($(#[$field_meta])* pub $field: $ty,)*
        }

        impl $name {
            pub fn validate(&self) -> Result<()> {
                $(
                    ($validator)(&self.$field)
                        .map_err(|e| ConfigError::ValidationError(
                            stringify!($field).to_string(),
                            e.to_string()
                        ))?;
                )*
                Ok(())
            }
        }
    };
}

// Usage example
validated_config! {
    #[derive(Debug, Clone)]
    pub struct MemoryConfig {
        max_buffer_size: usize => |v: &usize| {
            if *v > 0 { Ok(()) } else { Err("must be > 0".to_string()) }
        },
        buffer_pool_size: usize => |v: &usize| {
            if *v > 0 && *v <= 65536 {
                Ok(())
            } else {
                Err("must be in range [1, 65536]".to_string())
            }
        },
        enable_pooling: bool => |_: &bool| Ok(()),
        pool_capacity: usize => |v: &usize| {
            if *v.is_power_of_two() {
                Ok(())
            } else {
                Err("must be power of 2".to_string())
            }
        },
    }
}
```

### Validation Rules

```rust
impl Config {
    /// Validate entire configuration
    pub fn validate(&self) -> Result<()> {
        // Backend validation
        match self.backend.backend_type {
            BackendType::Cuda | BackendType::Metal => {
                if self.backend.device_id.is_none() {
                    return Err(ConfigError::ValidationError(
                        "device_id".to_string(),
                        "GPU backends require device_id".to_string()
                    ));
                }
            }
            _ => {}
        }

        // Memory validation
        self.memory.validate()?;

        // Execution validation
        if let Some(threads) = self.execution.num_threads {
            if threads == 0 || threads > num_cpus::get() * 2 {
                return Err(ConfigError::ValidationError(
                    "num_threads".to_string(),
                    format!("must be in range [1, {}]", num_cpus::get() * 2)
                ));
            }
        }

        Ok(())
    }
}
```

## Configuration Profiles

### Profile System

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Development,
    Production,
    Testing,
}

impl Config {
    /// Load configuration for specific profile
    pub fn for_profile(profile: Profile) -> Result<Self> {
        let config_file = match profile {
            Profile::Development => "hologram.dev.toml",
            Profile::Production => "hologram.prod.toml",
            Profile::Testing => "hologram.test.toml",
        };

        let mut config = if Path::new(config_file).exists() {
            Self::from_file(Path::new(config_file))?
        } else {
            Self::profile_defaults(profile)
        };

        // Override with environment
        config.merge_env()?;
        config.validate()?;

        Ok(config)
    }

    fn profile_defaults(profile: Profile) -> Self {
        match profile {
            Profile::Development => Self {
                backend: BackendConfig {
                    backend_type: BackendType::Cpu,
                    device_id: None,
                    optimizations: false,
                },
                memory: MemoryConfig {
                    max_buffer_size: 100 * 1024 * 1024, // 100MB
                    buffer_pool_size: 256,
                    enable_pooling: true,
                    pool_capacity: 1024,
                },
                execution: ExecutionConfig {
                    num_threads: Some(1),
                    parallel: false,
                    launch_config: None,
                    timeout_ms: Some(10000),
                },
                logging: LoggingConfig {
                    level: LogLevel::Debug,
                    profiling: true,
                    log_file: None,
                },
            },
            Profile::Production => Self {
                backend: BackendConfig {
                    backend_type: BackendType::Cuda,
                    device_id: Some(0),
                    optimizations: true,
                },
                memory: MemoryConfig {
                    max_buffer_size: 8 * 1024 * 1024 * 1024, // 8GB
                    buffer_pool_size: 4096,
                    enable_pooling: true,
                    pool_capacity: 16384,
                },
                execution: ExecutionConfig {
                    num_threads: Some(num_cpus::get()),
                    parallel: true,
                    launch_config: None,
                    timeout_ms: None,
                },
                logging: LoggingConfig {
                    level: LogLevel::Info,
                    profiling: false,
                    log_file: Some(PathBuf::from("/var/log/hologram.log")),
                },
            },
            Profile::Testing => Self {
                backend: BackendConfig {
                    backend_type: BackendType::Cpu,
                    device_id: None,
                    optimizations: false,
                },
                memory: MemoryConfig {
                    max_buffer_size: 10 * 1024 * 1024, // 10MB
                    buffer_pool_size: 128,
                    enable_pooling: true,
                    pool_capacity: 512,
                },
                execution: ExecutionConfig {
                    num_threads: Some(1),
                    parallel: false,
                    launch_config: None,
                    timeout_ms: Some(5000),
                },
                logging: LoggingConfig {
                    level: LogLevel::Trace,
                    profiling: false,
                    log_file: None,
                },
            },
        }
    }
}
```

### Profile Selection

```bash
# Via environment variable
export HOLOGRAM_PROFILE=production

# Via command line
hologram-compile --profile production kernel.py
```

## Hot-Reload Configuration

### Watch-Based Reload

```rust
use notify::{Watcher, RecursiveMode, Event};

pub struct ConfigWatcher {
    config: Arc<RwLock<Config>>,
    watcher: RecommendedWatcher,
}

impl ConfigWatcher {
    pub fn new(config_path: impl AsRef<Path>) -> Result<Self> {
        let config = Arc::new(RwLock::new(Config::from_file(config_path.as_ref())?));
        let config_clone = config.clone();
        let path = config_path.as_ref().to_path_buf();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                if event.kind.is_modify() {
                    if let Ok(new_config) = Config::from_file(&path) {
                        if let Ok(mut guard) = config_clone.write() {
                            *guard = new_config;
                            tracing::info!("Configuration reloaded");
                        }
                    }
                }
            }
        })?;

        watcher.watch(config_path.as_ref(), RecursiveMode::NonRecursive)?;

        Ok(Self { config, watcher })
    }

    pub fn get(&self) -> Config {
        self.config.read().unwrap().clone()
    }
}
```

### Usage

```rust
let watcher = ConfigWatcher::new("hologram.toml")?;

// Configuration automatically reloads on file changes
loop {
    let config = watcher.get();
    // Use latest configuration
}
```

## Migration from Current Codebase

Port any existing configuration code from `hologram-core` or other crates to this centralized config crate.
