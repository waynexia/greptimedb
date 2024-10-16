// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod topic;
pub mod topic_manager;
pub mod topic_selector;

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub use crate::wal::kafka::topic::Topic;
pub use crate::wal::kafka::topic_manager::TopicManager;
use crate::wal::kafka::topic_selector::SelectorType as TopicSelectorType;

/// Configurations for kafka wal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KafkaConfig {
    /// The broker endpoints of the Kafka cluster.
    pub broker_endpoints: Vec<String>,
    /// Number of topics to be created upon start.
    pub num_topics: usize,
    /// The type of the topic selector with which to select a topic for a region.
    pub selector_type: TopicSelectorType,
    /// Topic name prefix.
    pub topic_name_prefix: String,
    /// Number of partitions per topic.
    pub num_partitions: i32,
    /// The replication factor of each topic.
    pub replication_factor: i16,
    /// Above which a topic creation operation will be cancelled.
    #[serde(with = "humantime_serde")]
    pub create_topic_timeout: Duration,
    /// The initial backoff for kafka clients.
    #[serde(with = "humantime_serde")]
    pub backoff_init: Duration,
    /// The maximum backoff for kafka clients.
    #[serde(with = "humantime_serde")]
    pub backoff_max: Duration,
    /// Exponential backoff rate, i.e. next backoff = base * current backoff.
    // Sets to u32 type since the `backoff_base` field in the KafkaConfig for datanode is of type u32,
    // and we want to unify their types.
    pub backoff_base: u32,
    /// Stop reconnecting if the total wait time reaches the deadline.
    /// If it's None, the reconnecting won't terminate.
    #[serde(with = "humantime_serde")]
    pub backoff_deadline: Option<Duration>,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            broker_endpoints: vec!["127.0.0.1:9090".to_string()],
            num_topics: 64,
            selector_type: TopicSelectorType::RoundRobin,
            topic_name_prefix: "greptimedb_wal_topic".to_string(),
            num_partitions: 1,
            replication_factor: 3,
            create_topic_timeout: Duration::from_secs(30),
            backoff_init: Duration::from_millis(500),
            backoff_max: Duration::from_secs(10),
            backoff_base: 2,
            backoff_deadline: Some(Duration::from_secs(60 * 5)), // 5 mins
        }
    }
}
