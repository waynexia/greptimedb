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

use std::any::Any;

use common_config::wal::KafkaWalTopic;
use common_error::ext::ErrorExt;
use common_macro::stack_trace_debug;
use common_runtime::error::Error as RuntimeError;
use snafu::{Location, Snafu};

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[stack_trace_debug]
pub enum Error {
    #[snafu(display("Failed to start log store gc task"))]
    StartGcTask {
        location: Location,
        source: RuntimeError,
    },

    #[snafu(display("Failed to stop log store gc task"))]
    StopGcTask {
        location: Location,
        source: RuntimeError,
    },

    #[snafu(display("Failed to add entry to LogBatch"))]
    AddEntryLogBatch {
        #[snafu(source)]
        error: raft_engine::Error,
        location: Location,
    },

    #[snafu(display("Failed to perform raft-engine operation"))]
    RaftEngine {
        #[snafu(source)]
        error: raft_engine::Error,
        location: Location,
    },

    #[snafu(display("Log store not started yet"))]
    IllegalState { location: Location },

    #[snafu(display("Namespace is illegal: {}", ns))]
    IllegalNamespace { ns: u64, location: Location },

    #[snafu(display(
        "Failed to fetch entries from namespace: {}, start: {}, end: {}, max size: {}",
        ns,
        start,
        end,
        max_size,
    ))]
    FetchEntry {
        ns: u64,
        start: u64,
        end: u64,
        max_size: usize,
        #[snafu(source)]
        error: raft_engine::Error,
        location: Location,
    },

    #[snafu(display(
        "Cannot override compacted entry, namespace: {}, first index: {}, attempt index: {}",
        namespace,
        first_index,
        attempt_index
    ))]
    OverrideCompactedEntry {
        namespace: u64,
        first_index: u64,
        attempt_index: u64,
        location: Location,
    },

    #[snafu(display(
        "Failed to build a Kafka client, broker endpoints: {:?}",
        broker_endpoints
    ))]
    BuildClient {
        broker_endpoints: Vec<String>,
        location: Location,
        #[snafu(source)]
        error: rskafka::client::error::Error,
    },

    #[snafu(display(
        "Failed to build a Kafka partition client, topic: {}, partition: {}",
        topic,
        partition
    ))]
    BuildPartitionClient {
        topic: String,
        partition: i32,
        location: Location,
        #[snafu(source)]
        error: rskafka::client::error::Error,
    },

    #[snafu(display(
        "Failed to get a Kafka topic client, topic: {}, source: {}",
        topic,
        error
    ))]
    GetClient {
        topic: KafkaWalTopic,
        location: Location,
        error: String,
    },

    #[snafu(display("Failed to encode a record meta"))]
    EncodeMeta {
        location: Location,
        #[snafu(source)]
        error: serde_json::Error,
    },

    #[snafu(display("Failed to decode a record meta"))]
    DecodeMeta {
        location: Location,
        #[snafu(source)]
        error: serde_json::Error,
    },

    #[snafu(display("Missing required key in a record"))]
    MissingKey { location: Location },

    #[snafu(display("Missing required value in a record"))]
    MissingValue { location: Location },

    #[snafu(display("Cannot build a record from empty entries"))]
    EmptyEntries { location: Location },

    #[snafu(display("Failed to produce records to Kafka, topic: {}", topic))]
    ProduceRecord {
        topic: KafkaWalTopic,
        location: Location,
        #[snafu(source)]
        error: rskafka::client::producer::Error,
    },

    #[snafu(display(
        "Failed to read a record from Kafka, topic: {}, region_id: {}, offset: {}",
        topic,
        region_id,
        offset,
    ))]
    ConsumeRecord {
        topic: String,
        region_id: u64,
        offset: i64,
        location: Location,
        #[snafu(source)]
        error: rskafka::client::error::Error,
    },

    #[snafu(display("Failed to do a cast"))]
    Cast { location: Location },
}

impl ErrorExt for Error {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;
