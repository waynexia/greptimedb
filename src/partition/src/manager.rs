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

use std::collections::HashMap;
use std::sync::Arc;

use api::v1::Rows;
use common_meta::cache::{TableRoute, TableRouteCacheRef};
use common_meta::key::table_info::TableInfoManager;
use common_meta::key::table_route::{PhysicalTableRouteValue, TableRouteManager};
use common_meta::kv_backend::KvBackendRef;
use common_meta::peer::Peer;
use common_meta::rpc::router::{self, RegionRoute};
use snafu::{ensure, OptionExt, ResultExt};
use store_api::metric_engine_consts::LOGICAL_TABLE_METADATA_KEY;
use store_api::storage::{RegionId, RegionNumber};
use table::metadata::{TableId, TableInfo};

use crate::error::{
    FindLeaderSnafu, FindTableRoutesSnafu, Result, TableRouteManagerSnafu, TableRouteNotFoundSnafu,
    UnexpectedSnafu,
};
use crate::expr::PartitionExpr;
use crate::multi_dim::MultiDimPartitionRule;
use crate::splitter::RowSplitter;
use crate::PartitionRuleRef;

#[async_trait::async_trait]
pub trait TableRouteCacheInvalidator: Send + Sync {
    async fn invalidate_table_route(&self, table: TableId);
}

pub type TableRouteCacheInvalidatorRef = Arc<dyn TableRouteCacheInvalidator>;

pub type PartitionRuleManagerRef = Arc<PartitionRuleManager>;

/// PartitionRuleManager manages the table routes and partition rules.
/// It provides methods to find regions by:
/// - values (in case of insertion)
/// - filters (in case of select, deletion and update)
pub struct PartitionRuleManager {
    table_route_manager: TableRouteManager,
    table_route_cache: TableRouteCacheRef,
    table_info_manager: TableInfoManager,
}

#[derive(Debug)]
pub struct PartitionInfo {
    pub id: RegionId,
    pub partition_expr: Option<PartitionExpr>,
}

impl PartitionRuleManager {
    pub fn new(kv_backend: KvBackendRef, table_route_cache: TableRouteCacheRef) -> Self {
        Self {
            table_route_manager: TableRouteManager::new(kv_backend.clone()),
            table_route_cache,
            table_info_manager: TableInfoManager::new(kv_backend),
        }
    }

    pub async fn find_physical_table_route(
        &self,
        table_id: TableId,
    ) -> Result<Arc<PhysicalTableRouteValue>> {
        match self
            .table_route_cache
            .get(table_id)
            .await
            .context(TableRouteManagerSnafu)?
            .context(TableRouteNotFoundSnafu { table_id })?
            .as_ref()
        {
            TableRoute::Physical(physical_table_route) => Ok(physical_table_route.clone()),
            TableRoute::Logical(logical_table_route) => {
                let physical_table_id = logical_table_route.physical_table_id();
                let physical_table_route = self
                    .table_route_cache
                    .get(physical_table_id)
                    .await
                    .context(TableRouteManagerSnafu)?
                    .context(TableRouteNotFoundSnafu { table_id })?;

                let physical_table_route = physical_table_route
                    .as_physical_table_route_ref()
                    .context(UnexpectedSnafu{
                        err_msg: format!(
                            "Expected the physical table route, but got logical table route, table: {table_id}"
                        ),
                    })?;

                Ok(physical_table_route.clone())
            }
        }
    }

    pub async fn batch_find_region_routes(
        &self,
        table_ids: &[TableId],
    ) -> Result<HashMap<TableId, Vec<RegionRoute>>> {
        let table_routes = self
            .table_route_manager
            .batch_get_physical_table_routes(table_ids)
            .await
            .context(TableRouteManagerSnafu)?;

        let mut table_region_routes = HashMap::with_capacity(table_routes.len());

        for (table_id, table_route) in table_routes {
            let region_routes = table_route.region_routes;
            table_region_routes.insert(table_id, region_routes);
        }

        Ok(table_region_routes)
    }

    /// Fetch partitions for a single table.
    ///
    /// Will return physical table partitions for logical tables.
    pub async fn find_table_partitions(&self, table_id: TableId) -> Result<Vec<PartitionInfo>> {
        let region_routes = &self
            .find_physical_table_route(table_id)
            .await?
            .region_routes;
        ensure!(!region_routes.is_empty(), FindTableRoutesSnafu { table_id });

        create_partitions_from_region_routes(table_id, region_routes)
    }

    /// Fetch partitions for multiple tables in one batch.
    ///
    /// Will return physical table partitions for logical tables.
    pub async fn batch_find_table_partitions(
        &self,
        table_ids: &[TableId],
    ) -> Result<HashMap<TableId, Vec<PartitionInfo>>> {
        let batch_region_routes = self.batch_find_region_routes(table_ids).await?;
        let mut results = HashMap::with_capacity(batch_region_routes.len());

        for (table_id, region_routes) in batch_region_routes {
            let partitions = create_partitions_from_region_routes(table_id, &region_routes)?;
            results.insert(table_id, partitions);
        }

        Ok(results)
    }

    pub async fn find_table_partition_rule(
        &self,
        table_info: &TableInfo,
    ) -> Result<PartitionRuleRef> {
        let (partitions, partition_columns) = if self.is_logical_table(table_info) {
            self.get_physical_table_partition_data(table_info).await?
        } else {
            self.get_table_partition_data(table_info).await?
        };

        let regions = Self::extract_regions(&partitions);
        let exprs = Self::extract_partition_expressions(&partitions);

        let rule = MultiDimPartitionRule::try_new(partition_columns, regions, exprs, false)?;
        Ok(Arc::new(rule) as _)
    }

    /// Check if a table is logical by looking for the logical table metadata key
    fn is_logical_table(&self, table_info: &TableInfo) -> bool {
        table_info
            .meta
            .options
            .extra_options
            .contains_key(LOGICAL_TABLE_METADATA_KEY)
    }

    /// Get partition data (partitions and columns) for a physical table
    async fn get_table_partition_data(
        &self,
        table_info: &TableInfo,
    ) -> Result<(Vec<PartitionInfo>, Vec<String>)> {
        let partitions = self.find_table_partitions(table_info.table_id()).await?;
        let partition_columns = table_info
            .meta
            .partition_column_names()
            .cloned()
            .collect::<Vec<_>>();

        Ok((partitions, partition_columns))
    }

    /// Get partition data from the physical table for a logical table
    async fn get_physical_table_partition_data(
        &self,
        logical_table_info: &TableInfo,
    ) -> Result<(Vec<PartitionInfo>, Vec<String>)> {
        let physical_table_id = self.get_physical_table_id(logical_table_info).await?;

        // Get physical table's partitions
        let physical_partitions = self.find_table_partitions(physical_table_id).await?;

        // Get physical table's partition columns
        let physical_table_info = self.get_table_info(physical_table_id).await?;
        let partition_columns = physical_table_info
            .meta
            .partition_column_names()
            .cloned()
            .collect::<Vec<_>>();

        Ok((physical_partitions, partition_columns))
    }

    /// Get the physical table ID for a logical table
    async fn get_physical_table_id(&self, logical_table_info: &TableInfo) -> Result<TableId> {
        let table_route = self
            .table_route_cache
            .get(logical_table_info.table_id())
            .await
            .context(TableRouteManagerSnafu)?
            .context(TableRouteNotFoundSnafu {
                table_id: logical_table_info.table_id(),
            })?;

        match table_route.as_ref() {
            TableRoute::Logical(logical_route) => Ok(logical_route.physical_table_id()),
            _ => Err(UnexpectedSnafu {
                err_msg: format!(
                    "Expected logical table route for table {}",
                    logical_table_info.table_id()
                ),
            }
            .build()),
        }
    }

    /// Get TableInfo for a given table ID
    async fn get_table_info(&self, table_id: TableId) -> Result<TableInfo> {
        let table_info_value = self
            .table_info_manager
            .get(table_id)
            .await
            .context(TableRouteManagerSnafu)?
            .context(TableRouteNotFoundSnafu { table_id })?;

        TableInfo::try_from(table_info_value.table_info.clone()).map_err(|e| {
            UnexpectedSnafu {
                err_msg: format!("Failed to convert RawTableInfo to TableInfo: {}", e),
            }
            .build()
        })
    }

    /// Extract region numbers from partitions
    fn extract_regions(partitions: &[PartitionInfo]) -> Vec<RegionNumber> {
        partitions.iter().map(|x| x.id.region_number()).collect()
    }

    /// Extract partition expressions from partitions
    fn extract_partition_expressions(partitions: &[PartitionInfo]) -> Vec<PartitionExpr> {
        partitions
            .iter()
            .filter_map(|x| x.partition_expr.as_ref())
            .cloned()
            .collect()
    }

    /// Find the leader of the region.
    pub async fn find_region_leader(&self, region_id: RegionId) -> Result<Peer> {
        let region_routes = &self
            .find_physical_table_route(region_id.table_id())
            .await?
            .region_routes;

        router::find_region_leader(region_routes, region_id.region_number()).context(
            FindLeaderSnafu {
                region_id,
                table_id: region_id.table_id(),
            },
        )
    }

    pub async fn split_rows(
        &self,
        table_info: &TableInfo,
        rows: Rows,
    ) -> Result<HashMap<RegionNumber, Rows>> {
        let partition_rule = self.find_table_partition_rule(table_info).await?;
        RowSplitter::new(partition_rule).split(rows)
    }
}

fn create_partitions_from_region_routes(
    table_id: TableId,
    region_routes: &[RegionRoute],
) -> Result<Vec<PartitionInfo>> {
    let mut partitions = Vec::with_capacity(region_routes.len());
    for r in region_routes {
        let partition_expr = PartitionExpr::from_json_str(&r.region.partition_expr())?;

        // The region routes belong to the physical table but are shared among all logical tables.
        // That it to say, the region id points to the physical table, so we need to use the actual
        // table id (which may be a logical table) to renew the region id.
        let id = RegionId::new(table_id, r.region.id.region_number());
        partitions.push(PartitionInfo { id, partition_expr });
    }

    Ok(partitions)
}
