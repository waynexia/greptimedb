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

//! Utils for page level pruning.

use std::any::Any;
use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use common_telemetry::info;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::datatypes::{DataType as DfDataType, Schema, UInt16Type};
use datafusion::datasource::physical_plan::parquet::page_filter::PagePruningPredicate;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_plan::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_plan::PhysicalExpr;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, Operator};
use datatypes::prelude::{ConcreteDataType, DataType};
use datatypes::value::ValueRef;
use store_api::storage::consts::PRIMARY_KEY_COLUMN_NAME;
use table::predicate::Predicate;

use crate::row_converter::{McmpRowCodec, RowCodec, SortField};
use crate::sst::parquet::format::ReadFormat;

/// []
#[derive(PartialEq, Eq, Clone)]
pub struct DecodePrimaryKey {
    read_format: ReadFormat,
    /// The column to decode, `__primary_key`
    encoded_column: Arc<Column>,
    required_column_name: String,
    /// Relative column index in the primary key
    required_index: usize,
    decoder: McmpRowCodec,
    /// Output datatype in DataFusion
    datatype: DfDataType,
    /// Output datatype in ours
    concrete_datatype: ConcreteDataType,
}

impl DecodePrimaryKey {
    pub(crate) fn new(read_format: ReadFormat, required_column: &str) -> Self {
        let encoded_column = Arc::new(Column::new(
            PRIMARY_KEY_COLUMN_NAME,
            read_format.primary_key_position(),
        ));
        let required_column = read_format
            .metadata()
            .column_by_name(required_column)
            .unwrap(); // safety: it's checked
        let required_column_id = required_column.column_id;
        let relative_index = read_format
            .metadata()
            .primary_key
            .iter()
            .position(|id| *id == required_column_id)
            .unwrap(); // safety: it's checked
        let required_column_type = required_column.column_schema.data_type.clone();
        let required_column_name = required_column.column_schema.name.clone();
        let decoder = McmpRowCodec::new(vec![SortField::new(required_column_type.clone())]);
        Self {
            read_format,
            encoded_column,
            required_column_name,
            required_index: relative_index,
            decoder,
            datatype: required_column_type.as_arrow_type(),
            concrete_datatype: required_column_type,
        }
    }

    fn decode_one(&self, bytes: &[u8]) -> DfResult<ScalarValue> {
        let decoded = self
            .decoder
            .decode(bytes)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        decoded[self.required_index]
            .try_to_scalar_value(&self.concrete_datatype)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

impl PhysicalExpr for DecodePrimaryKey {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _: &Schema) -> DfResult<DfDataType> {
        Ok(self.datatype.clone())
    }

    fn nullable(&self, _: &Schema) -> DfResult<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &common_recordbatch::DfRecordBatch) -> DfResult<ColumnarValue> {
        let encoded_col = self.encoded_column.evaluate(batch)?;

        match encoded_col {
            ColumnarValue::Array(array) => {
                let array = array
                    .as_any()
                    .downcast_ref::<DictionaryArray<UInt16Type>>()
                    .unwrap();
                todo!()
            }
            ColumnarValue::Scalar(ScalarValue::Binary(Some(bytes))) => {
                let scalar_value = self.decode_one(&bytes)?;
                Ok(ColumnarValue::Scalar(scalar_value))
            }
            ColumnarValue::Scalar(ScalarValue::Binary(None)) => Ok(ColumnarValue::Scalar(
                // safety: PK won't have strange types
                ScalarValue::try_from(&self.datatype).unwrap(),
            )),
            _ => Err(DataFusionError::Internal(
                "DecodePrimaryKey requires a binary column".to_string(),
            )),
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.encoded_column.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DfResult<Arc<dyn PhysicalExpr>> {
        let Some(child) = children.get(0) else {
            return Err(DataFusionError::Internal(
                "DecodePrimaryKey requires one child".to_string(),
            ));
        };
        let Some(column) = child.as_any().downcast_ref::<Column>() else {
            return Err(DataFusionError::Internal(
                "DecodePrimaryKey requires a Column expr".to_string(),
            ));
        };

        Ok(Arc::new(Self {
            read_format: self.read_format.clone(),
            encoded_column: Arc::new(column.clone()),
            required_column_name: self.required_column_name.clone(),
            required_index: self.required_index,
            decoder: self.decoder.clone(),
            datatype: self.datatype.clone(),
            concrete_datatype: self.concrete_datatype.clone(),
        }))
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        let mut s = state;
        self.read_format.arrow_schema().hash(&mut s);
        self.encoded_column.hash(&mut s);
        self.datatype.hash(&mut s);
    }
}

impl std::fmt::Debug for DecodePrimaryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodePrimaryKey")
            .field("column", &self.required_column_name)
            .field("child", &self.encoded_column)
            .finish()
    }
}

impl Display for DecodePrimaryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @ {}", self.required_column_name, self.encoded_column)
    }
}

impl PartialEq<dyn Any> for DecodePrimaryKey {
    fn eq(&self, other: &dyn Any) -> bool {
        let other = if other.is::<Arc<dyn PhysicalExpr>>() {
            other
                .downcast_ref::<Arc<dyn PhysicalExpr>>()
                .unwrap()
                .as_any()
        } else if other.is::<Box<dyn PhysicalExpr>>() {
            other
                .downcast_ref::<Box<dyn PhysicalExpr>>()
                .unwrap()
                .as_any()
        } else {
            other
        };
        other
            .downcast_ref::<Self>()
            .map(|x| self == x)
            .unwrap_or(false)
    }
}

/// Build [PagePruningPredicate] that is compatible with the encoded primary key column.
pub struct PagePruningPredicateBuilder;

impl PagePruningPredicateBuilder {
    pub(crate) fn build(
        predicate: Predicate,
        read_format: ReadFormat,
        file_schema: &Arc<Schema>,
    ) -> Option<PagePruningPredicate> {
        let page_filter_exprs = Self::filter_map_physical_expr(predicate, &read_format);
        if page_filter_exprs.is_empty() {
            return None;
        }
        let conjoined = Self::conjoin_exprs(page_filter_exprs);
        PagePruningPredicate::try_new(
            &conjoined,
            // read_format.metadata().schema.arrow_schema().clone(),
            file_schema.clone(),
        )
        .ok()
    }

    /// Only exprs referencing
    /// 1. non-PK columns
    /// 2. the first PK column
    /// are kept.
    ///
    /// Valid primary key exprs will be transformed into [DecodePrimaryKey]. Other non-PK
    /// columns leave as is.
    fn filter_map_physical_expr(
        predicate: Predicate,
        read_format: &ReadFormat,
    ) -> Vec<Arc<dyn PhysicalExpr>> {
        // gather valid column names
        let first_primary_key = read_format
            .metadata()
            .primary_key
            .get(0)
            .and_then(|id| read_format.metadata().column_by_id(*id));
        let valid_set = read_format
            .metadata()
            .field_columns()
            .chain(first_primary_key)
            .chain(Some(read_format.metadata().time_index_column()))
            .map(|c| c.column_schema.name.clone())
            .collect::<HashSet<_>>();

        // transform exprs
        predicate
            .exprs
            .into_iter()
            .filter_map(|e| {
                // e.transform(&|e| {
                //     if let Some(c) = e.as_any().downcast_ref::<Column>() {
                //         if valid_set.contains(c.name()) {
                //             Ok(Self::transform_expr(read_format, c.clone(), e.clone()))
                //         } else {
                //             info!("[DEBUG] not a valid column: {:?}", c);
                //             // Just throw a whatever error to indicate that this column is not valid.
                //             Err(DataFusionError::Internal("not a valid column".to_string()))
                //         }
                //     } else {
                //         Ok(Transformed::No(e))
                //     }
                // })
                // .ok()
                let Some(binary_expr) = e.as_any().downcast_ref::<BinaryExpr>() else {
                    return Some(e);
                };
                // assume col is on the left
                let Some(col) = binary_expr.left().as_any().downcast_ref::<Column>() else {
                    return Some(e);
                };
                // if valid_set.contains(c.name())
                if let Some(first_pk) = first_primary_key && first_pk.column_schema.name == col.name(){
                    // return Some(Self::rewrite_primary_key_eq(&read_format, e.clone()))
                    let transformed = Self::rewrite_primary_key_eq(&read_format, e.clone());
                    if let Transformed::Yes(e) = transformed {
                        return Some(e);
                    } else{
                        return None;
                    }
                }
                Some(e)
            })
            .collect()
    }

    /// Insert [DecodePrimaryKey] into expr if necessary
    ///
    /// This function doesn't check if the given column is a
    /// valid primary key.
    fn transform_expr(
        read_format: &ReadFormat,
        original: Column,
        outermost: Arc<dyn PhysicalExpr>,
    ) -> Transformed<Arc<dyn PhysicalExpr>> {
        // only transform primary keys
        if !read_format.is_tag_column(original.name()) {
            return Transformed::No(Arc::new(original));
        }

        Transformed::Yes(Arc::new(DecodePrimaryKey::new(
            read_format.clone(),
            original.name(),
        )))
    }

    fn rewrite_primary_key_eq(
        read_format: &ReadFormat,
        outermost: Arc<dyn PhysicalExpr>,
    ) -> Transformed<Arc<dyn PhysicalExpr>> {
        let Some(binary_expr) = outermost.as_any().downcast_ref::<BinaryExpr>() else {
            return Transformed::No(outermost);
        };
        // assume literal is on the right
        let Some(lit) = binary_expr.right().as_any().downcast_ref::<Literal>() else {
            return Transformed::No(outermost);
        };
        let ScalarValue::Utf8(Some(lit)) = lit.value() else {
            return Transformed::No(outermost);
        };

        let encoder = McmpRowCodec::new(vec![SortField::new(ConcreteDataType::string_datatype())]);
        let lower_bound = encoder.encode([ValueRef::String(lit)].into_iter()).unwrap();
        let upper_bound = get_prefix_end_key(&lower_bound);

        let pk_col = Arc::new(Column::new(
            PRIMARY_KEY_COLUMN_NAME,
            read_format.primary_key_position(),
        ));
        let transformed = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                pk_col.clone(),
                Operator::GtEq,
                Arc::new(Literal::new(ScalarValue::Binary(Some(lower_bound)))),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                pk_col.clone(),
                Operator::Lt,
                Arc::new(Literal::new(ScalarValue::Binary(Some(upper_bound)))),
            )),
        ));
        Transformed::Yes(transformed)
    }

    /// Conjoin exprs with `AND`
    ///
    /// Caller should ensure input `exprs` are not empty
    fn conjoin_exprs(mut exprs: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn PhysicalExpr> {
        let first_expr = exprs.pop().unwrap();
        if exprs.is_empty() {
            return first_expr;
        }
        exprs.into_iter().fold(first_expr, |acc, e| {
            Arc::new(BinaryExpr::new(acc, Operator::And, e))
        })
    }
}

pub fn get_prefix_end_key(key: &[u8]) -> Vec<u8> {
    for (i, v) in key.iter().enumerate().rev() {
        if *v < 0xFF {
            let mut end = Vec::from(&key[..=i]);
            end[i] = *v + 1;
            return end;
        }
    }

    // next prefix does not exist (e.g., 0xffff);
    vec![0]
}
