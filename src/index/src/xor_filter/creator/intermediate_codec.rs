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

use asynchronous_codec::{BytesMut, Decoder, Encoder};
use bytes::{Buf, BufMut};
use snafu::{ensure, ResultExt};

use crate::xor_filter::creator::finalize_segment::FinalizedXorFilterSegment;
use crate::xor_filter::error::{
    DeserializeXorFilterSnafu, Error, InvalidIntermediateMagicSnafu, IoSnafu, Result,
    SerializeXorFilterSnafu,
};

/// The magic number for the codec version 1 of the intermediate XOR filter.
const CODEC_V1_MAGIC: &[u8; 4] = b"xi01";

/// Codec of the intermediate finalized XOR filter segment.
///
/// # Format
///
/// [ magic ][ keys count ][    keys size    ][ keys data ][ keys count ][    keys size    ][ keys data ]...
///    [4]       [8]              [8]          [size]          [8]              [8]           [size]
#[derive(Debug, Default)]
pub struct IntermediateXorFilterCodecV1 {
    handled_header_magic: bool,
}

impl Encoder for IntermediateXorFilterCodecV1 {
    type Item<'a> = FinalizedXorFilterSegment;
    type Error = Error;

    fn encode(&mut self, item: FinalizedXorFilterSegment, dst: &mut BytesMut) -> Result<()> {
        if !self.handled_header_magic {
            dst.extend_from_slice(CODEC_V1_MAGIC);
            self.handled_header_magic = true;
        }

        let keys_bytes = bincode::serialize(&item.keys).context(SerializeXorFilterSnafu)?;
        let keys_count = item.keys.len();

        dst.reserve(2 * std::mem::size_of::<u64>() + keys_bytes.len());
        dst.put_u64_le(keys_count as u64);
        dst.put_u64_le(keys_bytes.len() as u64);
        dst.extend_from_slice(&keys_bytes);
        Ok(())
    }
}

impl Decoder for IntermediateXorFilterCodecV1 {
    type Item = FinalizedXorFilterSegment;
    type Error = Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>> {
        if !self.handled_header_magic {
            let m_len = CODEC_V1_MAGIC.len();
            if src.remaining() < m_len {
                return Ok(None);
            }
            let magic_bytes = &src[..m_len];
            ensure!(
                magic_bytes == CODEC_V1_MAGIC,
                InvalidIntermediateMagicSnafu {
                    invalid: magic_bytes,
                }
            );
            self.handled_header_magic = true;
            src.advance(m_len);
        }

        let s = &src[..];

        let u64_size = std::mem::size_of::<u64>();
        let n_size = u64_size * 2;
        if s.len() < n_size {
            return Ok(None);
        }

        let _keys_count = u64::from_le_bytes(s[0..u64_size].try_into().unwrap()) as usize;
        let keys_size = u64::from_le_bytes(s[u64_size..n_size].try_into().unwrap()) as usize;

        if s.len() < n_size + keys_size {
            return Ok(None);
        }

        let keys: Vec<u64> = bincode::deserialize(&s[n_size..n_size + keys_size])
            .context(DeserializeXorFilterSnafu)?;
        src.advance(n_size + keys_size);
        Ok(Some(FinalizedXorFilterSegment { keys }))
    }
}

/// Required for [`Encoder`] and [`Decoder`] implementations.
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Err::<(), std::io::Error>(error)
            .context(IoSnafu)
            .unwrap_err()
    }
}

#[cfg(test)]
mod tests {
    use asynchronous_codec::{FramedRead, FramedWrite};
    use futures::io::Cursor;
    use futures::{SinkExt, StreamExt};

    use super::*;

    #[test]
    fn test_intermediate_xor_filter_codec_v1_basic() {
        let mut encoder = IntermediateXorFilterCodecV1::default();
        let mut buf = BytesMut::new();

        let item1 = FinalizedXorFilterSegment {
            keys: vec![1, 2, 3, 4],
        };
        let item2 = FinalizedXorFilterSegment {
            keys: vec![5, 6, 7, 8],
        };
        let item3 = FinalizedXorFilterSegment {
            keys: vec![9, 10, 11, 12],
        };

        encoder.encode(item1.clone(), &mut buf).unwrap();
        encoder.encode(item2.clone(), &mut buf).unwrap();
        encoder.encode(item3.clone(), &mut buf).unwrap();

        let mut buf = buf.freeze().try_into_mut().unwrap();

        let mut decoder = IntermediateXorFilterCodecV1::default();
        let decoded_item1 = decoder.decode(&mut buf).unwrap().unwrap();
        let decoded_item2 = decoder.decode(&mut buf).unwrap().unwrap();
        let decoded_item3 = decoder.decode(&mut buf).unwrap().unwrap();

        assert_eq!(item1.keys, decoded_item1.keys);
        assert_eq!(item2.keys, decoded_item2.keys);
        assert_eq!(item3.keys, decoded_item3.keys);
    }

    #[tokio::test]
    async fn test_intermediate_xor_filter_codec_v1_frame_read_write() {
        let item1 = FinalizedXorFilterSegment {
            keys: vec![1, 2, 3, 4],
        };
        let item2 = FinalizedXorFilterSegment {
            keys: vec![5, 6, 7, 8],
        };
        let item3 = FinalizedXorFilterSegment {
            keys: vec![9, 10, 11, 12],
        };

        let mut bytes = Cursor::new(vec![]);

        let mut writer = FramedWrite::new(&mut bytes, IntermediateXorFilterCodecV1::default());
        writer.send(item1.clone()).await.unwrap();
        writer.send(item2.clone()).await.unwrap();
        writer.send(item3.clone()).await.unwrap();
        writer.flush().await.unwrap();
        writer.close().await.unwrap();

        let bytes = bytes.into_inner();
        let mut reader = FramedRead::new(bytes.as_slice(), IntermediateXorFilterCodecV1::default());
        let decoded_item1 = reader.next().await.unwrap().unwrap();
        let decoded_item2 = reader.next().await.unwrap().unwrap();
        let decoded_item3 = reader.next().await.unwrap().unwrap();
        assert!(reader.next().await.is_none());

        assert_eq!(item1.keys, decoded_item1.keys);
        assert_eq!(item2.keys, decoded_item2.keys);
        assert_eq!(item3.keys, decoded_item3.keys);
    }

    #[tokio::test]
    async fn test_intermediate_xor_filter_codec_v1_frame_read_write_only_magic() {
        let bytes = CODEC_V1_MAGIC.to_vec();
        let mut reader = FramedRead::new(bytes.as_slice(), IntermediateXorFilterCodecV1::default());
        assert!(reader.next().await.is_none());
    }

    #[tokio::test]
    async fn test_intermediate_xor_filter_codec_v1_frame_read_write_partial_magic() {
        let bytes = CODEC_V1_MAGIC[..3].to_vec();
        let mut reader = FramedRead::new(bytes.as_slice(), IntermediateXorFilterCodecV1::default());
        let e = reader.next().await.unwrap();
        assert!(e.is_err());
    }

    #[tokio::test]
    async fn test_intermediate_xor_filter_codec_v1_frame_read_write_partial_item() {
        let mut bytes = vec![];
        bytes.extend_from_slice(CODEC_V1_MAGIC);
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&8u64.to_le_bytes());

        let mut reader = FramedRead::new(bytes.as_slice(), IntermediateXorFilterCodecV1::default());
        let e = reader.next().await.unwrap();
        assert!(e.is_err());
    }

    #[tokio::test]
    async fn test_intermediate_xor_filter_codec_v1_frame_read_write_corrupted_magic() {
        let mut bytes = vec![];
        bytes.extend_from_slice(b"xi02");
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&8u64.to_le_bytes());

        // We need valid bincode data here, so serialize a real keys vector
        let keys_bytes = bincode::serialize(&vec![1, 2, 3, 4]).unwrap();
        bytes.extend_from_slice(&keys_bytes);

        let mut reader = FramedRead::new(bytes.as_slice(), IntermediateXorFilterCodecV1::default());
        let e = reader.next().await.unwrap();
        assert!(e.is_err());
    }
}
