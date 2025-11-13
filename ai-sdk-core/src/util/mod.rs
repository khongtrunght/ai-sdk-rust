/// Utilities for the AI SDK Core
/// Base64 encoding and decoding utilities
pub mod base64;
/// File download utilities
pub mod download;
/// Media type detection from binary data
pub mod media_type;

pub use base64::{decode_base64, encode_base64};
pub use download::{download, DownloadError, DownloadedFile};
pub use media_type::detect_media_type;
