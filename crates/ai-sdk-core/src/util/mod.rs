/// Utilities for the AI SDK Core
/// Base64 encoding and decoding utilities
pub mod base64;
/// Deep equality checking for JSON values
pub mod deep_equal;
/// File download utilities
pub mod download;
/// JSON repair utilities for fixing incomplete JSON
pub mod fix_json;
/// Media type detection from binary data
pub mod media_type;
/// Parse partial JSON with automatic repair
pub mod parse_partial_json;

pub use base64::{decode_base64, encode_base64};
pub use deep_equal::is_deep_equal;
pub use download::{download, DownloadError, DownloadedFile};
pub use fix_json::fix_json;
pub use media_type::detect_media_type;
pub use parse_partial_json::{parse_partial_json, ParseResult, ParseState};
