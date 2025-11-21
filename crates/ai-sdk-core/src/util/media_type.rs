/// Media type detection from binary data using magic bytes (file signatures)
/// Common file signatures for media type detection
const IMAGE_SIGNATURES: &[(&[u8], &str)] = &[
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xFF\xD8\xFF", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"), // WebP starts with RIFF, need further check
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"), // Little endian TIFF
    (b"MM\x00*", "image/tiff"), // Big endian TIFF
    // AVIF detection requires checking ftyp box
    (b"\x00\x00\x00\x1cftypavif", "image/avif"),
    (b"\x00\x00\x00\x18ftypavif", "image/avif"),
    (b"\x00\x00\x00\x20ftypavif", "image/avif"),
    // HEIC detection
    (b"\x00\x00\x00\x1cftypheic", "image/heic"),
    (b"\x00\x00\x00\x18ftypheic", "image/heic"),
    (b"\x00\x00\x00\x20ftypheic", "image/heic"),
];

const AUDIO_SIGNATURES: &[(&[u8], &str)] = &[
    (b"ID3", "audio/mpeg"),      // MP3 with ID3 tag
    (b"\xFF\xFB", "audio/mpeg"), // MP3 frame sync
    (b"\xFF\xF3", "audio/mpeg"), // MP3 frame sync
    (b"\xFF\xF2", "audio/mpeg"), // MP3 frame sync
    (b"RIFF", "audio/wav"),      // WAV (needs further check for WAVE)
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"\xFF\xF1", "audio/aac"), // AAC ADTS
    (b"\xFF\xF9", "audio/aac"), // AAC ADTS
];

const VIDEO_SIGNATURES: &[(&[u8], &str)] = &[
    (b"\x00\x00\x00\x1cftyp", "video/mp4"), // MP4
    (b"\x00\x00\x00\x18ftyp", "video/mp4"),
    (b"\x00\x00\x00\x20ftyp", "video/mp4"),
    (b"\x1A\x45\xDF\xA3", "video/webm"), // WebM/Matroska
];

/// Detects the media type of binary data by checking magic bytes
///
/// # Arguments
/// * `data` - The binary data to analyze
///
/// # Returns
/// The IANA media type string if detected, or `None` if the type cannot be determined
///
/// # Example
/// ```
/// use ai_sdk_core::util::detect_media_type;
///
/// let png_data = b"\x89PNG\r\n\x1a\n...";
/// assert_eq!(detect_media_type(png_data), Some("image/png".to_string()));
/// ```
pub fn detect_media_type(data: &[u8]) -> Option<String> {
    if data.is_empty() {
        return None;
    }

    // Check image signatures
    for (signature, media_type) in IMAGE_SIGNATURES {
        if data.starts_with(signature) {
            // Special handling for WebP - need to check for "WEBP" after RIFF
            if *media_type == "image/webp" && data.len() >= 12 {
                if &data[8..12] == b"WEBP" {
                    return Some(media_type.to_string());
                }
                continue; // Not WebP, might be WAV
            }
            return Some(media_type.to_string());
        }
    }

    // Check audio signatures
    for (signature, media_type) in AUDIO_SIGNATURES {
        if data.starts_with(signature) {
            // Special handling for WAV - need to check for "WAVE" after RIFF
            if *media_type == "audio/wav" && data.len() >= 12 {
                if &data[8..12] == b"WAVE" {
                    return Some(media_type.to_string());
                }
                continue; // Not WAV, might be WebP or AVI
            }
            return Some(media_type.to_string());
        }
    }

    // Check video signatures
    for (signature, media_type) in VIDEO_SIGNATURES {
        if data.starts_with(signature) {
            return Some(media_type.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_png() {
        let png_header = b"\x89PNG\r\n\x1a\n";
        assert_eq!(detect_media_type(png_header), Some("image/png".to_string()));
    }

    #[test]
    fn test_detect_jpeg() {
        let jpeg_header = b"\xFF\xD8\xFF\xE0\x00\x10JFIF";
        assert_eq!(
            detect_media_type(jpeg_header),
            Some("image/jpeg".to_string())
        );
    }

    #[test]
    fn test_detect_gif() {
        let gif_header = b"GIF89a";
        assert_eq!(detect_media_type(gif_header), Some("image/gif".to_string()));
    }

    #[test]
    fn test_detect_webp() {
        let webp_header = b"RIFF\x00\x00\x00\x00WEBP";
        assert_eq!(
            detect_media_type(webp_header),
            Some("image/webp".to_string())
        );
    }

    #[test]
    fn test_detect_wav() {
        let wav_header = b"RIFF\x00\x00\x00\x00WAVE";
        assert_eq!(detect_media_type(wav_header), Some("audio/wav".to_string()));
    }

    #[test]
    fn test_detect_mp3() {
        let mp3_header = b"ID3\x03\x00\x00\x00";
        assert_eq!(
            detect_media_type(mp3_header),
            Some("audio/mpeg".to_string())
        );
    }

    #[test]
    fn test_detect_flac() {
        let flac_header = b"fLaC\x00\x00\x00\x22";
        assert_eq!(
            detect_media_type(flac_header),
            Some("audio/flac".to_string())
        );
    }

    #[test]
    fn test_detect_unknown() {
        let unknown_data = b"random data";
        assert_eq!(detect_media_type(unknown_data), None);
    }

    #[test]
    fn test_detect_empty() {
        assert_eq!(detect_media_type(&[]), None);
    }

    #[test]
    fn test_detect_mp4() {
        let mp4_header = b"\x00\x00\x00\x20ftypmp42";
        assert_eq!(detect_media_type(mp4_header), Some("video/mp4".to_string()));
    }

    #[test]
    fn test_detect_bmp() {
        let bmp_header = b"BM";
        assert_eq!(detect_media_type(bmp_header), Some("image/bmp".to_string()));
    }
}
