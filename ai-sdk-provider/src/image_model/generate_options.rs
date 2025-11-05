use crate::{SharedHeaders, SharedProviderOptions};

/// Options for image generation
#[derive(Debug, Clone)]
pub struct ImageGenerateOptions {
    /// Prompt for the image generation
    pub prompt: String,

    /// Number of images to generate
    pub n: usize,

    /// Size of the images to generate.
    /// Must have the format `{width}x{height}`.
    /// `None` will use the provider's default size.
    pub size: Option<String>,

    /// Aspect ratio of the images to generate.
    /// Must have the format `{width}:{height}`.
    /// `None` will use the provider's default aspect ratio.
    pub aspect_ratio: Option<String>,

    /// Seed for the image generation.
    /// `None` will use the provider's default seed.
    pub seed: Option<u32>,

    /// Additional provider-specific options that are passed through to the provider
    /// as body parameters.
    ///
    /// The outer record is keyed by the provider name, and the inner
    /// record is keyed by the provider-specific metadata key.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "openai": {
    ///     "style": "vivid"
    ///   }
    /// }
    /// ```
    pub provider_options: Option<SharedProviderOptions>,

    /// Additional HTTP headers to be sent with the request.
    /// Only applicable for HTTP-based providers.
    pub headers: Option<SharedHeaders>,
}
