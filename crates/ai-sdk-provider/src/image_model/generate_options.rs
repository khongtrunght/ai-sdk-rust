use crate::{SharedHeaders, SharedProviderOptions};

/// Options for image generation
#[derive(Debug, Clone)]
pub struct ImageGenerateOptions {
    /// Prompt for the image generation
    pub prompt: String,

    /// Number of images to generate
    pub n: Option<usize>,

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
    pub seed: Option<i64>,

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

impl ImageGenerateOptions {
    /// Creates a new builder for image generation options.
    pub fn builder() -> ImageGenerateOptionsBuilder {
        ImageGenerateOptionsBuilder::new()
    }
}

/// Builder for `ImageGenerateOptions`.
#[derive(Default)]
pub struct ImageGenerateOptionsBuilder {
    prompt: Option<String>,
    n: Option<usize>,
    size: Option<String>,
    aspect_ratio: Option<String>,
    seed: Option<i64>,
    provider_options: Option<SharedProviderOptions>,
    headers: Option<SharedHeaders>,
}

impl ImageGenerateOptionsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the prompt (Required).
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the number of images (Defaults to 1).
    pub fn n(mut self, n: usize) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets the size using width and height integers.
    /// Formats them as "WIDTHxHEIGHT" automatically.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.size = Some(format!("{}x{}", width, height));
        self
    }

    /// Sets the raw size string (e.g. "1024x1024") if you prefer raw input.
    pub fn size_str(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the aspect ratio using width and height integers.
    /// Formats them as "WIDTH:HEIGHT" automatically.
    pub fn aspect_ratio(mut self, width: u32, height: u32) -> Self {
        self.aspect_ratio = Some(format!("{}:{}", width, height));
        self
    }

    /// Sets the seed.
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Adds provider-specific options.
    pub fn provider_options(mut self, options: SharedProviderOptions) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// Adds custom headers.
    pub fn headers(mut self, headers: SharedHeaders) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Builds the options struct.
    /// Panics or returns Result if `prompt` is missing (Design choice: usually Result is better).
    pub fn build(self) -> Result<ImageGenerateOptions, String> {
        Ok(ImageGenerateOptions {
            prompt: self.prompt.ok_or("Prompt is required")?,
            n: self.n,
            size: self.size,
            aspect_ratio: self.aspect_ratio,
            seed: self.seed,
            provider_options: self.provider_options,
            headers: self.headers,
        })
    }
}
