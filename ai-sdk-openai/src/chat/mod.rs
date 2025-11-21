mod model;
mod options;

pub use model::OpenAIChatModel;

use std::sync::atomic::{AtomicU64, Ordering};

// Simple ID generator for source parts
static SOURCE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn generate_source_id() -> String {
    let id = SOURCE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("source-{}", id)
}
