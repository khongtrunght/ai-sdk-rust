use backoff::{backoff::Backoff, ExponentialBackoff};
use std::time::Duration;

/// Retry policy for API calls
#[derive(Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(32),
        }
    }
}

impl RetryPolicy {
    /// Creates a new RetryPolicy with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of retry attempts
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the initial delay before first retry
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Sets the maximum delay between retries
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Executes the given function with retry logic using exponential backoff
    pub async fn retry<F, Fut, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut backoff = ExponentialBackoff {
            initial_interval: self.initial_delay,
            max_interval: self.max_delay,
            max_elapsed_time: None,
            ..Default::default()
        };

        let mut attempt = 0;
        loop {
            match f().await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempt += 1;
                    if attempt > self.max_retries {
                        return Err(err);
                    }

                    if let Some(delay) = backoff.next_backoff() {
                        tracing::debug!(
                            "Retrying after {:?}, attempt {}/{}",
                            delay,
                            attempt,
                            self.max_retries
                        );
                        tokio::time::sleep(delay).await;
                    } else {
                        return Err(err);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_succeeds_on_first_attempt() {
        let policy = RetryPolicy::default();
        let result = policy
            .retry(|| async { Ok::<i32, String>(42) })
            .await
            .unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_retry_succeeds_after_failures() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let policy = RetryPolicy::default();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = policy
            .retry(move || {
                let attempts = attempts_clone.clone();
                async move {
                    let count = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                    if count < 2 {
                        Err("transient error")
                    } else {
                        Ok(42)
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(result, 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_fails_after_max_retries() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let policy = RetryPolicy::default().with_max_retries(1);
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = policy
            .retry(move || {
                let attempts = attempts_clone.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("permanent error")
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 2); // Initial attempt + 1 retry
    }
}
