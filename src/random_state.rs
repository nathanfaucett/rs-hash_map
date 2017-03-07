#[allow(deprecated)] use core::hash::{BuildHasher, SipHasher13};
use core::fmt;

use prng::Prng;
use rng::Rng;

use super::default_hasher::DefaultHasher;


#[derive(Clone)]
pub struct RandomState {
    k0: u64,
    k1: u64,
}

impl RandomState {
    #[inline]
    pub fn new() -> RandomState {
        let mut prng = Prng::new();

        RandomState {
            k0: prng.next_u64(),
            k1: prng.next_u64(),
        }
    }
}

impl BuildHasher for RandomState {
    type Hasher = DefaultHasher;

    #[allow(deprecated)]
    #[inline(always)]
    fn build_hasher(&self) -> DefaultHasher {
        DefaultHasher::new(SipHasher13::new_with_keys(self.k0, self.k1))
    }
}

impl Default for RandomState {
    #[inline(always)]
    fn default() -> RandomState {
        RandomState::new()
    }
}

impl fmt::Debug for RandomState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("RandomState { .. }")
    }
}
