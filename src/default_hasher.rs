#[allow(deprecated)] use core::hash::{Hasher, SipHasher13};


#[allow(deprecated)]
pub struct DefaultHasher {
    hasher: SipHasher13,
}

impl DefaultHasher {
    #[allow(deprecated)]
    #[inline(always)]
    pub fn new(hasher: SipHasher13) -> DefaultHasher {
        DefaultHasher {
            hasher: hasher,
        }
    }
}

impl Default for DefaultHasher {
    #[allow(deprecated)]
    #[inline(always)]
    fn default() -> DefaultHasher {
        DefaultHasher::new(SipHasher13::new_with_keys(0, 0))
    }
}

impl Hasher for DefaultHasher {
    #[inline(always)]
    fn write(&mut self, msg: &[u8]) {
        self.hasher.write(msg)
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.hasher.finish()
    }
}
