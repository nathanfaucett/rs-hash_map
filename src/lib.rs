#![feature(alloc)]
#![feature(collections)]
#![feature(pub_restricted)]
#![feature(fused)]
#![feature(heap_api)]
#![feature(core_intrinsics)]
#![feature(unique)]
#![feature(shared)]
#![feature(oom)]
#![feature(dropck_eyepatch)]
#![feature(generic_param_attrs)]
#![feature(sip_hash_13)]
#![no_std]


extern crate alloc;
extern crate collections;

extern crate collection_traits;
extern crate prng;
extern crate rng;
extern crate vector;


mod default_hasher;
mod hash_map;
mod random_state;
mod table;


pub use self::default_hasher::DefaultHasher;
pub use self::hash_map::*;
pub use self::random_state::RandomState;
pub use self::table::*;



trait Recover<Q: ?Sized> {
    type Key;

    fn get(&self, key: &Q) -> Option<&Self::Key>;
    fn take(&mut self, key: &Q) -> Option<Self::Key>;
    fn replace(&mut self, key: Self::Key) -> Option<Self::Key>;
}
