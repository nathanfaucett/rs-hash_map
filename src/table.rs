use alloc::heap::{allocate, deallocate};

use core::cmp;
use core::hash::{BuildHasher, Hash, Hasher};
use core::intrinsics::needs_drop;
use core::marker;
use core::mem::{self, align_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, Unique, Shared};

use self::BucketState::*;


type HashUint = usize;


const EMPTY_BUCKET: HashUint = 0;


pub struct RawTable<K, V> {
    capacity: usize,
    size: usize,
    hashes: Unique<HashUint>,
    marker: marker::PhantomData<(K, V)>,
}

unsafe impl<K: Send, V: Send> Send for RawTable<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RawTable<K, V> {}

pub struct RawBucket<K, V> {
    hash: *mut HashUint,
    pair: *const (K, V),
    _marker: marker::PhantomData<(K, V)>,
}

impl<K, V> Copy for RawBucket<K, V> {}
impl<K, V> Clone for RawBucket<K, V> {
    fn clone(&self) -> RawBucket<K, V> {
        *self
    }
}

pub struct Bucket<K, V, M> {
    raw: RawBucket<K, V>,
    idx: usize,
    table: M,
}

impl<K, V, M: Copy> Copy for Bucket<K, V, M> {}
impl<K, V, M: Copy> Clone for Bucket<K, V, M> {
    fn clone(&self) -> Bucket<K, V, M> {
        *self
    }
}

pub struct EmptyBucket<K, V, M> {
    raw: RawBucket<K, V>,
    idx: usize,
    table: M,
}

pub struct FullBucket<K, V, M> {
    raw: RawBucket<K, V>,
    idx: usize,
    table: M,
}

pub type FullBucketMut<'table, K, V> = FullBucket<K, V, &'table mut RawTable<K, V>>;

pub enum BucketState<K, V, M> {
    Empty(EmptyBucket<K, V, M>),
    Full(FullBucket<K, V, M>),
}

pub struct GapThenFull<K, V, M> {
    gap: EmptyBucket<K, V, ()>,
    full: FullBucket<K, V, M>,
}

#[derive(PartialEq, Copy, Clone)]
pub struct SafeHash {
    hash: HashUint,
}

impl SafeHash {
    #[inline(always)]
    pub fn inspect(&self) -> HashUint {
        self.hash
    }

    #[inline(always)]
    pub fn new(hash: u64) -> Self {
        let hash_bits = size_of::<HashUint>() * 8;
        SafeHash { hash: (1 << (hash_bits - 1)) | (hash as HashUint) }
    }
}

pub fn make_hash<T: ?Sized, S>(hash_state: &S, t: &T) -> SafeHash
    where T: Hash,
          S: BuildHasher
{
    let mut state = hash_state.build_hasher();
    t.hash(&mut state);
    SafeHash::new(state.finish())
}

#[test]
fn can_alias_safehash_as_hash() {
    assert_eq!(size_of::<SafeHash>(), size_of::<HashUint>())
}

impl<K, V> RawBucket<K, V> {
    unsafe fn offset(self, count: isize) -> RawBucket<K, V> {
        RawBucket {
            hash: self.hash.offset(count),
            pair: self.pair.offset(count),
            _marker: marker::PhantomData,
        }
    }
}

impl<K, V, M> FullBucket<K, V, M> {
    pub fn table(&self) -> &M {
        &self.table
    }
    pub fn into_table(self) -> M {
        self.table
    }
    pub fn index(&self) -> usize {
        self.idx
    }
    pub fn raw(&self) -> RawBucket<K, V> {
        self.raw
    }
}

impl<K, V, M> EmptyBucket<K, V, M> {
    pub fn table(&self) -> &M {
        &self.table
    }
}

impl<K, V, M> Bucket<K, V, M> {
    pub fn index(&self) -> usize {
        self.idx
    }
    pub fn into_table(self) -> M {
        self.table
    }
}

impl<K, V, M> Deref for FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>>
{
    type Target = RawTable<K, V>;
    fn deref(&self) -> &RawTable<K, V> {
        &self.table
    }
}


pub trait Put<K, V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V>;
}


impl<'t, K, V> Put<K, V> for &'t mut RawTable<K, V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        *self
    }
}

impl<K, V, M> Put<K, V> for Bucket<K, V, M>
    where M: Put<K, V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        self.table.borrow_table_mut()
    }
}

impl<K, V, M> Put<K, V> for FullBucket<K, V, M>
    where M: Put<K, V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        self.table.borrow_table_mut()
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> Bucket<K, V, M> {
    pub fn new(table: M, hash: SafeHash) -> Bucket<K, V, M> {
        Bucket::at_index(table, hash.inspect() as usize)
    }

    pub fn new_from(r: RawBucket<K, V>, i: usize, t: M)
        -> Bucket<K, V, M>
    {
        Bucket {
            raw: r,
            idx: i,
            table: t,
        }
    }

    pub fn at_index(table: M, ib_index: usize) -> Bucket<K, V, M> {
        debug_assert!(table.capacity() > 0,
                      "Table should have capacity at this point");
        let ib_index = ib_index & (table.capacity() - 1);
        Bucket {
            raw: unsafe { table.first_bucket_raw().offset(ib_index as isize) },
            idx: ib_index,
            table: table,
        }
    }

    pub fn first(table: M) -> Bucket<K, V, M> {
        Bucket {
            raw: table.first_bucket_raw(),
            idx: 0,
            table: table,
        }
    }

    pub fn head_bucket(table: M) -> Bucket<K, V, M> {
        let mut bucket = Bucket::first(table);

        loop {
            bucket = match bucket.peek() {
                Full(full) => {
                    if full.displacement() == 0 {
                        bucket = full.into_bucket();
                        break;
                    }
                    full.into_bucket()
                }
                Empty(b) => {
                    b.into_bucket()
                }
            };
            bucket.next();
        }
        bucket
    }

    pub fn peek(self) -> BucketState<K, V, M> {
        match unsafe { *self.raw.hash } {
            EMPTY_BUCKET => {
                Empty(EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table,
                })
            }
            _ => {
                Full(FullBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table,
                })
            }
        }
    }

    pub fn next(&mut self) {
        self.idx += 1;
        let range = self.table.capacity();
        let dist = if self.idx & (range - 1) == 0 {
            1 - range as isize
        } else {
            1
        };
        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }

    pub fn prev(&mut self) {
        let range = self.table.capacity();
        let new_idx = self.idx.wrapping_sub(1) & (range - 1);
        let dist = (new_idx as isize).wrapping_sub(self.idx as isize);
        self.idx = new_idx;
        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> EmptyBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }
    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table,
        }
    }
    pub fn gap_peek(self) -> Result<GapThenFull<K, V, M>, Bucket<K, V, M>> {
        let gap = EmptyBucket {
            raw: self.raw,
            idx: self.idx,
            table: (),
        };

        match self.next().peek() {
            Full(bucket) => {
                Ok(GapThenFull {
                    gap: gap,
                    full: bucket,
                })
            }
            Empty(e) => Err(e.into_bucket()),
        }
    }
}

impl<K, V, M> EmptyBucket<K, V, M>
    where M: Put<K, V>
{
    pub fn put(mut self, hash: SafeHash, key: K, value: V) -> FullBucket<K, V, M> {
        unsafe {
            *self.raw.hash = hash.inspect();
            ptr::write(self.raw.pair as *mut (K, V), (key, value));

            self.table.borrow_table_mut().size += 1;
        }

        FullBucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table,
        }
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> FullBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table,
        }
    }

    pub fn stash(self) -> FullBucket<K, V, Self> {
        FullBucket {
            raw: self.raw,
            idx: self.idx,
            table: self,
        }
    }

    pub fn displacement(&self) -> usize {
        (self.idx.wrapping_sub(self.hash().inspect() as usize)) & (self.table.capacity() - 1)
    }

    #[inline]
    pub fn hash(&self) -> SafeHash {
        unsafe { SafeHash { hash: *self.raw.hash } }
    }

    pub fn read(&self) -> (&K, &V) {
        unsafe { (&(*self.raw.pair).0, &(*self.raw.pair).1) }
    }
}

impl<'t, K, V> FullBucket<K, V, &'t mut RawTable<K, V>> {
    pub fn take(mut self) -> (EmptyBucket<K, V, &'t mut RawTable<K, V>>, K, V) {
        self.table.size -= 1;

        unsafe {
            *self.raw.hash = EMPTY_BUCKET;
            let (k, v) = ptr::read(self.raw.pair);
            (EmptyBucket {
                 raw: self.raw,
                 idx: self.idx,
                 table: self.table,
             },
            k,
            v)
        }
    }
}

impl<K, V, M> FullBucket<K, V, M>
    where M: Put<K, V>
{
    pub fn replace(&mut self, h: SafeHash, k: K, v: V) -> (SafeHash, K, V) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash as *mut SafeHash, h);
            let (old_key, old_val) = ptr::replace(self.raw.pair as *mut (K, V), (k, v));

            (old_hash, old_key, old_val)
        }
    }
}

impl<K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + DerefMut
{
    pub fn read_mut(&mut self) -> (&mut K, &mut V) {
        let pair_mut = self.raw.pair as *mut (K, V);
        unsafe { (&mut (*pair_mut).0, &mut (*pair_mut).1) }
    }
}

impl<'t, K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + 't
{
    pub fn into_refs(self) -> (&'t K, &'t V) {
        unsafe { (&(*self.raw.pair).0, &(*self.raw.pair).1) }
    }
}

impl<'t, K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + DerefMut + 't
{
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) {
        let pair_mut = self.raw.pair as *mut (K, V);
        unsafe { (&mut (*pair_mut).0, &mut (*pair_mut).1) }
    }
}

impl<K, V, M> GapThenFull<K, V, M>
    where M: Deref<Target = RawTable<K, V>>
{
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> {
        &self.full
    }

    pub fn into_bucket(self) -> Bucket<K, V, M> {
        self.full.into_bucket()
    }

    pub fn shift(mut self) -> Result<GapThenFull<K, V, M>, Bucket<K, V, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, EMPTY_BUCKET);
            ptr::copy_nonoverlapping(self.full.raw.pair, self.gap.raw.pair as *mut (K, V), 1);
        }

        let FullBucket { raw: prev_raw, idx: prev_idx, .. } = self.full;

        match self.full.next().peek() {
            Full(bucket) => {
                self.gap.raw = prev_raw;
                self.gap.idx = prev_idx;

                self.full = bucket;

                Ok(self)
            }
            Empty(b) => Err(b.into_bucket()),
        }
    }
}


#[inline]
fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}

#[test]
fn test_rounding() {
    assert_eq!(round_up_to_next(0, 4), 0);
    assert_eq!(round_up_to_next(1, 4), 4);
    assert_eq!(round_up_to_next(2, 4), 4);
    assert_eq!(round_up_to_next(3, 4), 4);
    assert_eq!(round_up_to_next(4, 4), 4);
    assert_eq!(round_up_to_next(5, 4), 8);
}

#[inline]
fn calculate_offsets(hashes_size: usize,
                     pairs_size: usize,
                     pairs_align: usize)
                     -> (usize, usize, bool) {
    let pairs_offset = round_up_to_next(hashes_size, pairs_align);
    let (end_of_pairs, oflo) = pairs_offset.overflowing_add(pairs_size);

    (pairs_offset, end_of_pairs, oflo)
}

fn calculate_allocation(hash_size: usize,
                        hash_align: usize,
                        pairs_size: usize,
                        pairs_align: usize)
                        -> (usize, usize, usize, bool) {
    let hash_offset = 0;
    let (_, end_of_pairs, oflo) = calculate_offsets(hash_size, pairs_size, pairs_align);

    let align = cmp::max(hash_align, pairs_align);

    (align, hash_offset, end_of_pairs, oflo)
}

#[test]
fn test_offset_calculation() {
    assert_eq!(calculate_allocation(128, 8, 16, 8), (8, 0, 144, false));
    assert_eq!(calculate_allocation(3, 1, 2, 1), (1, 0, 5, false));
    assert_eq!(calculate_allocation(6, 2, 12, 4), (4, 0, 20, false));
    assert_eq!(calculate_offsets(128, 15, 4), (128, 143, false));
    assert_eq!(calculate_offsets(3, 2, 4), (4, 6, false));
    assert_eq!(calculate_offsets(6, 12, 4), (8, 20, false));
}

impl<K, V> RawTable<K, V> {
    unsafe fn new_uninitialized(capacity: usize) -> RawTable<K, V> {
        if capacity == 0 {
            return RawTable {
                size: 0,
                capacity: 0,
                hashes: Unique::empty(),
                marker: marker::PhantomData,
            };
        }

        let hashes_size = capacity.wrapping_mul(size_of::<HashUint>());
        let pairs_size = capacity.wrapping_mul(size_of::<(K, V)>());
        let (alignment, hash_offset, size, oflo) = calculate_allocation(hashes_size,
                                                                        align_of::<HashUint>(),
                                                                        pairs_size,
                                                                        align_of::<(K, V)>());
        assert!(!oflo, "capacity overflow");

        let size_of_bucket = size_of::<HashUint>().checked_add(size_of::<(K, V)>()).unwrap();
        assert!(size >=
                capacity.checked_mul(size_of_bucket)
                    .expect("capacity overflow"),
                "capacity overflow");

        let buffer = allocate(size, alignment);
        if buffer.is_null() {
            ::alloc::oom()
        }

        let hashes = buffer.offset(hash_offset as isize) as *mut HashUint;

        RawTable {
            capacity: capacity,
            size: 0,
            hashes: Unique::new(hashes),
            marker: marker::PhantomData,
        }
    }

    fn first_bucket_raw(&self) -> RawBucket<K, V> {
        let hashes_size = self.capacity * size_of::<HashUint>();
        let pairs_size = self.capacity * size_of::<(K, V)>();

        let buffer = self.hashes.as_ptr() as *mut u8;
        let (pairs_offset, _, oflo) =
            calculate_offsets(hashes_size, pairs_size, align_of::<(K, V)>());
        debug_assert!(!oflo, "capacity overflow");
        unsafe {
            RawBucket {
                hash: self.hashes.as_ptr(),
                pair: buffer.offset(pairs_offset as isize) as *const _,
                _marker: marker::PhantomData,
            }
        }
    }

    pub fn new(capacity: usize) -> RawTable<K, V> {
        unsafe {
            let ret = RawTable::new_uninitialized(capacity);
            ptr::write_bytes(ret.hashes.as_ptr(), 0, capacity);
            ret
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<K, V> {
        RawBuckets {
            raw: self.first_bucket_raw(),
            hashes_end: unsafe { self.hashes.as_ptr().offset(self.capacity as isize) },
            marker: marker::PhantomData,
        }
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.raw_buckets(),
            elems_left: self.size(),
            _marker: marker::PhantomData,
        }
    }

    pub fn into_iter(self) -> IntoIter<K, V> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();

        IntoIter {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    pub fn drain(&mut self) -> Drain<K, V> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();

        Drain {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: unsafe { Shared::new(self) },
            marker: marker::PhantomData,
        }
    }

    unsafe fn rev_move_buckets(&mut self) -> RevMoveBuckets<K, V> {
        let raw_bucket = self.first_bucket_raw();
        RevMoveBuckets {
            raw: raw_bucket.offset(self.capacity as isize),
            hashes_end: raw_bucket.hash,
            elems_left: self.size,
            marker: marker::PhantomData,
        }
    }
}

struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut HashUint,
    marker: marker::PhantomData<&'a ()>,
}

impl<'a, K, V> Clone for RawBuckets<'a, K, V> {
    fn clone(&self) -> RawBuckets<'a, K, V> {
        RawBuckets {
            raw: self.raw,
            hashes_end: self.hashes_end,
            marker: marker::PhantomData,
        }
    }
}

impl<'a, K, V> Iterator for RawBuckets<'a, K, V> {
    type Item = RawBucket<K, V>;

    fn next(&mut self) -> Option<RawBucket<K, V>> {
        while self.raw.hash != self.hashes_end {
            unsafe {
                let prev = ptr::replace(&mut self.raw, self.raw.offset(1));
                if *prev.hash != EMPTY_BUCKET {
                    return Some(prev);
                }
            }
        }
        None
    }
}

struct RevMoveBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut HashUint,
    elems_left: usize,
    marker: marker::PhantomData<&'a ()>,
}

impl<'a, K, V> Iterator for RevMoveBuckets<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            debug_assert!(self.raw.hash != self.hashes_end);

            unsafe {
                self.raw = self.raw.offset(-1);

                if *self.raw.hash != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some(ptr::read(self.raw.pair));
                }
            }
        }
    }
}

pub struct Iter<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: usize,
}

unsafe impl<'a, K: Sync, V: Sync> Sync for Iter<'a, K, V> {}
unsafe impl<'a, K: Sync, V: Sync> Send for Iter<'a, K, V> {}

impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            iter: self.iter.clone(),
            elems_left: self.elems_left,
        }
    }
}


pub struct IterMut<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: usize,
    // To ensure invariance with respect to V
    _marker: marker::PhantomData<&'a mut V>,
}

unsafe impl<'a, K: Sync, V: Sync> Sync for IterMut<'a, K, V> {}
unsafe impl<'a, K: Send, V: Send> Send for IterMut<'a, K, V> {}

impl<'a, K: 'a, V: 'a> IterMut<'a, K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.iter.clone(),
            elems_left: self.elems_left,
        }
    }
}


pub struct IntoIter<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>,
}

unsafe impl<K: Sync, V: Sync> Sync for IntoIter<K, V> {}
unsafe impl<K: Send, V: Send> Send for IntoIter<K, V> {}

impl<K, V> IntoIter<K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.iter.clone(),
            elems_left: self.table.size,
        }
    }
}


pub struct Drain<'a, K: 'a, V: 'a> {
    table: Shared<RawTable<K, V>>,
    iter: RawBuckets<'static, K, V>,
    marker: marker::PhantomData<&'a RawTable<K, V>>,
}

unsafe impl<'a, K: Sync, V: Sync> Sync for Drain<'a, K, V> {}
unsafe impl<'a, K: Send, V: Send> Send for Drain<'a, K, V> {}

impl<'a, K, V> Drain<'a, K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        unsafe {
            Iter {
                iter: self.iter.clone(),
                elems_left: (self.table).as_ref().size,
            }
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe { (&(*bucket.pair).0, &(*bucket.pair).1) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.elems_left
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            let pair_mut = bucket.pair as *mut (K, V);
            unsafe { (&(*pair_mut).0, &mut (*pair_mut).1) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.elems_left
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (SafeHash, K, V);

    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                let (k, v) = ptr::read(bucket.pair);
                (SafeHash { hash: *bucket.hash }, k, v)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.table.size();
        (size, Some(size))
    }
}
impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.table.size()
    }
}

impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (SafeHash, K, V);

    #[inline]
    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            unsafe {
                (*(self.table.as_ptr() as *mut RawTable<K, V>)).size -= 1;
                let (k, v) = ptr::read(bucket.pair);
                (SafeHash { hash: ptr::replace(bucket.hash, EMPTY_BUCKET) }, k, v)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = unsafe { self.table.as_ref().size() };
        (size, Some(size))
    }
}
impl<'a, K, V> ExactSizeIterator for Drain<'a, K, V> {
    fn len(&self) -> usize {
        unsafe { self.table.as_ref().size() }
    }
}

impl<'a, K: 'a, V: 'a> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> RawTable<K, V> {
        unsafe {
            let mut new_ht = RawTable::new_uninitialized(self.capacity());

            {
                let cap = self.capacity();
                let mut new_buckets = Bucket::first(&mut new_ht);
                let mut buckets = Bucket::first(self);
                while buckets.index() != cap {
                    match buckets.peek() {
                        Full(full) => {
                            let (h, k, v) = {
                                let (k, v) = full.read();
                                (full.hash(), k.clone(), v.clone())
                            };
                            *new_buckets.raw.hash = h.inspect();
                            ptr::write(new_buckets.raw.pair as *mut (K, V), (k, v));
                        }
                        Empty(..) => {
                            *new_buckets.raw.hash = EMPTY_BUCKET;
                        }
                    }
                    new_buckets.next();
                    buckets.next();
                }
            };

            new_ht.size = self.size();

            new_ht
        }
    }
}

unsafe impl<#[may_dangle] K, #[may_dangle] V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }

        unsafe {
            if needs_drop::<(K, V)>() {
                for _ in self.rev_move_buckets() {}
            }
        }

        let hashes_size = self.capacity * size_of::<HashUint>();
        let pairs_size = self.capacity * size_of::<(K, V)>();
        let (align, _, size, oflo) = calculate_allocation(hashes_size,
                                                          align_of::<HashUint>(),
                                                          pairs_size,
                                                          align_of::<(K, V)>());

        debug_assert!(!oflo, "should be impossible");

        unsafe {
            deallocate(self.hashes.as_ptr() as *mut u8, size, align);
        }
    }
}
