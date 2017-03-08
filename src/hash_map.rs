use self::Entry::*;
use self::VacantEntryState::*;

use core::borrow::Borrow;
use core::cmp::max;
use core::fmt::{self, Debug};
#[allow(deprecated)]
use core::hash::{Hash, BuildHasher};
use core::iter::{FromIterator, FusedIterator};
use core::mem::{self, replace};
use core::ops::{Deref, Index, IndexMut};

use collection_traits::*;

use super::random_state::RandomState;

use super::table::{self, Bucket, EmptyBucket, FullBucket, FullBucketMut, RawTable, SafeHash};
use super::table::BucketState::{Empty, Full};


const MIN_NONZERO_RAW_CAPACITY: usize = 32;


#[derive(Clone)]
struct DefaultResizePolicy;

impl DefaultResizePolicy {
    #[inline(always)]
    fn new() -> Self {
        DefaultResizePolicy
    }
    #[inline]
    pub fn raw_capacity(&self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            let mut raw_cap = len * 11 / 10;
            assert!(raw_cap >= len, "raw_cap overflow");
            raw_cap = raw_cap.checked_next_power_of_two().expect("raw_capacity overflow");
            raw_cap = max(MIN_NONZERO_RAW_CAPACITY, raw_cap);
            raw_cap
        }
    }
    #[inline]
    fn capacity(&self, raw_cap: usize) -> usize {
        (raw_cap * 10 + 10 - 1) / 11
    }
}


const DISPLACEMENT_THRESHOLD: usize = 128;


#[derive(Clone)]
pub struct HashMap<K, V, S = RandomState> {
    hash_builder: S,
    table: RawTable<K, V>,
    resize_policy: DefaultResizePolicy,
    long_probes: bool,
}

fn search_hashed<K, V, M, F>(table: M, hash: SafeHash, mut is_match: F) -> InternalEntry<K, V, M>
    where M: Deref<Target = RawTable<K, V>>,
          F: FnMut(&K) -> bool
{
    if table.capacity() == 0 {
        return InternalEntry::TableIsEmpty;
    }

    let size = table.size();
    let mut probe = Bucket::new(table, hash);
    let mut displacement = 0;

    loop {
        let full = match probe.peek() {
            Empty(bucket) => {
                return InternalEntry::Vacant {
                    hash: hash,
                    elem: NoElem(bucket, displacement),
                };
            }
            Full(bucket) => bucket,
        };

        let probe_displacement = full.displacement();

        if probe_displacement < displacement {
            return InternalEntry::Vacant {
                hash: hash,
                elem: NeqElem(full, probe_displacement),
            };
        }

        if hash == full.hash() {
            if is_match(full.read().0) {
                return InternalEntry::Occupied { elem: full };
            }
        }
        displacement += 1;
        probe = full.next();
        debug_assert!(displacement <= size);
    }
}
fn pop_internal<K, V>(starting_bucket: FullBucketMut<K, V>)
    -> (K, V, &mut RawTable<K, V>)
{
    let (empty, retkey, retval) = starting_bucket.take();
    let mut gap = match empty.gap_peek() {
        Ok(b) => b,
        Err(b) => return (retkey, retval, b.into_table()),
    };

    while gap.full().displacement() != 0 {
        gap = match gap.shift() {
            Ok(b) => b,
            Err(b) => {
                return (retkey, retval, b.into_table());
            },
        };
    }

    (retkey, retval, gap.into_bucket().into_table())
}
fn robin_hood<'a, K: 'a, V: 'a>(bucket: FullBucketMut<'a, K, V>,
                                mut displacement: usize,
                                mut hash: SafeHash,
                                mut key: K,
                                mut val: V)
                                -> &'a mut V {
    let start_index = bucket.index();
    let size = bucket.table().size();
    let mut bucket = bucket.stash();
    let idx_end = start_index + size - bucket.displacement();

    loop {
        let (old_hash, old_key, old_val) = bucket.replace(hash, key, val);
        hash = old_hash;
        key = old_key;
        val = old_val;

        loop {
            displacement += 1;
            let probe = bucket.next();
            debug_assert!(probe.index() != idx_end);

            let full_bucket = match probe.peek() {
                Empty(bucket) => {
                    let bucket = bucket.put(hash, key, val);
                    return bucket.into_table().into_mut_refs().1;
                }
                Full(bucket) => bucket,
            };

            let probe_displacement = full_bucket.displacement();

            bucket = full_bucket;

            if probe_displacement < displacement {
                displacement = probe_displacement;
                break;
            }
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    #[inline(always)]
    fn make_hash<X: ?Sized>(&self, x: &X) -> SafeHash
        where X: Hash
    {
        table::make_hash(&self.hash_builder, x)
    }
    #[inline]
    fn search<'a, Q: ?Sized>(&'a self, q: &Q) -> InternalEntry<K, V, &'a RawTable<K, V>>
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        search_hashed(&self.table, hash, |k| q.eq(k.borrow()))
    }
    #[inline]
    fn search_mut<'a, Q: ?Sized>(&'a mut self, q: &Q) -> InternalEntry<K, V, &'a mut RawTable<K, V>>
        where K: Borrow<Q>,
              Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        search_hashed(&mut self.table, hash, |k| q.eq(k.borrow()))
    }
    fn insert_hashed_ordered(&mut self, hash: SafeHash, k: K, v: V) {
        let raw_cap = self.raw_capacity();
        let mut buckets = Bucket::new(&mut self.table, hash);
        // note that buckets.index() keeps increasing
        // even if the pointer wraps back to the first bucket.
        let limit_bucket = buckets.index() + raw_cap;

        loop {
            // We don't need to compare hashes for value swap.
            // Not even DIBs for Robin Hood.
            buckets = match buckets.peek() {
                Empty(empty) => {
                    empty.put(hash, k, v);
                    return;
                }
                Full(b) => b.into_bucket(),
            };
            buckets.next();
            debug_assert!(buckets.index() < limit_bucket);
        }
    }
}

impl<K: Hash + Eq, V> HashMap<K, V, RandomState> {
    #[inline(always)]
    pub fn new() -> HashMap<K, V, RandomState> {
        Default::default()
    }
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> HashMap<K, V, RandomState> {
        HashMap::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K, V, S> HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    #[inline(always)]
    pub fn with_hasher(hash_builder: S) -> HashMap<K, V, S> {
        HashMap {
            hash_builder: hash_builder,
            resize_policy: DefaultResizePolicy::new(),
            table: RawTable::new(0),
            long_probes: false,
        }
    }
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        let resize_policy = DefaultResizePolicy::new();
        let raw_cap = resize_policy.raw_capacity(capacity);
        HashMap {
            hash_builder: hash_builder,
            resize_policy: resize_policy,
            table: RawTable::new(raw_cap),
            long_probes: false,
        }
    }
    #[inline(always)]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.resize_policy.capacity(self.raw_capacity())
    }
    #[inline(always)]
    pub fn raw_capacity(&self) -> usize {
        self.table.capacity()
    }
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let remaining = self.capacity() - self.len();

        if remaining < additional {
            let min_cap = self.len().checked_add(additional).expect("reserve overflow");
            let raw_cap = self.resize_policy.raw_capacity(min_cap);
            self.resize(raw_cap);
        } else if self.long_probes && remaining <= self.len() {
            let new_capacity = self.table.capacity() * 2;
            self.resize(new_capacity);
        }
    }
    fn resize(&mut self, new_raw_cap: usize) {
        assert!(self.table.size() <= new_raw_cap);
        assert!(new_raw_cap.is_power_of_two() || new_raw_cap == 0);

        self.long_probes = false;
        let mut old_table = replace(&mut self.table, RawTable::new(new_raw_cap));
        let old_size = old_table.size();

        if old_table.size() == 0 {
            return;
        }

        let mut bucket = Bucket::head_bucket(&mut old_table);

        loop {
            bucket = match bucket.peek() {
                Full(bucket) => {
                    let h = bucket.hash();
                    let (b, k, v) = bucket.take();
                    self.insert_hashed_ordered(h, k, v);
                    if b.table().size() == 0 {
                        break;
                    }
                    b.into_bucket()
                }
                Empty(b) => b.into_bucket(),
            };
            bucket.next();
        }

        assert_eq!(self.table.size(), old_size);
    }
    pub fn shrink_to_fit(&mut self) {
        let new_raw_cap = self.resize_policy.raw_capacity(self.len());

        if self.raw_capacity() != new_raw_cap {
            let old_table = replace(&mut self.table, RawTable::new(new_raw_cap));
            let old_size = old_table.size();

            for (h, k, v) in old_table.into_iter() {
                self.insert_hashed_nocheck(h, k, v);
            }

            debug_assert_eq!(self.table.size(), old_size);
        }
    }
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, k: K, v: V) -> Option<V> {
        let entry = search_hashed(&mut self.table, hash, |key| *key == k)
            .into_entry(k, &mut self.long_probes);
        match entry {
            Some(Occupied(mut elem)) => Some(elem.insert(v)),
            Some(Vacant(elem)) => {
                elem.insert(v);
                None
            }
            None => unreachable!(),
        }
    }
    #[inline(always)]
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            inner: self.iter(),
        }
    }
    #[inline(always)]
    pub fn values(&self) -> Values<K, V> {
        Values {
            inner: self.iter(),
        }
    }
    #[inline(always)]
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut { inner: self.iter_mut() }
    }
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        self.reserve(1);
        let hash = self.make_hash(&key);
        search_hashed(&mut self.table, hash, |q| q.eq(&key))
            .into_entry(key, &mut self.long_probes).expect("unreachable")
    }
    #[inline(always)]
    pub fn drain(&mut self) -> Drain<K, V> {
        Drain {
            inner: self.table.drain(),
        }
    }
    #[inline(always)]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.search(k).into_occupied_bucket().map(|bucket| bucket.into_refs().1)
    }
    #[inline(always)]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.search_mut(k).into_occupied_bucket().map(|bucket| bucket.into_mut_refs().1)
    }
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&K, &mut V) -> bool
    {
        if self.table.capacity() == 0 || self.table.size() == 0 {
            return;
        }
        let mut bucket = Bucket::head_bucket(&mut self.table);
        bucket.prev();
        let tail = bucket.index();
        loop {
            bucket = match bucket.peek() {
                Full(mut full) => {
                    let should_remove = {
                        let (k, v) = full.read_mut();
                        !f(k, v)
                    };
                    if should_remove {
                        let prev_idx = full.index();
                        let prev_raw = full.raw();
                        let (_, _, t) = pop_internal(full);
                        Bucket::new_from(prev_raw, prev_idx, t)
                    } else {
                        full.into_bucket()
                    }
                },
                Empty(b) => {
                    b.into_bucket()
                }
            };
            bucket.prev();
            if bucket.index() == tail {
                break;
            }
        }
    }
}


impl<K, V, S> PartialEq for HashMap<K, V, S>
    where K: Eq + Hash,
          V: PartialEq,
          S: BuildHasher
{
    #[inline]
    fn eq(&self, other: &HashMap<K, V, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}


impl<K, V, S> Eq for HashMap<K, V, S>
    where K: Eq + Hash,
          V: Eq,
          S: BuildHasher
{
}


impl<K, V, S> Debug for HashMap<K, V, S>
    where K: Eq + Hash + Debug,
          V: Debug,
          S: BuildHasher
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}


impl<K, V, S> Default for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher + Default
{
    #[inline(always)]
    fn default() -> HashMap<K, V, S> {
        HashMap::with_hasher(Default::default())
    }
}


impl<K, V, S> Collection for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.table.size()
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.drain();
    }
}


impl<'a, K, V, S> Insert<K, V> for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    type Output = Option<V>;

    #[inline]
    fn insert(&mut self, k: K, v: V) -> Self::Output {
        let hash = self.make_hash(&k);
        self.reserve(1);
        self.insert_hashed_nocheck(hash, k, v)
    }
}

impl<'a, K, Q: ?Sized, V, S> Remove<&'a Q> for HashMap<K, V, S>
    where K: Eq + Hash + Borrow<Q>,
          Q: Eq + Hash,
          S: BuildHasher
{
    type Output = Option<V>;

    #[inline]
    fn remove(&mut self, k: &Q) -> Self::Output {
        if self.table.size() == 0 {
            return None;
        }
        self.search_mut(k).into_occupied_bucket().map(|bucket| pop_internal(bucket).1)
    }
}

impl<'a, K, V, S> Iterable<'a, (&'a K, &'a V)> for HashMap<K, V, S>
    where K: 'a + Eq + Hash,
          V: 'a,
          S: 'a + BuildHasher,
{
    type Iter = Iter<'a, K, V>;

    #[inline(always)]
    fn iter(&'a self) -> Self::Iter {
        Iter {
            inner: self.table.iter(),
        }
    }
}

impl<'a, K, V, S> IterableMut<'a, (&'a K, &'a mut V)> for HashMap<K, V, S>
    where K: 'a + Eq + Hash,
          V: 'a,
          S: 'a + BuildHasher,
{
    type IterMut = IterMut<'a, K, V>;

    #[inline(always)]
    fn iter_mut(&'a mut self) -> Self::IterMut {
        IterMut {
            inner: self.table.iter_mut(),
        }
    }
}

impl<'a, K, Q: ?Sized, V, S> Index<&'a Q> for HashMap<K, V, S>
    where K: Eq + Hash + Borrow<Q>,
          Q: Eq + Hash,
          S: BuildHasher
{
    type Output = V;

    #[inline(always)]
    fn index(&self, index: &Q) -> &Self::Output {
        self.get(index).expect("no entry found for key")
    }
}

impl<'a, K, Q: ?Sized, V, S> IndexMut<&'a Q> for HashMap<K, V, S>
    where K: Eq + Hash + Borrow<Q>,
          Q: Eq + Hash,
          S: BuildHasher,
{
    #[inline(always)]
    fn index_mut(&mut self, index: &Q) -> &mut Self::Output {
        self.get_mut(index).expect("no entry found for key")
    }
}

impl<'a, K, Q, V, S> Map<'a, K, Q, V> for HashMap<K, V, S>
    where K: 'a + Eq + Hash + Borrow<Q>,
          V: 'a,
          Q: 'a + ?Sized + Hash + Eq,
          S: 'a + BuildHasher,
{
    #[inline(always)]
    fn contains_key(&self, k: &Q) -> bool {
        self.search(k).into_occupied_bucket().is_some()
    }
}

impl<'a, K, Q, V, S> MapMut<'a, K, Q, V> for HashMap<K, V, S>
    where K: 'a + Eq + Hash + Borrow<Q>,
          V: 'a,
          Q: 'a + ?Sized + Hash + Eq,
          S: 'a + BuildHasher {}


pub struct Iter<'a, K: 'a, V: 'a> {
    inner: table::Iter<'a, K, V>,
}

impl<'a, K, V> Clone for Iter<'a, K, V> {
    #[inline(always)]
    fn clone(&self) -> Iter<'a, K, V> {
        Iter { inner: self.inner.clone() }
    }
}


impl<'a, K: Debug, V: Debug> fmt::Debug for Iter<'a, K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.clone())
            .finish()
    }
}

pub struct IterMut<'a, K: 'a, V: 'a> {
    inner: table::IterMut<'a, K, V>,
}

pub struct IntoIter<K, V> {
    inner: table::IntoIter<K, V>,
}

pub struct Keys<'a, K: 'a, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Clone for Keys<'a, K, V> {
    #[inline(always)]
    fn clone(&self) -> Keys<'a, K, V> {
        Keys { inner: self.inner.clone() }
    }
}


impl<'a, K: Debug, V: Debug> fmt::Debug for Keys<'a, K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.clone())
            .finish()
    }
}

pub struct Values<'a, K: 'a, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Clone for Values<'a, K, V> {
    #[inline(always)]
    fn clone(&self) -> Values<'a, K, V> {
        Values { inner: self.inner.clone() }
    }
}


impl<'a, K: Debug, V: Debug> fmt::Debug for Values<'a, K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.clone())
            .finish()
    }
}


pub struct Drain<'a, K: 'a, V: 'a> {
    inner: table::Drain<'a, K, V>,
}

pub struct ValuesMut<'a, K: 'a, V: 'a> {
    inner: IterMut<'a, K, V>,
}

enum InternalEntry<K, V, M> {
    Occupied { elem: FullBucket<K, V, M> },
    Vacant {
        hash: SafeHash,
        elem: VacantEntryState<K, V, M>,
    },
    TableIsEmpty,
}

impl<K, V, M> InternalEntry<K, V, M> {
    #[inline(always)]
    fn into_occupied_bucket(self) -> Option<FullBucket<K, V, M>> {
        match self {
            InternalEntry::Occupied { elem } => Some(elem),
            _ => None,
        }
    }
}

impl<'a, K, V> InternalEntry<K, V, &'a mut RawTable<K, V>> {
    #[inline]
    fn into_entry(self, key: K, long_probes: &'a mut bool) -> Option<Entry<'a, K, V>> {
        match self {
            InternalEntry::Occupied { elem } => {
                Some(Occupied(OccupiedEntry {
                    key: Some(key),
                    elem: elem,
                }))
            }
            InternalEntry::Vacant { hash, elem } => {
                Some(Vacant(VacantEntry {
                    hash: hash,
                    key: key,
                    elem: elem,
                    long_probes: long_probes,
                }))
            }
            InternalEntry::TableIsEmpty => None,
        }
    }
}

pub enum Entry<'a, K: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}


impl<'a, K: 'a + Debug, V: 'a + Debug> Debug for Entry<'a, K, V> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Vacant(ref v) => {
                f.debug_tuple("Entry")
                    .field(v)
                    .finish()
            }
            Occupied(ref o) => {
                f.debug_tuple("Entry")
                    .field(o)
                    .finish()
            }
        }
    }
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    key: Option<K>,
    elem: FullBucket<K, V, &'a mut RawTable<K, V>>,
}


impl<'a, K: 'a + Debug, V: 'a + Debug> Debug for OccupiedEntry<'a, K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

pub struct VacantEntry<'a, K: 'a, V: 'a> {
    hash: SafeHash,
    key: K,
    elem: VacantEntryState<K, V, &'a mut RawTable<K, V>>,
    long_probes: &'a mut bool,
}


impl<'a, K: 'a + Debug, V: 'a> Debug for VacantEntry<'a, K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("VacantEntry")
            .field(self.key())
            .finish()
    }
}

enum VacantEntryState<K, V, M> {
    NeqElem(FullBucket<K, V, M>, usize),
    NoElem(EmptyBucket<K, V, M>, usize),
}


impl<'a, K, V, S> IntoIterator for &'a HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline(always)]
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}


impl<'a, K, V, S> IntoIterator for &'a mut HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline(always)]
    fn into_iter(mut self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}


impl<K, V, S> IntoIterator for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline(always)]
    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter { inner: self.table.into_iter() }
    }
}


impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline(always)]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next()
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}


impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}


impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline(always)]
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.inner.next()
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}


impl<'a, K, V> fmt::Debug for IterMut<'a, K, V>
    where K: fmt::Debug,
          V: fmt::Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter())
            .finish()
    }
}


impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline(always)]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next().map(|(_, k, v)| (k, v))
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}


impl<K: Debug, V: Debug> fmt::Debug for IntoIter<K, V> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter())
            .finish()
    }
}


impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline(always)]
    fn next(&mut self) -> Option<(&'a K)> {
        self.inner.next().map(|(k, _)| k)
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}


impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline(always)]
    fn next(&mut self) -> Option<(&'a V)> {
        self.inner.next().map(|(_, v)| v)
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for Values<'a, K, V> {}


impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline(always)]
    fn next(&mut self) -> Option<(&'a mut V)> {
        self.inner.next().map(|(_, v)| v)
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}


impl<'a, K, V> fmt::Debug for ValuesMut<'a, K, V>
    where K: fmt::Debug,
          V: fmt::Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.inner.iter())
            .finish()
    }
}


impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    #[inline(always)]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next().map(|(_, k, v)| (k, v))
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Drain<'a, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for Drain<'a, K, V> {}


impl<'a, K, V> fmt::Debug for Drain<'a, K, V>
    where K: fmt::Debug,
          V: fmt::Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.inner.iter())
            .finish()
    }
}

impl<'a, K, V> Entry<'a, K, V> {
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
    #[inline]
    pub fn key(&self) -> &K {
        match *self {
            Occupied(ref entry) => entry.key(),
            Vacant(ref entry) => entry.key(),
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    #[inline(always)]
    pub fn key(&self) -> &K {
        self.elem.read().0
    }
    #[inline(always)]
    pub fn remove_entry(self) -> (K, V) {
        let (k, v, _) = pop_internal(self.elem);
        (k, v)
    }
    #[inline(always)]
    pub fn get(&self) -> &V {
        self.elem.read().1
    }
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut V {
        self.elem.read_mut().1
    }
    #[inline(always)]
    pub fn into_mut(self) -> &'a mut V {
        self.elem.into_mut_refs().1
    }
    #[inline(always)]
    pub fn insert(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }
    #[inline(always)]
    pub fn remove(self) -> V {
        pop_internal(self.elem).1
    }
    #[inline(always)]
    fn take_key(&mut self) -> Option<K> {
        self.key.take()
    }
}

impl<'a, K: 'a, V: 'a> VacantEntry<'a, K, V> {
    #[inline(always)]
    pub fn key(&self) -> &K {
        &self.key
    }
    #[inline(always)]
    pub fn into_key(self) -> K {
        self.key
    }
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        match self.elem {
            NeqElem(bucket, disp) => {
                if disp >= DISPLACEMENT_THRESHOLD {
                    *self.long_probes = true;
                }
                robin_hood(bucket, disp, self.hash, self.key, value)
            },
            NoElem(bucket, disp) => {
                if disp >= DISPLACEMENT_THRESHOLD {
                    *self.long_probes = true;
                }
                bucket.put(self.hash, self.key, value).into_mut_refs().1
            },
        }
    }
}


impl<K, V, S> FromIterator<(K, V)> for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher + Default
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> HashMap<K, V, S> {
        let mut map = HashMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}


impl<K, V, S> Extend<(K, V)> for HashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };
        self.reserve(reserve);
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}


impl<'a, K, V, S> Extend<(&'a K, &'a V)> for HashMap<K, V, S>
    where K: Eq + Hash + Copy,
          V: Copy,
          S: BuildHasher
{
    #[inline(always)]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K, S, Q: ?Sized> super::Recover<Q> for HashMap<K, (), S>
    where K: Eq + Hash + Borrow<Q>,
          S: BuildHasher,
          Q: Eq + Hash
{
    type Key = K;

    #[inline]
    fn get(&self, key: &Q) -> Option<&K> {
        self.search(key).into_occupied_bucket().map(|bucket| bucket.into_refs().0)
    }

    #[inline]
    fn take(&mut self, key: &Q) -> Option<K> {
        if self.table.size() == 0 {
            return None;
        }

        self.search_mut(key).into_occupied_bucket().map(|bucket| pop_internal(bucket).0)
    }

    #[inline]
    fn replace(&mut self, key: K) -> Option<K> {
        self.reserve(1);

        match self.entry(key) {
            Occupied(mut occupied) => {
                let key = occupied.take_key().unwrap();
                Some(mem::replace(occupied.elem.read_mut().0, key))
            }
            Vacant(vacant) => {
                vacant.insert(());
                None
            }
        }
    }
}

#[allow(dead_code)]
fn assert_covariance() {
    #[inline(always)]
    fn map_key<'new>(v: HashMap<&'static str, u8>) -> HashMap<&'new str, u8> {
        v
    }
    #[inline(always)]
    fn map_val<'new>(v: HashMap<u8, &'static str>) -> HashMap<u8, &'new str> {
        v
    }
    #[inline(always)]
    fn iter_key<'a, 'new>(v: Iter<'a, &'static str, u8>) -> Iter<'a, &'new str, u8> {
        v
    }
    #[inline(always)]
    fn iter_val<'a, 'new>(v: Iter<'a, u8, &'static str>) -> Iter<'a, u8, &'new str> {
        v
    }
    #[inline(always)]
    fn into_iter_key<'new>(v: IntoIter<&'static str, u8>) -> IntoIter<&'new str, u8> {
        v
    }
    #[inline(always)]
    fn into_iter_val<'new>(v: IntoIter<u8, &'static str>) -> IntoIter<u8, &'new str> {
        v
    }
    #[inline(always)]
    fn keys_key<'a, 'new>(v: Keys<'a, &'static str, u8>) -> Keys<'a, &'new str, u8> {
        v
    }
    #[inline(always)]
    fn keys_val<'a, 'new>(v: Keys<'a, u8, &'static str>) -> Keys<'a, u8, &'new str> {
        v
    }
    #[inline(always)]
    fn values_key<'a, 'new>(v: Values<'a, &'static str, u8>) -> Values<'a, &'new str, u8> {
        v
    }
    #[inline(always)]
    fn values_val<'a, 'new>(v: Values<'a, u8, &'static str>) -> Values<'a, u8, &'new str> {
        v
    }
    #[inline(always)]
    fn drain<'new>(d: Drain<'static, &'static str, &'static str>)
                   -> Drain<'new, &'new str, &'new str> {
        d
    }
}
