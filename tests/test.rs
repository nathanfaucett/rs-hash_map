extern crate collection_traits;
extern crate prng;
extern crate rng;
extern crate hash_map;


use collection_traits::*;

use hash_map::*;
use hash_map::Entry::*;

use rng::Rng;
use prng::Prng;


#[test]
fn test_str() {
    let mut m = HashMap::new();
    m.insert("a", 1);
    m.insert("b", 2);
    m.insert("c", 3);
    assert!(m.contains_key("a"));
    assert!(m.contains_key("b"));
    assert!(m.contains_key("c"));
    m["a"] = 2;
    m["b"] = 3;
    m["c"] = 4;
    assert_eq!(m["a"], 2);
    assert_eq!(m["b"], 3);
    assert_eq!(m["c"], 4);
}

#[test]
fn test_zero_capacities() {
    type HM = HashMap<i32, i32>;

    let m = HM::new();
    assert_eq!(m.capacity(), 0);

    let m = HM::default();
    assert_eq!(m.capacity(), 0);

    let m = HM::with_hasher(RandomState::new());
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity(0);
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity_and_hasher(0, RandomState::new());
    assert_eq!(m.capacity(), 0);

    let mut m = HM::new();
    m.insert(1, 1);
    m.insert(2, 2);
    m.remove(&1);
    m.remove(&2);
    m.shrink_to_fit();
    assert_eq!(m.capacity(), 0);

    let mut m = HM::new();
    m.reserve(0);
    assert_eq!(m.capacity(), 0);
}

#[test]
fn test_create_capacity_zero() {
    let mut m = HashMap::with_capacity(0);

    assert!(m.insert(1, 1).is_none());

    assert!(m.contains_key(&1));
    assert!(!m.contains_key(&0));
}

#[test]
fn test_insert() {
    let mut m = HashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&2).unwrap(), 4);
}

#[test]
fn test_clone() {
    let mut m = HashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    let m2 = m.clone();
    assert_eq!(*m2.get(&1).unwrap(), 2);
    assert_eq!(*m2.get(&2).unwrap(), 4);
    assert_eq!(m2.len(), 2);
}

#[test]
fn test_empty_remove() {
    let mut m: HashMap<isize, bool> = HashMap::new();
    assert_eq!(m.remove(&0), None);
}

#[test]
fn test_empty_entry() {
    let mut m: HashMap<isize, bool> = HashMap::new();
    match m.entry(0) {
        Occupied(_) => panic!(),
        Vacant(_) => {}
    }
    assert!(*m.entry(0).or_insert(true));
    assert_eq!(m.len(), 1);
}

#[test]
fn test_empty_iter() {
    let mut m: HashMap<isize, bool> = HashMap::new();
    assert_eq!(m.drain().next(), None);
    assert_eq!(m.keys().next(), None);
    assert_eq!(m.values().next(), None);
    assert_eq!(m.values_mut().next(), None);
    assert_eq!(m.iter().next(), None);
    assert_eq!(m.iter_mut().next(), None);
    assert_eq!(m.len(), 0);
    assert!(m.is_empty());
    assert_eq!(m.into_iter().next(), None);
}

#[test]
fn test_lots_of_insertions() {
    let mut m = HashMap::new();

    // Try this a few times to make sure we never screw up the hashmap's
    // internal state.
    for _ in 0..10 {
        assert!(m.is_empty());

        for i in 1..1001 {
            assert!(m.insert(i, i).is_none());

            for j in 1..i + 1 {
                let r = m.get(&j);
                assert_eq!(r, Some(&j));
            }

            for j in i + 1..1001 {
                let r = m.get(&j);
                assert_eq!(r, None);
            }
        }

        for i in 1001..2001 {
            assert!(!m.contains_key(&i));
        }

        // remove forwards
        for i in 1..1001 {
            assert!(m.remove(&i).is_some());

            for j in 1..i + 1 {
                assert!(!m.contains_key(&j));
            }

            for j in i + 1..1001 {
                assert!(m.contains_key(&j));
            }
        }

        for i in 1..1001 {
            assert!(!m.contains_key(&i));
        }

        for i in 1..1001 {
            assert!(m.insert(i, i).is_none());
        }

        // remove backwards
        for i in (1..1001).rev() {
            assert!(m.remove(&i).is_some());

            for j in i..1001 {
                assert!(!m.contains_key(&j));
            }

            for j in 1..i {
                assert!(m.contains_key(&j));
            }
        }
    }
}

#[test]
fn test_find_mut() {
    let mut m = HashMap::new();
    assert!(m.insert(1, 12).is_none());
    assert!(m.insert(2, 8).is_none());
    assert!(m.insert(5, 14).is_none());
    let new = 100;
    match m.get_mut(&5) {
        None => panic!(),
        Some(x) => *x = new,
    }
    assert_eq!(m.get(&5), Some(&new));
}

#[test]
fn test_insert_overwrite() {
    let mut m = HashMap::new();
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(!m.insert(1, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 3);
}

#[test]
fn test_insert_conflicts() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(5, 3).is_none());
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&1).unwrap(), 2);
}

#[test]
fn test_conflict_remove() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(m.insert(5, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert!(m.remove(&1).is_some());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
}

#[test]
fn test_is_empty() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(!m.is_empty());
    assert!(m.remove(&1).is_some());
    assert!(m.is_empty());
}

#[test]
fn test_pop() {
    let mut m = HashMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove(&1), Some(2));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_iterate() {
    let mut m = HashMap::with_capacity(4);
    for i in 0..32 {
        assert!(m.insert(i, i*2).is_none());
    }
    assert_eq!(m.len(), 32);

    let mut observed: u32 = 0;

    for (k, v) in &m {
        assert_eq!(*v, *k * 2);
        observed |= 1 << *k;
    }
    assert_eq!(observed, 0xFFFF_FFFF);
}

#[test]
fn test_keys() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = vec.into_iter().collect();
    let keys: Vec<_> = map.keys().cloned().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_values() {
    let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = vec.into_iter().collect();
    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn test_values_mut() {
    let vec = vec![(1, 1), (2, 2), (3, 3)];
    let mut map: HashMap<_, _> = vec.into_iter().collect();
    for value in map.values_mut() {
        *value = (*value) * 2
    }
    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&2));
    assert!(values.contains(&4));
    assert!(values.contains(&6));
}

#[test]
fn test_find() {
    let mut m = HashMap::new();
    assert!(m.get(&1).is_none());
    m.insert(1, 2);
    match m.get(&1) {
        None => panic!(),
        Some(v) => assert_eq!(*v, 2),
    }
}

#[test]
fn test_eq() {
    let mut m1 = HashMap::new();
    m1.insert(1, 2);
    m1.insert(2, 3);
    m1.insert(3, 4);

    let mut m2 = HashMap::new();
    m2.insert(1, 2);
    m2.insert(2, 3);

    assert!(m1 != m2);

    m2.insert(3, 4);

    assert_eq!(m1, m2);
}

#[test]
fn test_show() {
    let mut map = HashMap::new();
    let empty: HashMap<i32, i32> = HashMap::new();

    map.insert(1, 2);
    map.insert(3, 4);

    let map_str = format!("{:?}", map);

    assert!(map_str == "{1: 2, 3: 4}" ||
            map_str == "{3: 4, 1: 2}");
    assert_eq!(format!("{:?}", empty), "{}");
}

#[test]
fn test_expand() {
    let mut m = HashMap::new();

    assert_eq!(m.len(), 0);
    assert!(m.is_empty());

    let mut i = 0;
    let old_raw_cap = m.raw_capacity();
    while old_raw_cap == m.raw_capacity() {
        m.insert(i, i);
        i += 1;
    }

    assert_eq!(m.len(), i);
    assert!(!m.is_empty());
}

#[test]
fn test_behavior_resize_policy() {
    let mut m = HashMap::new();

    assert_eq!(m.len(), 0);
    assert_eq!(m.raw_capacity(), 0);
    assert!(m.is_empty());

    m.insert(0, 0);
    m.remove(&0);
    assert!(m.is_empty());
    let initial_raw_cap = m.raw_capacity();
    m.reserve(initial_raw_cap);
    let raw_cap = m.raw_capacity();

    assert_eq!(raw_cap, initial_raw_cap * 2);

    let mut i = 0;
    for _ in 0..raw_cap * 3 / 4 {
        m.insert(i, i);
        i += 1;
    }
    // three quarters full

    assert_eq!(m.len(), i);
    assert_eq!(m.raw_capacity(), raw_cap);

    for _ in 0..raw_cap / 4 {
        m.insert(i, i);
        i += 1;
    }
    // half full

    let new_raw_cap = m.raw_capacity();
    assert_eq!(new_raw_cap, raw_cap * 2);

    for _ in 0..raw_cap / 2 - 1 {
        i -= 1;
        m.remove(&i);
        assert_eq!(m.raw_capacity(), new_raw_cap);
    }
    // A little more than one quarter full.
    m.shrink_to_fit();
    assert_eq!(m.raw_capacity(), raw_cap);
    // again, a little more than half full
    for _ in 0..raw_cap / 2 - 1 {
        i -= 1;
        m.remove(&i);
    }
    m.shrink_to_fit();

    assert_eq!(m.len(), i);
    assert!(!m.is_empty());
    assert_eq!(m.raw_capacity(), initial_raw_cap);
}

#[test]
fn test_reserve_shrink_to_fit() {
    let mut m = HashMap::new();
    m.insert(0, 0);
    m.remove(&0);
    assert!(m.capacity() >= m.len());
    for i in 0..128 {
        m.insert(i, i);
    }
    m.reserve(256);

    let usable_cap = m.capacity();
    for i in 128..(128 + 256) {
        m.insert(i, i);
        assert_eq!(m.capacity(), usable_cap);
    }

    for i in 100..(128 + 256) {
        assert_eq!(m.remove(&i), Some(i));
    }
    m.shrink_to_fit();

    assert_eq!(m.len(), 100);
    assert!(!m.is_empty());
    assert!(m.capacity() >= m.len());

    for i in 0..100 {
        assert_eq!(m.remove(&i), Some(i));
    }
    m.shrink_to_fit();
    m.insert(0, 0);

    assert_eq!(m.len(), 1);
    assert!(m.capacity() >= m.len());
    assert_eq!(m.remove(&0), Some(0));
}

#[test]
fn test_from_iter() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    for &(k, v) in &xs {
        assert_eq!(map.get(&k), Some(&v));
    }
}

#[test]
fn test_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_mut_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_mut_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_index() {
    let mut map = HashMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    assert_eq!(map[&2], 1);
}

#[test]
#[should_panic]
fn test_index_nonexistent() {
    let mut map = HashMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    map[&4];
}

#[test]
fn test_entry() {
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    // Existing key (insert)
    match map.entry(1) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            assert_eq!(view.get(), &10);
            assert_eq!(view.insert(100), 10);
        }
    }
    assert_eq!(map.get(&1).unwrap(), &100);
    assert_eq!(map.len(), 6);


    // Existing key (update)
    match map.entry(2) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            let v = view.get_mut();
            let new_v = (*v) * 10;
            *v = new_v;
        }
    }
    assert_eq!(map.get(&2).unwrap(), &200);
    assert_eq!(map.len(), 6);

    // Existing key (take)
    match map.entry(3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove(), 30);
        }
    }
    assert_eq!(map.get(&3), None);
    assert_eq!(map.len(), 5);


    // Inexistent key (insert)
    match map.entry(10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(*view.insert(1000), 1000);
        }
    }
    assert_eq!(map.get(&10).unwrap(), &1000);
    assert_eq!(map.len(), 6);
}

#[test]
fn test_entry_take_doesnt_corrupt() {
    #![allow(deprecated)] //rand
    // Test for #19292
    fn check(m: &HashMap<isize, ()>) {
        for k in m.keys() {
            assert!(m.contains_key(k),
                    "{} is in keys() but not in the map?", k);
        }
    }

    let mut m = HashMap::new();
    let mut rng = Prng::new();

    // Populate the map with some items.
    for _ in 0..50 {
        let x = -10 + (rng.next_f64() * 20f64) as isize;
        m.insert(x, ());
    }

    for _ in 0..1000 {
        let x = -10 + (rng.next_f64() * 20f64) as isize;
        match m.entry(x) {
            Vacant(_) => {}
            Occupied(e) => {
                //println!("{}: remove {}", i, x);
                e.remove();
            }
        }

        check(&m);
    }
}

#[test]
fn test_extend_ref() {
    let mut a = HashMap::new();
    a.insert(1, "one");
    let mut b = HashMap::new();
    b.insert(2, "two");
    b.insert(3, "three");

    a.extend(&b);

    assert_eq!(a.len(), 3);
    assert_eq!(a[&1], "one");
    assert_eq!(a[&2], "two");
    assert_eq!(a[&3], "three");
}

#[test]
fn test_capacity_not_less_than_len() {
    let mut a = HashMap::new();
    let mut item = 0;

    for _ in 0..116 {
        a.insert(item, 0);
        item += 1;
    }

    assert!(a.capacity() > a.len());

    let free = a.capacity() - a.len();
    for _ in 0..free {
        a.insert(item, 0);
        item += 1;
    }

    assert_eq!(a.len(), a.capacity());

    // Insert at capacity should cause allocation.
    a.insert(item, 0);
    assert!(a.capacity() > a.len());
}

#[test]
fn test_occupied_entry_key() {
    let mut a = HashMap::new();
    let key = "hello there";
    let value = "value goes here";
    assert!(a.is_empty());
    a.insert(key.clone(), value.clone());
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);

    match a.entry(key.clone()) {
        Vacant(_) => panic!(),
        Occupied(e) => assert_eq!(key, *e.key()),
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
}

#[test]
fn test_vacant_entry_key() {
    let mut a = HashMap::new();
    let key = "hello there";
    let value = "value goes here";

    assert!(a.is_empty());
    match a.entry(key.clone()) {
        Occupied(_) => panic!(),
        Vacant(e) => {
            assert_eq!(key, *e.key());
            e.insert(value.clone());
        }
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
}

#[test]
fn test_retain() {
    let mut map: HashMap<isize, isize> = (0..100).map(|x|(x, x*10)).collect();

    map.retain(|&k, _| k % 2 == 0);
    assert_eq!(map.len(), 50);
    assert_eq!(map[&2], 20);
    assert_eq!(map[&4], 40);
    assert_eq!(map[&6], 60);
}

#[test]
fn test_adaptive() {
    const TEST_LEN: usize = 5000;
    // by cloning we get maps with the same hasher seed
    let mut first = HashMap::new();
    let mut second = first.clone();
    first.extend((0..TEST_LEN).map(|i| (i, i)));
    second.extend((TEST_LEN..TEST_LEN * 2).map(|i| (i, i)));

    for (&k, &v) in &second {
        let prev_cap = first.capacity();
        let expect_grow = first.len() == prev_cap;
        first.insert(k, v);
        if !expect_grow && first.capacity() != prev_cap {
            return;
        }
    }
    panic!("Adaptive early resize failed");
}
