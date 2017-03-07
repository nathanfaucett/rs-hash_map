#![feature(test)]


extern crate test;

extern crate hash_map;
extern crate collection_traits;


use test::Bencher;

use collection_traits::*;


const SIZE: usize = 1024;


#[bench]
fn bench_hash_map(b: &mut Bencher) {
    use hash_map::HashMap;

    b.iter(|| {
        let mut v = HashMap::new();
        for i in 0..SIZE {
            v.insert(i, i);
        }
        v
    });
}
#[bench]
fn bench_std_hash_map(b: &mut Bencher) {
    use std::collections::HashMap;

    b.iter(|| {
        let mut v = HashMap::new();
        for i in 0..SIZE {
            v.insert(i, i);
        }
        v
    });
}
