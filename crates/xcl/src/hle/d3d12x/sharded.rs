//! A map by the address of an object, in parts with a lock each.
//!
//! A title records its command lists on many threads at once, and each recorded call looks up what
//! XCL keeps of the list and of its pipeline. One lock for a whole map has those threads wait for
//! each other: with 8 of them recording, a dispatch took 5.7 µs instead of 0.6 µs.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Mutex, MutexGuard};

const SHARDS: usize = 64;

/// Addresses are spread by `spread` already.
#[derive(Default)]
pub(super) struct Spread(u64);

impl Hasher for Spread {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = spread((self.0 ^ b as u64) as usize);
        }
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = spread(i);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Which of `n` parts an address is in.
pub(super) fn part(address: usize, n: usize) -> usize {
    (spread(address) >> 40) as usize % n
}

fn spread(address: usize) -> u64 {
    let x = (address as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^ (x >> 29)
}

type Map<V> = HashMap<usize, V, BuildHasherDefault<Spread>>;

/// (a cache line each: threads at different objects do not write the same one)
#[repr(align(64))]
struct Shard<V>(Mutex<Option<Map<V>>>);

pub(super) struct Sharded<V>([Shard<V>; SHARDS]);

impl<V> Sharded<V> {
    pub const fn new() -> Self {
        Sharded([const { Shard(Mutex::new(None)) }; SHARDS])
    }

    fn shard(&self, key: usize) -> MutexGuard<'_, Option<Map<V>>> {
        self.0[part(key, SHARDS)].0.lock().unwrap()
    }

    pub fn insert(&self, key: usize, value: V) -> Option<V> {
        self.shard(key).get_or_insert_with(Default::default).insert(key, value)
    }

    pub fn remove(&self, key: usize) -> Option<V> {
        self.shard(key).as_mut()?.remove(&key)
    }

    pub fn contains(&self, key: usize) -> bool {
        self.shard(key).as_ref().is_some_and(|m| m.contains_key(&key))
    }

    pub fn get(&self, key: usize) -> Option<V>
    where
        V: Clone,
    {
        self.shard(key).as_ref()?.get(&key).cloned()
    }

    /// `f` with the value of `key` (made when there is none), with its part's lock held: `f` looks
    /// up nothing else here.
    pub fn with<R>(&self, key: usize, f: impl FnOnce(&mut V) -> R) -> R
    where
        V: Default,
    {
        f(self.shard(key).get_or_insert_with(Default::default).entry(key).or_default())
    }

    /// Every value, one part after another.
    pub fn for_each(&self, mut f: impl FnMut(usize, &V)) {
        for s in &self.0 {
            if let Some(m) = s.0.lock().unwrap().as_ref() {
                m.iter().for_each(|(&k, v)| f(k, v));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_stay_with_their_keys() {
        static M: Sharded<u64> = Sharded::new();
        // addresses as objects have them: 16 bytes to 64 KiB apart
        let keys: Vec<usize> = (0..4000).map(|k| 0x7FF6_1234_0000 + k * [16, 0x330, 0x1_0000][k % 3] + k).collect();
        for (n, &k) in keys.iter().enumerate() {
            assert_eq!(M.insert(k, n as u64), None);
        }
        for (n, &k) in keys.iter().enumerate() {
            assert!(M.contains(k) && M.get(k) == Some(n as u64));
            assert_eq!(M.with(k, |v| std::mem::replace(v, *v + 1)), n as u64);
        }
        let mut seen = 0;
        M.for_each(|_, _| seen += 1);
        assert_eq!(seen, keys.len());
        // (no part has much more than its share)
        let most = M.0.iter().map(|s| s.0.lock().unwrap().as_ref().map_or(0, |m| m.len())).max().unwrap();
        assert!(most < keys.len() / SHARDS * 3, "{most} of {} in one part", keys.len());
        for (n, &k) in keys.iter().enumerate() {
            assert_eq!(M.remove(k), Some(n as u64 + 1));
            assert!(!M.contains(k) && M.remove(k).is_none());
        }
        assert_eq!(M.with(5, |v| *v), 0, "made when there is none");
    }
}
