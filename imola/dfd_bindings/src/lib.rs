extern crate dfd;


use std::slice;
use dfd::dfd_euclidean;


#[no_mangle]
pub unsafe extern "C" fn dfd(
    p: *const f64,
    q: *const f64,
    len_p: usize,
    len_q: usize,
    num_dim: usize,
) -> f64 {
    assert!(!p.is_null());
    assert!(!q.is_null());
    dfd_euclidean(
        slice::from_raw_parts(p, len_p),
        slice::from_raw_parts(q, len_q),
        num_dim,
    )
}
