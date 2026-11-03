// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The launch wrapper exposes independent, typed input and result planes.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_contract(domain=3,block=(64,1,1),dynamic_shared=inferred)]
    pub fn select_tile<const ITEMS_PER_THREAD: usize>(
        keys: &[i32],
        values: &[f64],
        mut minimum_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut maximum_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut minimum_pair_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut minimum_pair_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut maximum_pair_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut maximum_pair_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        k: i64,
        valid: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (min_k, max_k, mut min_pk, mut min_pv, max_pk, max_pv, input_k, input_v) =
            coop_topk_helper::select_tile::<_, _, ITEMS_PER_THREAD, 16, 32>(
                block, keys, values, offset, k, valid,
            );
        let mut flags = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        for i in 0..ITEMS_PER_THREAD {
            flags.write(
                i,
                i32::from(input_k.get(i).is_some())
                    | (i32::from(input_v.get(i).is_some()) << 1)
                    | (i32::from(min_k.get(i).is_some()) << 2)
                    | (i32::from(max_k.get(i).is_some()) << 3)
                    | (i32::from(min_pk.get(i).is_some()) << 4)
                    | (i32::from(min_pv.get(i).is_some()) << 5)
                    | (i32::from(max_pk.get(i).is_some()) << 6)
                    | (i32::from(max_pv.get(i).is_some()) << 7),
            );
        }
        coop::store(
            block,
            &mut minimum_keys,
            &min_k,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut maximum_keys,
            &max_k,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut minimum_pair_keys,
            &min_pk,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut minimum_pair_values,
            &min_pv,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut maximum_pair_keys,
            &max_pk,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut maximum_pair_values,
            &max_pv,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        // Modify only already-valid slots of the fresh first pair. The other
        // results and both original inputs remain independently observable.
        if min_pk.get(0).is_some() {
            min_pk.write(0, 123_456_789);
            min_pv.write(0, -4096.5);
        }
        coop::store(
            block,
            &mut preserved_keys,
            &input_k,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved_values,
            &input_v,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut mutated_keys,
            &min_pk,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut mutated_values,
            &min_pv,
            coop::StoreOptions::default().offset(offset).valid_items(17),
        );
        coop::store(
            block,
            &mut validity,
            &flags,
            coop::StoreOptions::default().offset(offset),
        );
    }
}

// TopK is unsorted. Compare the selected key multiset, then require every
// returned pair to consume one original (key,value-bits) association. Equal
// threshold keys may choose any subset; stable tie selection is not assumed.
fn check_keys(input: &[i32], output: &[i32], maximum: bool) {
    let mut expected = input.to_vec();
    expected.sort_unstable();
    if maximum {
        expected.reverse();
    }
    expected.truncate(output.len());
    expected.sort_unstable();
    let mut actual = output.to_vec();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}
fn check_pairs(
    input_k: &[i32],
    input_v: &[f64],
    output_k: &[i32],
    output_v: &[f64],
    maximum: bool,
) {
    check_keys(input_k, output_k, maximum);
    let mut available = std::collections::BTreeMap::new();
    for (&k, v) in input_k.iter().zip(input_v) {
        *available.entry((k, v.to_bits())).or_insert(0_usize) += 1;
    }
    for (&k, v) in output_k.iter().zip(output_v) {
        let count = available
            .get_mut(&(k, v.to_bits()))
            .expect("output pair belongs to input");
        assert!(*count > 0, "pair multiplicity");
        *count -= 1;
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernel bytes are built together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {($items:literal)=>{{
        let tile=64*$items;let valid=tile-1;
        let keys:Vec<i32>=(0..2*tile).map(|i| if i%tile==valid {10_000} else {((i*17+i/tile*11)%23) as i32-11}).collect();
        let values:Vec<f64>=(0..keys.len()).map(|i| match i {0=>0.0,1=>-0.0,_ if i%7==0=>f64::from_bits(0x7ff8_0000_0000_0000|i as u64),_=>i as f64+0.25}).collect();
        let source_keys=DeviceBuffer::from_host(&stream,&keys)?;
        let source_values=DeviceBuffer::from_host(&stream,&values)?;
        let minimum_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let maximum_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let minimum_pair_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let minimum_pair_values_target=DeviceBuffer::from_host(&stream,&vec![-777777.25_f64;keys.len()])?;
        let maximum_pair_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let maximum_pair_values_target=DeviceBuffer::from_host(&stream,&vec![-777777.25_f64;keys.len()])?;
        let preserved_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let preserved_values_target=DeviceBuffer::from_host(&stream,&vec![-777777.25_f64;keys.len()])?;
        let mutated_keys_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let mutated_values_target=DeviceBuffer::from_host(&stream,&vec![-777777.25_f64;keys.len()])?;
        let validity_target=DeviceBuffer::from_host(&stream,&vec![-777777_i32;keys.len()])?;
        let plan=module.prepare_select_tile::<$items>(LaunchConfig3D::auto_shared((2,1,1),(64,1,1)))?;
        let (source_keys,source_values,minimum_keys_target,maximum_keys_target,minimum_pair_keys_target,minimum_pair_values_target,maximum_pair_keys_target,maximum_pair_values_target,preserved_keys_target,preserved_values_target,mutated_keys_target,mutated_values_target,validity_target)=module.select_tile_async_owned::<$items,_,_,_,_,_,_,_,_,_,_,_,_,_>( &plan,source_keys,source_values,minimum_keys_target,maximum_keys_target,minimum_pair_keys_target,minimum_pair_values_target,maximum_pair_keys_target,maximum_pair_values_target,preserved_keys_target,preserved_values_target,mutated_keys_target,mutated_values_target,validity_target,17,valid as i64).sync()?;
        let mut minimum_keys=vec![0_i32;keys.len()];minimum_keys_target.copy_to_host(&stream,&mut minimum_keys)?;
        let mut maximum_keys=vec![0_i32;keys.len()];maximum_keys_target.copy_to_host(&stream,&mut maximum_keys)?;
        let mut minimum_pair_keys=vec![0_i32;keys.len()];minimum_pair_keys_target.copy_to_host(&stream,&mut minimum_pair_keys)?;
        let mut minimum_pair_values=vec![0.0_f64;keys.len()];minimum_pair_values_target.copy_to_host(&stream,&mut minimum_pair_values)?;
        let mut maximum_pair_keys=vec![0_i32;keys.len()];maximum_pair_keys_target.copy_to_host(&stream,&mut maximum_pair_keys)?;
        let mut maximum_pair_values=vec![0.0_f64;keys.len()];maximum_pair_values_target.copy_to_host(&stream,&mut maximum_pair_values)?;
        let mut preserved_keys=vec![0_i32;keys.len()];preserved_keys_target.copy_to_host(&stream,&mut preserved_keys)?;
        let mut preserved_values=vec![0.0_f64;keys.len()];preserved_values_target.copy_to_host(&stream,&mut preserved_values)?;
        let mut mutated_keys=vec![0_i32;keys.len()];mutated_keys_target.copy_to_host(&stream,&mut mutated_keys)?;
        let mut mutated_values=vec![0.0_f64;keys.len()];mutated_values_target.copy_to_host(&stream,&mut mutated_values)?;
        let mut validity=vec![0_i32;keys.len()];validity_target.copy_to_host(&stream,&mut validity)?;
        let mut read_keys=vec![0_i32;keys.len()];source_keys.copy_to_host(&stream,&mut read_keys)?;
        let mut read_values=vec![0.0_f64;keys.len()];source_values.copy_to_host(&stream,&mut read_values)?;
        stream.synchronize()?;
        assert_eq!(read_keys,keys);assert_eq!(preserved_keys,keys);
        for i in 0..keys.len() {
            assert_eq!(read_values[i].to_bits(),values[i].to_bits());
            assert_eq!(preserved_values[i].to_bits(),values[i].to_bits());
            let local=i%tile;assert_eq!(validity[i],if local<17 {255} else {3});
            if local<17 {
                assert_eq!(mutated_keys[i],if local.is_multiple_of($items) {123_456_789} else {minimum_pair_keys[i]});
                assert_eq!(mutated_values[i].to_bits(),if local.is_multiple_of($items) {(-4096.5_f64).to_bits()} else {minimum_pair_values[i].to_bits()});
            } else {
                assert_eq!(minimum_keys[i],-777777);
                assert_eq!(maximum_keys[i],-777777);
                assert_eq!(minimum_pair_keys[i],-777777);
                assert_eq!(minimum_pair_values[i].to_bits(),(-777777.25_f64).to_bits());
                assert_eq!(maximum_pair_keys[i],-777777);
                assert_eq!(maximum_pair_values[i].to_bits(),(-777777.25_f64).to_bits());
                assert_eq!(mutated_keys[i],-777777);
                assert_eq!(mutated_values[i].to_bits(),(-777777.25_f64).to_bits());
            }
        }
        for block in 0..2 {
            let start=block*tile;let selected=start..start+17;
            check_keys(&keys[start..start+tile],&minimum_keys[selected.clone()],false);
            check_keys(&keys[start..start+valid],&maximum_keys[selected.clone()],true);
            check_pairs(&keys[start..start+31],&values[start..start+31],&minimum_pair_keys[selected.clone()],&minimum_pair_values[selected.clone()],false);
            check_pairs(&keys[start..start+valid],&values[start..start+valid],&maximum_pair_keys[selected.clone()],&maximum_pair_values[selected],true);
        }
        println!("N={}: 204 selected values, {} preserved, {} source, 68 mutated, {} validity, {} untouched suffix slots",$items,4*tile,4*tile,2*tile,16*(tile-17));
    }};}
    run!(1);
    run!(4);
    Ok(())
}
