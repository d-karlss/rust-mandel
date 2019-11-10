extern crate image;

#[cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx")]
unsafe fn unpack(pd: __m256d) -> (f64, f64, f64, f64) {
    let a = _mm256_extractf128_pd(pd,0);
    let b = _mm256_extractf128_pd(pd,1);
    let la =  _mm_cvtsd_f64 (a);
    let lb =  _mm_cvtsd_f64 (b);
    let _ha = _mm_unpackhi_pd(a,a);
    let _hb = _mm_unpackhi_pd(b,b);
    let ha =  _mm_cvtsd_f64 (_ha);
    let hb =  _mm_cvtsd_f64 (_hb);
    return (hb, lb, ha, la)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn iter(cx: __m256d, cy: __m256d) -> (f64, f64, f64, f64) {

    let mut done_mask = _mm256_setzero_pd();
    let mut a = _mm256_setzero_pd();
    let mut b = _mm256_setzero_pd();
    let mut masked_count = _mm256_setzero_pd();
    let norm_lim = _mm256_set1_pd(4f64);
    let two = _mm256_set1_pd(2f64);
    let all_ones = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
    
    for icounter in 0..256 {
        let asq = _mm256_mul_pd(a, a);
        let bsq = _mm256_mul_pd(b, b);
        let _ab = _mm256_mul_pd(a, b);
        let _2ab = _mm256_mul_pd(two, _ab);
        b = _mm256_add_pd(_2ab, cy);
        let tmp = _mm256_sub_pd(asq, bsq);
        a = _mm256_add_pd(tmp, cx);
        let norm = _mm256_add_pd(asq, bsq);              
        let all_gt_mask = _mm256_cmp_pd(norm_lim, norm, 2);             // (4, 4, 4, 4) < (0, 6, 20, 1) --> 0x00.., 0xFF..,0xFF, 0x00 
        let not_done_gt_mask = _mm256_xor_pd(done_mask, all_gt_mask);   // (0x00.., 0x00..,0xff..,0x00) XOR (0x00..,0xFF..,0xFF,0x00) -> 0x00,0xff,0x00,0x00
        let pd_counter = _mm256_set1_pd(icounter as f64);               // 10,10,10,10 (if at 10th iteration)
        let masked_count2 = _mm256_and_pd(not_done_gt_mask, pd_counter);// (0x00.., 0xff.., 0x00.., 0x00..) AND (10,10,10,10) -> (0, 10, 0, 0)
        masked_count = _mm256_or_pd(masked_count2, masked_count);       // (0, 10, 0, 0) OR (0, 0, 9, 0) -> (0, 10, 9, 0) 
        done_mask = _mm256_or_pd(not_done_gt_mask, done_mask);          // (0x00.,0xFF.,0x00, 0x00) OR (0x00.., 0x00..,0xFF..,0x00) -> (0x00,0xFF,0xFF,0x00)
        let res = _mm256_testc_pd(done_mask, all_ones);                 // (0x00., 0xFF, 0xFF, 0x00) != (0xFF..,0xFF..,0xFF., 0xFF..) -> res = 0

        if res == 1 {
            break
        }
    }
    return unpack(masked_count);
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn run(cxmin: f64, cymin: f64, scalex: f64, scaley: f64, x: f64, y: f64) -> (f64, f64, f64, f64) {
    let cx1 = cxmin + x as f64 * scalex;
    let cx2 = cxmin + (x + 1.0) as f64 * scalex;
    let cx3 = cxmin + (x + 2.0) as f64 * scalex;
    let cx4 = cxmin + (x + 3.0) as f64 * scalex;

    let cy1 = cymin + y as f64 * scaley;
    let cy2 = cymin + y as f64 * scaley;
    let cy3 = cymin + y as f64 * scaley;
    let cy4 = cymin + y as f64 * scaley;

    let cx256 = _mm256_set_pd(cx1, cx2, cx3, cx4);
    let cy256 = _mm256_set_pd(cy1, cy2, cy3, cy4);

    return iter(cx256, cy256);
}

fn main() {
    unsafe {
        unsafe_main();
    }
}

#[target_feature(enable = "avx")]
unsafe fn unsafe_main() {
    const IMG_SIDE : u32 = 1024u32;
    const CXMIN : f64 = -2f64;
    const CXMAX :f64 = 1f64;
    const CYMIN :f64 = -1.5f64;
    const CYMAX :f64 = 1.5f64;
    const SCALEX :f64 = (CXMAX - CXMIN) / (IMG_SIDE as f64);
    const SCALEY :f64 = (CYMAX - CYMIN) / (IMG_SIDE as f64);
    const X_LIMIT : u32 = (IMG_SIDE - 4);
    const Y_LIMIT : u32 = (IMG_SIDE - 1);
    let mut imgbuf = image::ImageBuffer::new(IMG_SIDE, IMG_SIDE);

    for x in (0..X_LIMIT).step_by(4) {
        for y in 0..Y_LIMIT {
            let (a, b, c, d) = run(CXMIN, CYMIN, SCALEX, SCALEY, x as f64, y as f64);
            let p1 = imgbuf.get_pixel_mut(x, y);
            *p1 = image::Luma([(15.0*a) as u8]);
            let p2 = imgbuf.get_pixel_mut(x + 1, y);
            *p2 = image::Luma([(15.0*b) as u8]);
            let p3 = imgbuf.get_pixel_mut(x + 2, y);
            *p3 = image::Luma([(15.0*c) as u8]);
            let p4 = imgbuf.get_pixel_mut(x + 3, y);
            *p4 = image::Luma([(15.0*d) as u8]);
        }
    }

    let _ = imgbuf.save("out.png");
}


