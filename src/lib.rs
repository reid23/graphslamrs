#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
use pyo3::prelude::*;
use itertools::*;
// use std::{cmp::Ordering, process::exit};
use finitediff::FiniteDiff;
use nalgebra as na;
use std::ptr::{addr_of_mut, null_mut};
use std::iter;
use std::collections::HashMap;
use std::time::Instant;

use core::f64;
// include!("bindings.rs");


const control: ma97_control_d = ma97_control_d { 
    f_arrays: 0, 
    action: 1, 
    nemin: 32,  // default 8
    multiplier: 1.1, // no affect
    ordering: 1, // 1 is fastest
    print_level: 0, 
    scaling: 0, // 0 (no scaling) fastest
    small: 1e-20, 
    u: 0.01, //no affect
    unit_diagnostics: 6, 
    unit_error: 6,
    unit_warning: 6,
    // factor_min: 20000000,
    factor_min: 200000,
    solve_blas3: 0, // no affect
    solve_min: 10000,
    solve_mf: 0, // no affect
    consist_tol: 2.220446049250313e-16,
    ispare: [0i32; 5], 
    rspare: [0.0f64; 10] 
};

static mut info: ma97_info_d = ma97_info_d {
    flag68: 0,
    flag: 0,
    flag77: 0,
    matrix_dup: 0,
    matrix_rank: 0,
    matrix_outrange: 0,
    matrix_missing_diag: 0,
    maxdepth: 0,
    maxfront: 0,
    num_delay: 0,
    num_factor: 0,
    num_flops: 0,
    num_neg: 0,
    num_sup: 0,
    num_two: 0,
    ordering: 0,
    stat: 0,
    maxsupernode: 0,
    ispare: [0i32; 4],
    rspare: [0.0f64; 10],
};

#[derive(Debug)]
pub struct Trimat {
    rows: Vec<i32>,
    cols: Vec<i32>,
    vals: Vec<f64>,
    nrows: i32,
    ncols: i32,
}
impl Trimat {
    pub fn new() -> Self {
        Trimat {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            nrows: 0,
            ncols: 0,
        }
    }
    pub fn add_triplet(&mut self, r: usize, c: usize, v: f64) {
        self.rows.push(r as i32);
        self.cols.push(c as i32);
        self.vals.push(v);
        if r as i32 >= self.nrows { self.nrows = r as i32 + 1; }
        if c as i32 >= self.ncols { self.ncols = c as i32 + 1; }
    }
    pub fn rows(&self) -> &Vec<i32> { &self.rows }
    pub fn cols(&self) -> &Vec<i32> { &self.cols }
    pub fn vals(&self) -> &Vec<f64> { &self.vals }
}

#[pyclass]
#[pyo3(name="GraphSLAMSolve")]
#[derive(Debug)]
pub struct GraphSLAMSolve {
    pub max_landmark_distance: f64,
    pub dx_weight: f64,
    pub z_weight: f64,
    pub dclip: HashMap<u8, f64>,
    pub max_newton_steps: usize,
    pub newton_solve_tol_sr: f64,

    pub A: Trimat,
    pub b: Vec<f64>,

    pub x: Vec<usize>,
    pub l: Vec<usize>,
    pub z: Vec<usize>,
    pub d: Vec<usize>,

    pub nvars: usize,
    pub neqns: usize,

    pub xhat: Vec<[f64; 2]>,
    pub lhat: Vec<[f64; 2]>,
    pub color: Vec<u8>,
}

/// utility to rotate and translate a set of points
fn transform(cones: &Vec<[f64; 2]>, x: &Vec<f64>) ->  Vec<[f64; 2]> {
    let (s, c) = x[2].sin_cos();
    // println!("transform: x: {:?}, s: {}, c: {}", x, s, c);
    cones.iter().map(|z| [z[0]*c - z[1]*s + x[0], 
                                     z[0]*s + z[1]*c + x[1]]).collect()
}

// this has to be separate because it's not a #[pymethods] thing
impl GraphSLAMSolve {
    fn data_association(&mut self, x0: &Vec<f64>, cone_measurements: &Vec<[f64; 2]>, color: &Vec<u8>) -> (Vec<f64>, Vec<[f64; 2]>) {
        if self.lhat.len() == 0 { return (x0.clone(), cone_measurements.clone()); }
        
        let mut x = vec![x0[0], x0[1], 0.0];
        let cost = | x: &Vec<f64> | -> f64 {
            transform(cone_measurements, x).iter()
            .zip(color)
            .map(|cone| (cone.1, self.lhat.iter()
                .zip(&self.color)
                .filter(|x| cone.1 == x.1)
                .map(|x| x.0)
                .map(|x| (x[0]-cone.0[0]).powi(2) + (x[1]-cone.0[1]).powi(2))
                .min_by(|a, b| {a.partial_cmp(b).unwrap()})))
            .map(|x| (x.0, x.1.unwrap_or(0.0)))
            .map(|x| x.1.clamp(0.0, self.dclip.get(x.0).unwrap().powi(2)))
            .sum()
        };
        for _ in 0..self.max_newton_steps {
            let grad = x.central_diff(&cost);
            // first check if grad = 0. if it is, we can just be done
            if grad.iter().map(|x| x.powi(2)).sum::<f64>() < self.newton_solve_tol_sr { break; }

            // otherwise, compute the hessian and put both it and the gradient into
            // Matrix objects so we can do linear algebra on them
            let grad = na::Matrix3x1::from_vec(grad);
            let hess= x.forward_hessian_nograd(&cost);
            
            // technically writing the args out like this is misleading,
            // since the data is stored and entered in a column-major format,
            // but it doesn't matter since the hessian is symmetric
            let hess = na::Matrix3::new(
                hess[0][0], hess[0][1], hess[0][2], 
                hess[1][0], hess[1][1], hess[1][2], 
                hess[2][0], hess[2][1], hess[2][2],
            );
            if let Some(inv) = hess.try_inverse() {
                let update = inv*grad;
                x[0] -= update[0];
                x[1] -= update[1];
                x[2] -= update[2];
            } else { break; }
        }
        return (x.clone(), transform(&cone_measurements, &x));
    }
}


#[pymethods]
impl GraphSLAMSolve {
    #[new]
    /// initialize GraphSLAMFast object
    ///
    /// Args:
    ///     x0 ((float, float), optional): tuple (x, y) of initial state. Defaults to (0.0, 0.0).
    ///     max_landmark_distance (float, optional): how far away landmarks can be from the closest landmark guess (AFTER data association optimally translates and rotates the landmarks) before they are recognized as independent. Defaults to 0.5.
    ///     dx_weight (float, optional): weight (certainty) for odometry measurements. Defaults to 1.0.
    ///     z_weight (float, optional): weight (certainty) for landmark measurements. Defaults to 5.0.
    ///     dclip (dict[int, float], optional): distance at which to clip cost function for data association. one entry in dictionary per possible landmark color. Defaults to {0: 0.2, 1: 0.2, 2: 10.0}.
    ///     max_newton_steps (int, optional): max number of steps of newton's method to take when optimizing during data association. Defaults to 15.
    ///     newton_solve_tol (float, optional): magnitude of gradient under which to consider data association optimization finished. Defaults to 1e-3.
    #[pyo3(signature=(x0=[0.0, 0.0], max_landmark_distance=0.5, dx_weight=1.0, z_weight=5.0, dclip=HashMap::from([(0u8, 0.2), (1u8, 0.2), (2u8, 10.0)]), max_newton_steps=15, newton_solve_tol=1e-3), text_signature="(x0: list[float] = [0.0, 0.0], max_landmark_distance: float = 0.5, dx_weight: float = 1.0, z_weight: float = 5.0, dclip: dict[int, float] = {0: 0.2, 1: 0.2, 2: 10.0}, max_newton_steps: int = 15, newton_solve_tol: float = 1e-3)")]
    pub fn new(
        x0: [f64; 2],
        max_landmark_distance: f64,
        dx_weight: f64,
        z_weight: f64,
        dclip: HashMap<u8, f64>,
        max_newton_steps: usize,
        newton_solve_tol: f64,
    ) -> Self {
        let x0 = [x0[0], x0[1]];
        let mut A = Trimat::new();
        A.add_triplet(0, 0, 1.0);
        A.add_triplet(1, 1, 1.0);

        GraphSLAMSolve {
            max_landmark_distance: max_landmark_distance,
            dx_weight: dx_weight,
            z_weight: z_weight,
            dclip: dclip,
            max_newton_steps: max_newton_steps,
            newton_solve_tol_sr: newton_solve_tol.powi(2),

            A: A,
            b: x0.to_vec(),

            x: vec![0],
            l: vec![],
            z: vec![],
            d: vec![0],

            neqns: 2,
            nvars: 2,

            xhat: vec![x0],
            lhat: vec![],
            color: vec![],
        }
    }
    
    /// estimate current pose by matching vision data with the map.
    /// Args:
    ///     xhat (ndarray or list): estimated current pose of the car. [x, y, theta]
    ///     z (ndarray or list[list]): measurements to landmarks in CAR FRAME. [[zx1, zy1], [zx2, zy2], ..., [zxn, zyn]]
    ///     color (ndarray or list): categorical array of which color each of the measurements are. Elements should be dtype=np.uint8 or python ints.
    pub fn pose_from_data_association(&mut self, xhat: [f64; 3], z: Vec<[f64; 2]>, color: Vec<u8>) -> Vec<f64> {
        // we've got to negate the theta (heading) coordinate, since data association is working 
        // with the angle of the cones (vectors => contravariant) while we're working with the angle 
        // of the car frame (bases => covariant)
        let (x, _) = self.data_association(&vec![xhat[0], xhat[1], -xhat[2]], &z, &color);
        return vec![x[0], x[1], -x[2]];
    }

    /// add edges to the graph corresponding to a movement and new vision data
    ///
    /// Args:
    ///     dx (ndarray or list): difference in position from last update. [dx, dy]
    ///     z (ndarray or list[list]): landmark locations (global frame) minus car location (global frame). [[zx1, zy1], [zx2, zy2], ..., [zxn, zyn]]
    ///     color (ndarray or list): categorical array of which color each of the measurements are. Elements should be dtype=np.uint8 or python ints.
    ///     run_data_association (bool, optional): whether or not to run the iterative alignment algorithm to match the observed cones to known ones.
    #[pyo3(signature = (dx, z, color, run_data_association=true))]
    pub fn update_graph(&mut self, dx: [f64; 2], z: Vec<[f64; 2]>, color: Vec<u8>, run_data_association: bool) {

        // first, add two equations and two variables
        // for the next position and corresponding dx
        self.x.push(self.nvars);
        self.d.push(self.neqns);
        self.nvars += 2;
        self.neqns += 2;

        // Add equations for car position update
        let d_end_idx = self.d.len()-1;
        let x_end_idx = self.x.len()-1;
        self.A.add_triplet(self.d[d_end_idx],     self.x[x_end_idx  ],      self.dx_weight);
        self.A.add_triplet(self.d[d_end_idx] + 1, self.x[x_end_idx  ] + 1,  self.dx_weight);
        self.A.add_triplet(self.d[d_end_idx],     self.x[x_end_idx-1],     -self.dx_weight);
        self.A.add_triplet(self.d[d_end_idx] + 1, self.x[x_end_idx-1] + 1, -self.dx_weight);
        // println!("{:?}, {:?}", self.d[d_end_idx], self.x[x_end_idx]);
        self.b.push(dx[0] * self.dx_weight);
        self.b.push(dx[1] * self.dx_weight);

        let prev_xhat = self.xhat.last().unwrap();
        self.xhat.push([prev_xhat[0]+dx[0], prev_xhat[1]+dx[1]]);


        let cur_xhat = self.xhat.last().unwrap();
        
        // Data association! tries to rotate and translate our measured cones
        // to optimally line them up with the cones we've seen already.
        let zprime = if run_data_association {
            self.data_association(&vec![cur_xhat[0], cur_xhat[1], 0.], &z, &color).1
        } else {
            z.iter().map(|x| [x[0]+cur_xhat[0], x[1]+cur_xhat[1]]).collect()
        };

        // match the cones!
        // for each cone we see, find the closest existing cone, and if that cone is within
        // `self.max_landmark_distance`, call them the same cone.
        // otherwise, make a new cone.
        for (idx, (cone, c)) in iter::zip(zprime, color).enumerate() {
            // get closest known cone of same color
            // find it's index in `lhat` and `l`
            let (mut l_idx, min_dist) = match self.lhat.iter()
                .zip(&self.color)
                .enumerate()
                .filter(|x| *(x.1).1==c)
                .map(|x| (x.0, (x.1.0[0] - cone[0]).powi(2) + (x.1.0[1] - cone[1]).powi(2)))
                .min_by(|a, b| (&a.1).partial_cmp(&b.1).unwrap()) {
                    Some(data) => data,
                    None => (0usize, f64::INFINITY), // if there were no cones with the same color
            };
            // add landmark if it's new
            if min_dist > self.max_landmark_distance {
                self.l.push(self.nvars);
                self.nvars += 2;
                self.lhat.push(cone);
                self.color.push(c);
                l_idx = self.lhat.len()-1;
            }

            // add equations corresponding to sight of landmark
            self.z.push(self.neqns);
            self.neqns += 2;
            self.A.add_triplet(*self.z.last().unwrap(),   self.l[l_idx],             self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap()+1, self.l[l_idx]+1,           self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap(),   *self.x.last().unwrap(),   -self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap()+1, *self.x.last().unwrap()+1, -self.z_weight);
            self.b.push(z[idx][0]*self.z_weight);
            self.b.push(z[idx][1]*self.z_weight);
        }
    }

    /// print a visual representation of the A matrix
    pub unsafe fn printmat(&mut self) {
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        // let mut b = self.b.clone();
        let a = cs_sparse { 
            nzmax:ra.len() as i32, 
            m: self.A.nrows, 
            n: self.A.ncols, 
            p: ca.as_mut_ptr(),
            i: ra.as_mut_ptr(),
            x: va.as_mut_ptr(), 
            nz: ra.len() as i32
        };
        let c: *const cs = cs_triplet(&a);
        cs_print(c, 0);
    }

    /// print this GraphSLAMFast using the #Debug trait
    pub fn printself(&mut self) {
        println!("{:?}", self);
    }

    // getters
    
    /// get list of row indices in the A matrix where the values are
    pub fn rows(&mut self) -> Vec<i32> { self.A.rows().clone() }
    /// get list of col indices in the A matrix where the values are
    pub fn cols(&mut self) -> Vec<i32> { self.A.cols().clone() }
    /// get list of nonzero values in the A matrix corresponding to the indices from rows() and cols()
    pub fn vals(&mut self) -> Vec<f64> { self.A.vals().clone() }
    /// get number of rows in the A matrix
    pub fn nrows(&mut self) -> i32 { self.A.nrows }
    /// get number of columns in the A matrix
    pub fn ncols(&mut self) -> i32 { self.A.ncols }
    /// get the number of equations
    pub fn neqns(&mut self) -> usize { self.neqns }
    /// get the number of variables
    pub fn nvars(&mut self) -> usize { self.nvars }

    /// solve the graph using the `qrsol` algorithm.
    /// This just solves the least squares problem directly,
    /// which is often slower than solving the PSD eqn A.T@A \ A.T@b
    /// solves in-place; does not return results.
    pub fn solve_graph_qrsol(&mut self) {
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        let mut b = self.b.clone();
        // let mut atb = vec![0.0; self.b.len()];

        unsafe {
            let a = cs_sparse { 
                nzmax:ra.len() as i32, 
                m: self.A.nrows, 
                n: self.A.ncols, 
                p: ca.as_mut_ptr(),
                i: ra.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };

            let c: *const cs = cs_triplet(&a);

            cs_qrsol(c, b.as_mut_ptr(), 0);
        }

        for (idx, i) in enumerate(&self.x) {
            self.xhat[idx].copy_from_slice(&b[*i..(i+2)]);
        }
        for (idx, i) in enumerate(&self.l) {
            self.lhat[idx].copy_from_slice(&b[*i..(i+2)]);
        }

    }
    /// solve the graph using the cholesky decomposition of A.T@A
    /// this is the fastest method provided here.
    /// solves in-place; does not return results.
    pub fn solve_graph(&mut self) {
        let tic = Instant::now();
        // use cholesky decomposition to solve A.T@A \ A.T@b
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        // let mut b = self.b.clone();
        let mut atb = vec![0.0; self.A.ncols as usize];
        // println!("ra: {:?}\nca: {:?}\nva: {:?}\nb: {:?}", &ra, &ca, &va, &b);
        // println!("m: {:?}\nn: {:?}", &self.A.nrows, &self.A.ncols);
        let dt = tic.elapsed();
        println!("initialized in {:?}", dt);
        unsafe {
            let tic = Instant::now();
            let a = cs_sparse { 
                nzmax:ra.len() as i32, 
                m: self.A.nrows, 
                n: self.A.ncols, 
                p: ca.as_mut_ptr(),
                i: ra.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let at = cs_sparse {
                nzmax:ra.len() as i32, 
                m: self.A.ncols, 
                n: self.A.nrows, 
                p: ra.as_mut_ptr(),
                i: ca.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let dt = tic.elapsed();
            println!("loaded into csparse in {:?}", dt);
            let tic = Instant::now();
            let a2: *const cs = cs_triplet(&a);
            let at2: *const cs = cs_triplet(&at);
            let dt = tic.elapsed();
            println!("converted from triplet in {:?}", dt);
            let tic = Instant::now();
            let ata: *const cs = cs_multiply(at2, a2);
            let dt = tic.elapsed();
            println!("multiplied a.T@a in {:?}", dt);
            
            let tic = Instant::now();
            let _ = cs_gaxpy(at2, self.b.as_ptr(), atb.as_mut_ptr());
            let dt = tic.elapsed();
            println!("multiplied a.T@b in {:?}", dt);
            
            let tic = Instant::now();
            cs_cholsol(ata, atb.as_mut_ptr(), 0);
            let dt = tic.elapsed();
            println!("completed solve in {:?}", dt);
            // cs_lusol(ata, atb.as_mut_ptr(), 0, 1e-14);
        }
        // println!("{:?}", b[0..(self.A.ncols as usize)].to_vec());

        let tic = Instant::now();
        for (idx, i) in enumerate(&self.x) {
            self.xhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        for (idx, i) in enumerate(&self.l) {
            self.lhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        let dt = tic.elapsed();
        println!("copied data back in {:?}", dt);

    }


    /// solve the graph using the cholesky decomposition of A.T@A
    /// this is the fastest method provided here.
    /// solves in-place; does not return results.
    pub fn solve_graph_ma97(&mut self) {
        let tic = Instant::now();
        // use cholesky decomposition to solve A.T@A \ A.T@b
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        // let mut b = self.b.clone();
        let mut atb = vec![0.0; self.A.ncols as usize];
        // println!("ra: {:?}\nca: {:?}\nva: {:?}\nb: {:?}", &ra, &ca, &va, &b);
        // println!("m: {:?}\nn: {:?}", &self.A.nrows, &self.A.ncols);
        let dt = tic.elapsed();
        println!("initialized in {:?}", dt);
        unsafe {
            let tic = Instant::now();
            let a = cs_sparse { 
                nzmax:ra.len() as i32, 
                m: self.A.nrows, 
                n: self.A.ncols, 
                p: ca.as_mut_ptr(),
                i: ra.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let at = cs_sparse {
                nzmax:ra.len() as i32, 
                m: self.A.ncols, 
                n: self.A.nrows, 
                p: ra.as_mut_ptr(),
                i: ca.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let dt = tic.elapsed();
            println!("loaded into csparse in {:?}", dt);
            let tic = Instant::now();
            let a2: *const cs = cs_triplet(&a);
            let at2: *const cs = cs_triplet(&at);
            let dt = tic.elapsed();
            println!("converted from triplet in {:?}", dt);
            let tic = Instant::now();
            let mut ata = *cs_multiply(at2, a2);
            let dt = tic.elapsed();
            println!("multiplied a.T@a in {:?}", dt);
            
            let tic = Instant::now();
            let _ = cs_gaxpy(at2, self.b.as_ptr(), atb.as_mut_ptr());
            let dt = tic.elapsed();
            println!("multiplied a.T@b in {:?}", dt);
            
            let tic = Instant::now();
            
            let mut akeep = std::ptr::null_mut();
            let mut fkeep = std::ptr::null_mut();
            // let mut order = 5;
            let other = std::ptr::null_mut();
            
            cs_fkeep(&mut ata, Some(fkeep_u), other);
            // cs_print(&ata, 0);
            ma97_analyse_d(0, ata.n, ata.p, ata.i, ata.x, &mut akeep, &control, addr_of_mut!(info), null_mut());
            ma97_factor_solve_d(3, ata.p, ata.i, ata.x, 1, atb.as_mut_ptr(), ata.n, &mut akeep, &mut fkeep, &control, addr_of_mut!(info), null_mut());
            // println!("factor_solved");
            ma97_finalise_d(&mut akeep, &mut fkeep);
            // println!("finalized");
            
            // cs_free(a2);
            // cs_free(at2);
            // cs_free(other);
            // cs_free(((&mut ata) as *mut cs) as *mut c_void);
            // cs_cholsol(ata, atb.as_mut_ptr(), 0);
            let dt = tic.elapsed();
            println!("completed solve in {:?}", dt);
            // cs_lusol(ata, atb.as_mut_ptr(), 0, 1e-14);
        }
        // println!("{:?}", b[0..(self.A.ncols as usize)].to_vec());

        let tic = Instant::now();
        for (idx, i) in enumerate(&self.x) {
            self.xhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        for (idx, i) in enumerate(&self.l) {
            self.lhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        let dt = tic.elapsed();
        println!("copied data back in {:?}", dt);

    }

    /// solve the graph using the LU decomposition of A.T@A.
    /// This is barely slower than the cholesky method.
    /// solves in-place; does not return results.
    #[pyo3(signature = (tol=1e-14))]
    pub fn solve_graph_lusol(&mut self, tol: f64) {
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        // let mut b = self.b.clone();
        let mut atb = vec![0.0; self.A.ncols as usize];
        // println!("ra: {:?}\nca: {:?}\nva: {:?}\nb: {:?}", &ra, &ca, &va, &b);
        // println!("m: {:?}\nn: {:?}", &self.A.nrows, &self.A.ncols);
        unsafe {
            let a = cs_sparse { 
                nzmax:ra.len() as i32, 
                m: self.A.nrows, 
                n: self.A.ncols, 
                p: ca.as_mut_ptr(),
                i: ra.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let at = cs_sparse {
                nzmax:ra.len() as i32, 
                m: self.A.ncols, 
                n: self.A.nrows, 
                p: ra.as_mut_ptr(),
                i: ca.as_mut_ptr(),
                x: va.as_mut_ptr(), 
                nz: ra.len() as i32
            };
            let a2: *const cs = cs_triplet(&a);
            let at2: *const cs = cs_triplet(&at);
            let ata: *const cs = cs_multiply(at2, a2);
            let _ = cs_gaxpy(at2, self.b.as_ptr(), atb.as_mut_ptr());

            cs_lusol(ata, atb.as_mut_ptr(), 0, tol);
        }

        for (idx, i) in enumerate(&self.x) {
            self.xhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        for (idx, i) in enumerate(&self.l) {
            self.lhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }

    }

    /// get all cones, or those matching `color`
    #[pyo3(signature=(color=None))]
    pub fn get_cones(&mut self, color: Option<u8>) -> Vec<[f64; 2]> {
        match color {
            Some(c) => self.lhat.iter()
                .zip(self.color.iter())
                .filter(|x| *x.1==c)
                .map(|x| *x.0)
                .collect(),
           None => self.lhat.clone()
        }
    }
    /// get all past x positions
    pub fn get_positions(&self) -> Vec<[f64; 2]> {
        self.xhat.clone()
    }
}



/// A Python module implemented in Rust.
#[pymodule]
fn graphslamrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GraphSLAMSolve>()?;
    Ok(())
}