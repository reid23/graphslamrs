use pyo3::prelude::*;
use itertools::*;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


#[pyfunction]
unsafe fn sparse_qr_solve(
    n: i32, m: i32,
    ra: Vec<i32>, ca: Vec<i32>, va: Vec<f64>,
    b: Vec<f64>,
) -> Vec<f64> {
    let mut rac = ra.clone();
    let mut cac = ca.clone();
    let mut vac = va.clone();
    let mut bc = b.clone();
    let a = cs_sparse { 
        nzmax:rac.len() as i32, 
        m: m, 
        n: n, 
        p: cac.as_mut_ptr(),
        i: rac.as_mut_ptr(),
        x: vac.as_mut_ptr(), 
        nz: rac.len() as i32
    };
    let c: *const cs = cs_triplet(&a);
    cs_qrsol(c, bc.as_mut_ptr(), 0);
    bc[0..(n as usize)].to_vec()
}

use core::{f64, slice};
use std::f64::consts::PI;

#[derive(Debug)]
struct Trimat {
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
        if (r as i32 >= self.nrows) { self.nrows = r as i32 + 1; }
        if (c as i32 >= self.ncols) { self.ncols = c as i32 + 1; }
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
    pub dclip: f64,

    pub A: Trimat,
    pub b: Vec<f64>,

    pub x: Vec<usize>,
    pub l: Vec<usize>,
    pub z: Vec<usize>,
    pub d: Vec<usize>,

    pub nvars: usize,
    pub neqns: usize,

    pub xhat: Vec<[f64; 2]>,  // Running estimate of where the car was at each time step
    pub lhat: Vec<[f64; 2]>,  // Running estimate of where each landmark is
    pub color: Vec<u8>,     // Color of the landmarks
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
    ///     dclip (float, optional): distance at which to clip cost function for data association. Defaults to 0.5.
    #[pyo3(signature=(x0=[0.0, 0.0], max_landmark_distance=0.5, dx_weight=1.0, z_weight=5.0, dclip=0.5), text_signature="(x0: list[float] = [0.0, 0.0], max_landmark_distance: float = 0.5, dx_weight: float = 1.0, z_weight: float = 5.0, dclip: float = 0.5)")]
    pub fn new(
        x0: [f64; 2],
        max_landmark_distance: f64,
        dx_weight: f64,
        z_weight: f64,
        dclip: f64,
    ) -> Self {
        let x0 = [x0[0], x0[1]];
        let mut A = Trimat::new();
        let mut b = x0.to_vec();
        A.add_triplet(0, 0, 1.0);
        A.add_triplet(1, 1, 1.0);

        let mut xhat = vec![x0];

        GraphSLAMSolve {
            max_landmark_distance: max_landmark_distance,
            dx_weight: dx_weight,
            z_weight: z_weight,
            dclip: dclip,

            A: A,
            b: b,

            x: vec![0],
            l: vec![],
            z: vec![],
            d: vec![0],

            neqns: 2,
            nvars: 2,

            xhat: xhat,
            lhat: vec![],
            color: vec![],
        }
    }



    /// add edges to the graph corresponding to a movement and new vision data
    ///
    /// Args:
    ///     dx (ndarray or list): difference in position from last update. [dx, dy]
    ///     z (ndarray or list[list]): measurements to landmarks. [[zx1, zy1], [zx2, zy2], ..., [zxn, zyn]]
    ///     color (ndarray or list): categorical array of which color each of the measurements are. Elements should be dtype=np.uint8 or python ints.
    #[pyo3(signature = (dx, z, color), text_signature="($self, dx, z, color)")]
    pub fn update_graph(&mut self, dx: [f64; 2], z: Vec<[f64; 2]>, color: Vec<u8>) {

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
        let zprime = z.iter().map(|x| [x[0]+cur_xhat[0], x[1]+cur_xhat[1]]);

        // for running data assocition
        // hold off on this for now
        // let zprime = self.data_association(&vec![0., 0., 0.], &zprime.collect(), &color);

        // Perform data association
        // let zprime = self.data_association(z, &color);
        for (cone, c) in zip(zprime, color) {
            // get closest cone of same color
            // find it's index in `lhat` and `l`
            let (mut l_idx, min_dist, measurement) = match self.lhat.iter()
                .zip(&self.color)
                .enumerate()
                .filter(|x| *(x.1).1==c)
                .map(|x| (x.0, (x.1.0[0] - cone[0]).powi(2) + (x.1.0[1] - cone[1]).powi(2), x.1.0))
                .min_by(|a, b: &(usize, f64, &[f64; 2])| (&a.1).partial_cmp(&b.1).unwrap()) {
                    Some((l_idx, min_dist, measurement)) => (l_idx, min_dist, cone),
                    None => (0usize, f64::INFINITY, cone), // if there were no cones with the same color
            };
            // add landmark if it's new
            if min_dist > self.max_landmark_distance {
                self.l.push(self.nvars);
                self.nvars += 2;
                self.lhat.push(cone);
                self.color.push(c);
                l_idx = self.lhat.len()-1;
            }
            self.z.push(self.neqns);
            self.neqns += 2;
            self.A.add_triplet(*self.z.last().unwrap(),   self.l[l_idx],             self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap()+1, self.l[l_idx]+1,           self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap(),   *self.x.last().unwrap(),   -self.z_weight);
            self.A.add_triplet(*self.z.last().unwrap()+1, *self.x.last().unwrap()+1, -self.z_weight);
            self.b.push((measurement[0]-self.xhat.last().unwrap()[0])*self.z_weight);
            self.b.push((measurement[1]-self.xhat.last().unwrap()[1])*self.z_weight);
        }
    }

    /// print a visual representation of the A matrix
    pub unsafe fn printmat(&mut self) {
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        let mut b = self.b.clone();
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

    pub fn rows(&mut self) -> Vec<i32> { self.A.rows().clone() }
    pub fn cols(&mut self) -> Vec<i32> { self.A.cols().clone() }
    pub fn vals(&mut self) -> Vec<f64> { self.A.vals().clone() }
    pub fn nrows(&mut self) -> i32 { self.A.nrows }
    pub fn ncols(&mut self) -> i32 { self.A.ncols }
    pub fn neqns(&mut self) -> usize { self.neqns }
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
        let mut atb = vec![0.0; self.b.len()];

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
        // use cholesky decomposition to solve A.T@A \ A.T@b
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        let mut b = self.b.clone();
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

            let _ = cs_gaxpy(at2, b.as_ptr(), atb.as_mut_ptr());

            cs_cholsol(ata, atb.as_mut_ptr(), 0);
            // cs_lusol(ata, atb.as_mut_ptr(), 0, 1e-14);
        }
        // println!("{:?}", b[0..(self.A.ncols as usize)].to_vec());

        for (idx, i) in enumerate(&self.x) {
            self.xhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }
        for (idx, i) in enumerate(&self.l) {
            self.lhat[idx].copy_from_slice(&atb[*i..(i+2)]);
        }

    }

    /// solve the graph using the LU decomposition of A.T@A.
    /// This is barely slower than the cholesky method.
    /// solves in-place; does not return results.
    #[pyo3(signature = (tol=1e-14))]
    pub fn solve_graph_lusol(&mut self, tol: f64) {
        let mut ra = self.A.rows().clone();
        let mut ca = self.A.cols().clone();
        let mut va = self.A.vals().clone();
        let mut b = self.b.clone();
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
            let _ = cs_gaxpy(at2, b.as_ptr(), atb.as_mut_ptr());

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