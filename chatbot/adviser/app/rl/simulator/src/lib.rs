use pyo3::prelude::*;
mod data;
mod simulator;
mod parsers;

use pyo3::prelude::*;

extern crate pest;
#[macro_use]
extern crate pest_derive;


// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn simulator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}


