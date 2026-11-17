use crate::{
    ffi,
    pool::{run_pool, setup_local_pool},
    transform::{
        Inputs, Output, finish, map_inputs, outputs_to_py, read_inputs, transform, want_mol,
        wrap_ptrs,
    },
};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyAny, PyList, PySequence, PyTuple},
};
use rayon::prelude::*;

/// Owner of the tautomer enumerator of one thread, which is freed on drop.
struct Tautomerizer(usize);

impl Drop for Tautomerizer {
    fn drop(&mut self) {
        ffi::free_tautomerizer(self.0);
    }
}

/// Names of the values of `TautomerEnumeratorStatus`.
const STATUS: [&str; 4] = [
    "completed",
    "max_tautomers_reached",
    "max_transforms_reached",
    "canceled",
];

/// Builds the enumerator of a thread, which cannot fail for valid limits.
fn tautomerizer(opts: &ffi::TautomerOpts) -> Tautomerizer {
    Tautomerizer(ffi::make_tautomerizer(opts))
}

/// Computes canonical tautomers of SMILES or molecules on real threads.
///
/// Each thread builds its own enumerator. Molecules are copied first, serially, so the inputs stay
/// unchanged. Releases the GIL for the computation.
#[pyfunction]
#[pyo3(signature = (
    inputs, output="smiles", *, max_tautomers=1000, max_transforms=1000, remove_sp3_stereo=true,
    remove_bond_stereo=true, remove_isotopic_hs=true, reassign_stereo=true, v1=false, n_threads=-1,
))]
#[allow(clippy::too_many_arguments)]
pub fn canonical_tautomers<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    max_tautomers: u32,
    max_transforms: u32,
    remove_sp3_stereo: bool,
    remove_bond_stereo: bool,
    remove_isotopic_hs: bool,
    reassign_stereo: bool,
    v1: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let opts = ffi::TautomerOpts {
        max_tautomers,
        max_transforms,
        remove_sp3_stereo,
        remove_bond_stereo,
        remove_isotopic_hs,
        reassign_stereo,
        v1,
    };
    transform(
        py,
        inputs,
        output,
        n_threads,
        true,
        || tautomerizer(&opts),
        |t, mol| ffi::tautomer_raw(t.0, mol),
    )
}

/// Enumerates all tautomers of SMILES or molecules on real threads.
///
/// Returns a tuple of the tautomers and the status of the enumeration per input.
#[pyfunction]
#[pyo3(signature = (
    inputs, output="smiles", *, max_tautomers=1000, max_transforms=1000, remove_sp3_stereo=true,
    remove_bond_stereo=true, remove_isotopic_hs=true, reassign_stereo=true, v1=false, n_threads=-1,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_tautomers<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    max_tautomers: u32,
    max_transforms: u32,
    remove_sp3_stereo: bool,
    remove_bond_stereo: bool,
    remove_isotopic_hs: bool,
    reassign_stereo: bool,
    v1: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let want_mol = want_mol(output)?;
    let opts = ffi::TautomerOpts {
        max_tautomers,
        max_transforms,
        remove_sp3_stereo,
        remove_bond_stereo,
        remove_isotopic_hs,
        reassign_stereo,
        v1,
    };
    let Some(inputs) = read_inputs(py, inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };
    let results = map_inputs(
        py,
        &inputs,
        n_threads,
        || tautomerizer(&opts),
        |t, mol| {
            let found = ffi::enumerate_raw(t.0, mol);
            let items: Vec<Output> = found
                .ptrs
                .iter()
                .map(|&p| finish(p, want_mol, false))
                .collect();
            (items, found.status)
        },
    )?;

    // a failure of RDKit has the status 255 and invalidates the input like a parse error.
    let results: Vec<Option<(Vec<Output>, &str)>> = results
        .into_iter()
        .map(|r| {
            let (items, status) = r?;
            let name = STATUS.get(usize::from(status))?;
            Some((items, *name))
        })
        .collect();

    let mut offsets = vec![0];
    let mut ptrs = Vec::new();
    for (items, _) in results.iter().flatten() {
        for item in items {
            if let Output::Mol(p) = item {
                ptrs.push(*p);
            }
        }
        offsets.push(ptrs.len());
    }
    let wrapped = if want_mol {
        Some(wrap_ptrs(py, &ptrs)?)
    } else {
        None
    };

    let mut out = Vec::with_capacity(results.len());
    let mut group = 0;
    for result in results {
        let Some((items, status)) = result else {
            out.push(py.None());
            continue;
        };
        let tautomers: Bound<'py, PyAny> = match &wrapped {
            Some(all) => all.get_slice(offsets[group], offsets[group + 1]).into_any(),
            None => {
                let smiles: Vec<String> = items
                    .into_iter()
                    .filter_map(|i| match i {
                        Output::Smiles(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                PyList::new(py, smiles)?.into_any()
            }
        };
        group += 1;
        out.push(
            PyTuple::new(py, [tautomers, status.into_pyobject(py)?.into_any()])?
                .into_any()
                .unbind(),
        );
    }
    Ok(PyList::new(py, out)?.into_any())
}

/// Scores tautomers of SMILES or molecules on real threads like `ScoreTautomer`.
#[pyfunction]
#[pyo3(signature = (inputs, *, n_threads=-1))]
pub fn score_tautomers<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(inputs) = read_inputs(py, inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };
    let scores = map_inputs(
        py,
        &inputs,
        n_threads,
        || (),
        |_, mol| ffi::score_raw(mol).ok(),
    )?;
    Ok(PyList::new(py, scores.into_iter().map(Option::flatten))?.into_any())
}

/// Picks the best tautomer of every group on real threads like `PickCanonical`.
///
/// A group is invalid if it is empty or has an invalid item.
#[pyfunction]
#[pyo3(signature = (groups, output="smiles", *, n_threads=-1))]
pub fn pick_canonical_tautomers<'py>(
    py: Python<'py>,
    groups: &Bound<'py, PySequence>,
    output: &str,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let want_mol = want_mol(output)?;
    let mut parts: Vec<Inputs> = Vec::with_capacity(groups.len()?);
    for ix in 0..groups.len()? {
        let group = groups.get_item(ix)?.cast_into::<PySequence>()?;
        // an empty group is empty SMILES, which is invalid like the empty molecule list.
        parts.push(read_inputs(py, &group)?.unwrap_or(Inputs::Smiles(Vec::new())));
    }
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let opts = ffi::TautomerOpts {
        max_tautomers: 1000,
        max_transforms: 1000,
        remove_sp3_stereo: true,
        remove_bond_stereo: true,
        remove_isotopic_hs: true,
        reassign_stereo: true,
        v1: false,
    };
    let results: Vec<Output> = run_pool(py, &pool, || {
        parts
            .par_iter()
            .map_init(|| tautomerizer(&opts), |t, part| pick(t, part, want_mol))
            .collect()
    });
    outputs_to_py(py, results, want_mol)
}

/// Picks the best tautomer from the items of one group.
fn pick(t: &Tautomerizer, part: &Inputs, want_mol: bool) -> Output {
    let parsed: Vec<usize>;
    let ptrs: &[usize] = match part {
        Inputs::Mols(copies) => &copies.0,
        Inputs::Smiles(smiles) => {
            let mols: Vec<usize> = smiles
                .iter()
                .map(|s| match s.as_ref().filter(|s| !s.is_empty()) {
                    Some(s) => ffi::parse_smiles_raw(s, true),
                    None => 0,
                })
                .collect();
            parsed = mols;
            &parsed
        }
    };
    let best = if ptrs.is_empty() || ptrs.contains(&0) {
        0
    } else {
        ffi::pick_canonical_raw(t.0, ptrs)
    };
    if matches!(part, Inputs::Smiles(_)) {
        for &p in ptrs.iter().filter(|&&p| p != 0) {
            ffi::free_mol_raw(p);
        }
    }
    finish(best, want_mol, true)
}
