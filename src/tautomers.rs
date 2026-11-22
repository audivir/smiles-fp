use crate::{
    ffi,
    handle::OwnedMol,
    pool::{Cancelled, run_pool_verbose, setup_local_pool},
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

/// Tautomers of one input: SMILES or the number of molecules in the list of all of them.
struct Group {
    smiles: Vec<String>,
    n_mols: usize,
    status: &'static str,
}

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
    remove_bond_stereo=true, remove_isotopic_hs=true, reassign_stereo=true, v1=false, n_threads=-1, verbose=false,
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
    verbose: bool,
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
        verbose,
        "Canonical tautomers",
        true,
        || tautomerizer(&opts),
        |t, mol| OwnedMol::from_raw(ffi::tautomer_raw(t.0, mol.as_raw())),
    )
}

/// Enumerates all tautomers of SMILES or molecules on real threads.
///
/// Returns a tuple of the tautomers and the status of the enumeration per input.
#[pyfunction]
#[pyo3(signature = (
    inputs, output="smiles", *, max_tautomers=1000, max_transforms=1000, remove_sp3_stereo=true,
    remove_bond_stereo=true, remove_isotopic_hs=true, reassign_stereo=true, v1=false, n_threads=-1, verbose=false,
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
    verbose: bool,
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
        inputs,
        n_threads,
        verbose,
        "Enumerating tautomers",
        || tautomerizer(&opts),
        |t, mol| {
            let found = ffi::enumerate_raw(t.0, mol.as_raw());
            let items: Vec<Output> = found
                .ptrs
                .iter()
                .map(|&p| finish(OwnedMol::from_raw(p), want_mol, false))
                .collect();
            (items, found.status)
        },
    )?;

    // a failure of RDKit has the status 255 and invalidates the input like a parse error.
    let mut ptrs = Vec::new();
    let mut groups: Vec<Option<Group>> = Vec::with_capacity(results.len());
    for result in results {
        groups.push(result.and_then(|(items, status)| {
            let status = *STATUS.get(usize::from(status))?;
            let mut group = Group {
                smiles: Vec::new(),
                n_mols: 0,
                status,
            };
            for item in items {
                match item {
                    Output::Smiles(s) => group.smiles.push(s),
                    Output::Mol(mol) => {
                        ptrs.push(mol.into_raw());
                        group.n_mols += 1;
                    }
                    Output::Invalid => {}
                }
            }
            Some(group)
        }));
    }
    let wrapped = if want_mol {
        Some(wrap_ptrs(py, &ptrs)?)
    } else {
        None
    };

    let mut out = Vec::with_capacity(groups.len());
    let mut start = 0;
    for group in groups {
        let Some(Group {
            smiles,
            n_mols,
            status,
        }) = group
        else {
            out.push(py.None());
            continue;
        };
        let tautomers: Bound<'py, PyAny> = match &wrapped {
            Some(all) => all.get_slice(start, start + n_mols).into_any(),
            None => PyList::new(py, smiles)?.into_any(),
        };
        start += n_mols;
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
#[pyo3(signature = (inputs, *, n_threads=-1, verbose=false))]
pub fn score_tautomers<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    n_threads: i32,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(inputs) = read_inputs(py, inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };
    let scores = map_inputs(
        py,
        inputs,
        n_threads,
        verbose,
        "Scoring tautomers",
        || (),
        |_, mol| ffi::score_raw(mol.as_raw()).ok(),
    )?;
    Ok(PyList::new(py, scores.into_iter().map(Option::flatten))?.into_any())
}

/// Picks the best tautomer of every group on real threads like `PickCanonical`.
///
/// A group is invalid if it is empty or has an invalid item.
#[pyfunction]
#[pyo3(signature = (groups, output="smiles", *, n_threads=-1, verbose=false))]
pub fn pick_canonical_tautomers<'py>(
    py: Python<'py>,
    groups: &Bound<'py, PySequence>,
    output: &str,
    n_threads: i32,
    verbose: bool,
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
    let total = parts.len();
    let results: Vec<Output> = run_pool_verbose(
        py,
        &pool,
        verbose,
        total,
        "Picking canonical tautomers",
        true,
        |progress| {
            parts
                .into_par_iter()
                .map_init(|| tautomerizer(&opts), |t, part| pick(t, part, want_mol))
                .map(|r| progress.track(r))
                .collect::<Result<Vec<_>, Cancelled>>()
        },
    )?;
    outputs_to_py(py, results, want_mol)
}

/// Picks the best tautomer from the items of one group.
fn pick(t: &Tautomerizer, part: Inputs, want_mol: bool) -> Output {
    // an invalid item makes the group invalid, so parsing stops at the first one.
    let mols: Option<Vec<OwnedMol>> = match part {
        Inputs::Mols(copies) => Some(copies),
        Inputs::Smiles(smiles) => smiles
            .iter()
            .map(|s| {
                let s = s.as_ref().filter(|s| !s.is_empty())?;
                OwnedMol::from_raw(ffi::parse_smiles_raw(s, true))
            })
            .collect(),
    };
    let best = mols.filter(|m| !m.is_empty()).and_then(|mols| {
        let ptrs: Vec<usize> = mols.iter().map(OwnedMol::as_raw).collect();
        OwnedMol::from_raw(ffi::pick_canonical_raw(t.0, &ptrs))
    });
    finish(best, want_mol, true)
}
