use crate::ffi;
use std::{marker::PhantomData, num::NonZeroUsize};

/// Frees one kind of C++ object that Rust owns.
pub(crate) trait Resource {
    fn free(ptr: usize);
}

/// A C++ object owned by Rust, which is freed on drop.
///
/// The pointer is never 0, so a failed creation is `None` and `Option<Owned<R>>` is as small as a
/// `usize`.
pub(crate) struct Owned<R: Resource>(NonZeroUsize, PhantomData<*mut R>);

// an owned object is private to the thread that holds it. It is `Send` but not `Sync`, so no two
// threads can use one object at the same time.
unsafe impl<R: Resource> Send for Owned<R> {}

impl<R: Resource> Owned<R> {
    /// Takes ownership of `ptr`, or returns `None` for 0, which C++ uses for a failure.
    pub(crate) fn from_raw(ptr: usize) -> Option<Self> {
        NonZeroUsize::new(ptr).map(|ptr| Self(ptr, PhantomData))
    }

    /// Returns the pointer without giving up ownership.
    pub(crate) fn as_raw(&self) -> usize {
        self.0.get()
    }

    /// Gives up ownership, for C++ or Python to free.
    pub(crate) fn into_raw(self) -> usize {
        let ptr = self.0.get();
        std::mem::forget(self);
        ptr
    }
}

impl<R: Resource> Drop for Owned<R> {
    fn drop(&mut self) {
        R::free(self.0.get());
    }
}

/// Marks an `ROMol`.
pub(crate) enum MolKind {}

impl Resource for MolKind {
    fn free(ptr: usize) {
        ffi::free_mol_raw(ptr);
    }
}

/// A molecule owned by Rust.
pub(crate) type OwnedMol = Owned<MolKind>;
