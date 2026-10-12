//! Test file generation, aligned buffers and the verification pattern.
//!
//! The file is a flat pool of 4 KiB blocks. Each block starts with its own
//! byte offset (u64 LE) followed by pseudo-random, incompressible bytes, so a
//! read can be checked for landing at the right place without storing a copy.

use std::alloc::{self, Layout};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

/// Apple Silicon page size. Every offset and length is a multiple of this so
/// uncached I/O (F_NOCACHE / O_DIRECT) never falls back to a bounce buffer.
pub const ALIGN: usize = 16 * 1024;
pub const BLOCK: usize = 4096;

pub fn align_up(n: u64) -> u64 {
    n.div_ceil(ALIGN as u64) * ALIGN as u64
}

pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// The buffer is plain memory; synchronisation is the caller's job.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, ALIGN).expect("buffer layout");
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self { ptr, len }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, ALIGN).unwrap();
        unsafe { alloc::dealloc(self.ptr, layout) };
    }
}

/// Opens a file so reads bypass the OS page cache.
/// macOS: F_NOCACHE. Linux: O_DIRECT (lets the benchmark be smoke-tested off-Mac).
pub fn open_uncached(path: &Path, write: bool) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    if write {
        opts.write(true).create(true).truncate(true);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_DIRECT);
    }
    let file = opts.open(path)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(file)
}

fn fill_block(block: &mut [u8], offset: u64) {
    block[..8].copy_from_slice(&offset.to_le_bytes());
    let mut s = offset ^ 0x9E37_79B9_7F4A_7C15;
    for chunk in block[8..].chunks_exact_mut(8) {
        // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        chunk.copy_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
}

/// Checks that `buf` holds the data stored at file offset `offset`.
/// Only block headers are compared, which is enough to catch wrong offsets,
/// short reads and untouched buffers.
pub fn verify(buf: &[u8], offset: u64) -> Result<(), String> {
    for (i, block) in buf.chunks_exact(BLOCK).enumerate() {
        let want = offset + (i * BLOCK) as u64;
        let got = u64::from_le_bytes(block[..8].try_into().unwrap());
        if got != want {
            return Err(format!("block at {want:#x} holds header {got:#x}"));
        }
    }
    Ok(())
}

pub fn generate(path: &Path, size: u64) -> io::Result<()> {
    let size = align_up(size);
    let chunk = 64 * 1024 * 1024;
    let mut buf = AlignedBuf::new(chunk);
    let mut file = open_uncached(path, true)?;
    let start = Instant::now();
    let mut written = 0u64;
    while written < size {
        let n = chunk.min((size - written) as usize);
        for (i, block) in buf.as_mut_slice()[..n].chunks_exact_mut(BLOCK).enumerate() {
            fill_block(block, written + (i * BLOCK) as u64);
        }
        file.write_all(&buf.as_slice()[..n])?;
        written += n as u64;
        let secs = start.elapsed().as_secs_f64();
        eprint!(
            "\r{:.1} / {:.1} GiB ({:.2} GB/s)",
            written as f64 / GIB,
            size as f64 / GIB,
            written as f64 / secs / 1e9
        );
    }
    file.sync_all()?;
    eprintln!();
    Ok(())
}

pub const GIB: f64 = (1u64 << 30) as f64;

/// Percentage of `path` currently held in the OS page cache (mmap + mincore).
/// Mapping without touching the pages does not fault anything in, so this
/// does not disturb what it measures.
pub fn resident_pct(path: &Path) -> io::Result<f64> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Ok(0.0);
    }
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut vec = vec![0u8; len.div_ceil(page)];
    let rc = unsafe { libc::mincore(ptr, len, vec.as_mut_ptr() as _) };
    let err = io::Error::last_os_error();
    unsafe { libc::munmap(ptr, len) };
    if rc != 0 {
        return Err(err);
    }
    // Bit 0 is "resident" on both macOS (MINCORE_INCORE) and Linux.
    let resident = vec.iter().filter(|&&b| b & 1 != 0).count();
    Ok(100.0 * resident as f64 / vec.len() as f64)
}

pub fn physical_memory() -> u64 {
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page = libc::sysconf(libc::_SC_PAGESIZE);
        if pages <= 0 || page <= 0 {
            0
        } else {
            pages as u64 * page as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_round_trip() {
        let mut buf = vec![0u8; 4 * BLOCK];
        for (i, b) in buf.chunks_exact_mut(BLOCK).enumerate() {
            fill_block(b, 0x10000 + (i * BLOCK) as u64);
        }
        assert!(verify(&buf, 0x10000).is_ok());
        assert!(verify(&buf, 0x20000).is_err());
    }

    #[test]
    fn resident_pct_in_range() {
        let dir = std::env::temp_dir().join(format!("expert-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty");
        let full = dir.join("full");
        std::fs::write(&empty, b"").unwrap();
        std::fs::write(&full, vec![7u8; 4 * ALIGN]).unwrap();
        assert_eq!(resident_pct(&empty).unwrap(), 0.0);
        let pct = resident_pct(&full).unwrap();
        assert!((0.0..=100.0).contains(&pct), "{pct}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn align_up_rounds_to_page() {
        assert_eq!(align_up(1), ALIGN as u64);
        assert_eq!(align_up(ALIGN as u64), ALIGN as u64);
        assert_eq!(align_up(ALIGN as u64 + 1), 2 * ALIGN as u64);
    }
}
