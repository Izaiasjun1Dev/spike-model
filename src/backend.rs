//! Read backends. Each one owns `slots × topk` destination buffers: a slot is
//! one MoE layer's worth of experts, and having several slots lets the driver
//! keep reads for upcoming layers in flight (lookahead / prefetch).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::data::{self, AlignedBuf};

pub trait Backend {
    /// Starts loading `offsets[i]` into buffer `i` of `slot`.
    fn submit(&mut self, slot: usize, offsets: &[u64]) -> Result<(), String>;
    /// Blocks until every read submitted to `slot` has landed.
    fn wait(&mut self, slot: usize) -> Result<(), String>;
    fn buffer(&self, slot: usize, idx: usize) -> &[u8];
}

#[derive(Clone, Copy)]
pub enum Source {
    /// pread on a file opened with F_NOCACHE (macOS) / O_DIRECT (Linux).
    Pread,
    /// memcpy out of an mmap of the whole file; goes through the page cache.
    Mmap,
}

struct Mapping {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut _, self.len) };
    }
}

enum Reader {
    File(File),
    Map(Mapping),
}

impl Reader {
    fn read(&self, dst: &mut [u8], offset: u64) -> Result<(), String> {
        match self {
            Reader::File(f) => f.read_exact_at(dst, offset).map_err(|e| e.to_string()),
            Reader::Map(m) => {
                let end = offset as usize + dst.len();
                if end > m.len {
                    return Err(format!("read past end of mapping ({end} > {})", m.len));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        m.ptr.add(offset as usize),
                        dst.as_mut_ptr(),
                        dst.len(),
                    )
                };
                Ok(())
            }
        }
    }
}

#[derive(Default)]
struct SlotState {
    pending: usize,
    error: Option<String>,
}

struct Job {
    slot: usize,
    dst: *mut u8,
    len: usize,
    offset: u64,
}

// `dst` points into a buffer the driver does not touch until `wait` returns.
unsafe impl Send for Job {}

/// pread / mmap on a fixed pool of worker threads.
pub struct ThreadPool {
    buffers: Vec<Vec<AlignedBuf>>,
    expert_bytes: usize,
    slots: Arc<Vec<(Mutex<SlotState>, Condvar)>>,
    tx: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(
        path: &Path,
        source: Source,
        threads: usize,
        slots: usize,
        topk: usize,
        expert_bytes: usize,
    ) -> Result<Self, String> {
        let reader = Arc::new(match source {
            Source::Pread => {
                Reader::File(data::open_uncached(path, false).map_err(|e| e.to_string())?)
            }
            Source::Mmap => {
                let file = File::open(path).map_err(|e| e.to_string())?;
                let len = file.metadata().map_err(|e| e.to_string())?.len() as usize;
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        len,
                        libc::PROT_READ,
                        libc::MAP_SHARED,
                        std::os::fd::AsRawFd::as_raw_fd(&file),
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    return Err(std::io::Error::last_os_error().to_string());
                }
                // Expert access is random; don't let the kernel read ahead.
                unsafe { libc::madvise(ptr, len, libc::MADV_RANDOM) };
                Reader::Map(Mapping {
                    ptr: ptr as *const u8,
                    len,
                })
            }
        });

        let buffers = (0..slots)
            .map(|_| (0..topk).map(|_| AlignedBuf::new(expert_bytes)).collect())
            .collect();
        let states: Arc<Vec<(Mutex<SlotState>, Condvar)>> =
            Arc::new((0..slots).map(|_| Default::default()).collect());
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));

        let workers = (0..threads)
            .map(|_| {
                let rx = Arc::clone(&rx);
                let reader = Arc::clone(&reader);
                let states = Arc::clone(&states);
                std::thread::spawn(move || {
                    loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        };
                        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst, job.len) };
                        let result = reader.read(dst, job.offset);
                        let (lock, cv) = &states[job.slot];
                        let mut st = lock.lock().unwrap();
                        if let Err(e) = result {
                            st.error
                                .get_or_insert(format!("read at {:#x}: {e}", job.offset));
                        }
                        st.pending -= 1;
                        if st.pending == 0 {
                            cv.notify_one();
                        }
                    }
                })
            })
            .collect();

        Ok(Self {
            buffers,
            expert_bytes,
            slots: states,
            tx: Some(tx),
            workers,
        })
    }
}

impl Backend for ThreadPool {
    fn submit(&mut self, slot: usize, offsets: &[u64]) -> Result<(), String> {
        self.slots[slot].0.lock().unwrap().pending += offsets.len();
        let tx = self.tx.as_ref().unwrap();
        for (i, &offset) in offsets.iter().enumerate() {
            let dst = self.buffers[slot][i].as_ptr();
            tx.send(Job {
                slot,
                dst,
                len: self.expert_bytes,
                offset,
            })
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn wait(&mut self, slot: usize) -> Result<(), String> {
        let (lock, cv) = &self.slots[slot];
        let mut st = cv
            .wait_while(lock.lock().unwrap(), |s| s.pending > 0)
            .unwrap();
        st.error.take().map_or(Ok(()), Err)
    }

    fn buffer(&self, slot: usize, idx: usize) -> &[u8] {
        self.buffers[slot][idx].as_slice()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.tx.take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
